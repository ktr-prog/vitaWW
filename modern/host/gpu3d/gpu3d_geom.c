
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "gpu3d_internal.h"
#include "../savestate.h"

static int sGeomOn, sRenderOn;
static u32 sDispCnt;
static u32 sGxStat;
static u16 sRegs[0xa0 / 2];
static u16 sDot1Depth = 0x7fff;
u8 gG3VramCnt[9];
u8 gG3VramCntFrame[9];
const uint8_t *const *gG3BankDirtyFrame;
static const uint8_t *const *sBankDirtyNext;
void gpu3d_set_bank_dirty(const uint8_t *const dirty[9]) { sBankDirtyNext = dirty; }

typedef struct { s32 m[16]; } Mtx;
static int sMtxMode;
static Mtx sProj, sPos, sVec, sTex, sClip;
static Mtx sProjStack, sTexStack, sPosStack[32], sVecStack[32];
static int sProjSP, sTexSP, sPosSP;
static int sClipDirty = 1;
// GX worker thread (see "GX worker" below). When sMT is on, the game thread keeps only the matrix stack (so GXSTAT,
// matrix reads and POS/VEC tests are answered locally) and every other command executes on a worker thread, which
// reads its matrices from the snapshots the game thread sends (wClip/wVec/wTex) instead of the live ones.
static int sMT;
static Mtx wClip, wVec, wTex;
static const Mtx *gClipM = &sClip, *gVecM = &sVec, *gTexM = &sTex;
static void mt_entry(u8 cmd, u32 param);

static s16 sVtx[3];
static u8 sVtxColor[3];
static s16 sTexRaw[2], sTexCoord[2];
static s16 sNormal[3];
static s16 sLightDir[4][3];
static u32 sLightParam[4]; static s32 sLightVec[4][9];
static u8 sLightColor[4][3];
static u8 sDiffuse[3], sAmbient[3], sSpecular[3], sEmission[3];
// Per-light-per-channel colour coefficients (material*lightColour), recomputed only when the light state
// changes (per object), not per vertex - hoisting those multiplies out of the per-vertex lighting loop.
static s32 sPSpec[4][3], sPDif[4][3], sPAmb[4][3];
static int sLightCoefDirty = 1;
static int sShininessOn;
static u8 sShininess[128];
static u32 sPolyAttr, sCurPolyAttr, sTexParam, sTexPal;
static u32 sViewport[6];
static int sPolyType;
static int sInBegin;
static G3Vertex sTemp[4];
static int sVtxCount, sVtxInPoly, sStripPolys;
static G3Vertex *sStripRam[2];
static int sStripValid;
static s32 sPosResult[4];
static s16 sVecResult[3];

// GL fast path (gG3ClipSpaceOut without gG3GpuXform, i.e. the Vita): every vertex is computed once straight into
// a compact per-bank pool and polygons point into it - no 136-byte G3Vertex built, copied into a polygon-local
// array, copied again into vertex RAM and shifted down for strips. sNumVerts still counts vertex RAM as the
// hardware does (strip-shared vertices once, culled polygons' none) for the overflow limit; the pool only holds
// the data, so it has room for the culled polygons' vertices too.
enum { G3_POOL_N = MAX_VERTICES * 2 };
static G3VHot *sPool[2];
static int sPoolN;
static G3VHot sPoolSpill;          // target when the pool is full: the polygon is then dropped as an overflow
static const G3VHot *sTempH[4];
static int sFastGL = -1;           // -1: decide at the first vertex (the backend sets gG3ClipSpaceOut at init)
HOST_STATE_KEEP(sPool) HOST_STATE_KEEP(sFastGL)
static G3Vertex sVertexRam[2][MAX_VERTICES];
static G3Polygon sPolyRam[2][MAX_POLYGONS];
static int sBank;
static int sNumVerts, sNumPolys, sNumOpaque;
static int sRamOverflow;
G3RenderState gG3Render;
const uint8_t *const *gG3Banks;

static u64 sFeatureFrame[F_COUNT];
static u64 sFrame;
static int sFeaturesOn = -1;
void g3_feature(int id) {
    if (sFeaturesOn && !sFeatureFrame[id]) sFeatureFrame[id] = sFrame + 1;
}

typedef struct { u8 cmd; u32 param; } Entry;
static Entry sPipe[4], sFifo[256];
static int sPipeHead, sPipeN, sFifoHead, sFifoN;
static int sPushPopQueued, sTestQueued;
static int sFlushPending;
static u32 sFlushParam;

static const u8 kParams[256] = {
    [0x10] = 1, [0x11] = 0, [0x12] = 1, [0x13] = 1, [0x14] = 1, [0x15] = 0, [0x16] = 16, [0x17] = 12, [0x18] = 16,
    [0x19] = 12, [0x1a] = 9, [0x1b] = 3, [0x1c] = 3, [0x20] = 1, [0x21] = 1, [0x22] = 1, [0x23] = 2, [0x24] = 1,
    [0x25] = 1, [0x26] = 1, [0x27] = 1, [0x28] = 1, [0x29] = 1, [0x2a] = 1, [0x2b] = 1, [0x30] = 1, [0x31] = 1,
    [0x32] = 1, [0x33] = 1, [0x34] = 32, [0x40] = 1, [0x41] = 0, [0x50] = 1, [0x60] = 1, [0x70] = 3, [0x71] = 2,
    [0x72] = 1,
};
static int is_command(u8 c) {
    return (c >= 0x10 && c <= 0x1c) || (c >= 0x20 && c <= 0x2b) || (c >= 0x30 && c <= 0x34) || c == 0x40 ||
           c == 0x41 || c == 0x50 || c == 0x60 || (c >= 0x70 && c <= 0x72);
}

static void fifo_put(u8 cmd, u32 param) {
    if (sMT) { mt_entry(cmd, param); return; }
    Entry e = {cmd, param};
    if (cmd == 0x11 || cmd == 0x12) sPushPopQueued++;
    if (cmd >= 0x70 && cmd <= 0x72) sTestQueued++;
    if (sFifoN == 0 && sPipeN < 4) {
        sPipe[(sPipeHead + sPipeN++) & 3] = e;
    } else if (sFifoN < 256) {
        sFifo[(sFifoHead + sFifoN++) & 255] = e;
    } else {
        g3_feature(F_FIFO_FULL);
    }
    if (sFlushPending) g3_feature(F_FIFO_QUEUED);
    if (sFlushPending && cmd == 0x50) g3_feature(F_SWAP_QUEUED);
    if (!sGeomOn) g3_feature(F_GEOM_OFF_WRITE);
}
static Entry fifo_get(void) {
    Entry e = sPipe[sPipeHead];
    sPipeHead = (sPipeHead + 1) & 3;
    sPipeN--;
    if (sPipeN <= 2) {
        for (int i = 0; i < 2 && sFifoN; i++) {
            sPipe[(sPipeHead + sPipeN++) & 3] = sFifo[sFifoHead];
            sFifoHead = (sFifoHead + 1) & 255;
            sFifoN--;
        }
    }
    if (e.cmd == 0x11 || e.cmd == 0x12) sPushPopQueued--;
    if (e.cmd >= 0x70 && e.cmd <= 0x72) sTestQueued--;
    return e;
}

static s32 sVtxPipe, sNrmPipe, sPolyPipe, sSlotCtr;
static u32 sSlots = 1;
static int sMtxModeForCost;
static int sPolyDone;
static int sPolyOutcome, sPolyNOut, sPolyClipped, sFarHidden;
static unsigned sPolySerial;
int gpu3d_debug_last_polygon(int *nout, int *clipped, unsigned *serial, int *type) { *nout = sPolyNOut; *clipped = sPolyClipped; *serial = sPolySerial; *type = sPolyType | (sStripPolys > 1) << 2; return sPolyOutcome; }
static void add_cycles(s32 n) {
    sVtxPipe = n < sVtxPipe ? sVtxPipe - n : 0;
    if (n < sPolyPipe) {
        sPolyPipe -= n;
        sSlotCtr += n;
        while (sSlotCtr > 9) { sSlots >>= 1; sSlotCtr -= 9; }
    } else {
        sPolyPipe = 0;
        sSlots = 1;
        sSlotCtr = 0;
    }
}
static void next_vertex_slot(void) {
    while (!(sSlots & 1)) add_cycles(10 - sSlotCtr);
    if (sPolyPipe) sSlots &= ~1u;
}
static int is_vertex_cmd(u8 c) { return c >= 0x23 && c <= 0x28; }
static void vertex_entry_first(void) {
    if (!(sSlots & 1)) next_vertex_slot();
    else add_cycles(1);
    sNrmPipe = 0;
}
static void timing_before(u8 c) {
    sMtxModeForCost = sMtxMode;
    if (is_vertex_cmd(c)) {
        if (c != 0x23) vertex_entry_first();
        return;
    }
    switch (c) {
    case 0x20: case 0x30: case 0x31: case 0x72:
        add_cycles(sVtxPipe > 2 ? sVtxPipe - 2 + 1 : sNrmPipe + 1);
        break;
    case 0x29: case 0x2a: case 0x2b: case 0x33: case 0x34: case 0x41: case 0x60: case 0x71:
        add_cycles(sVtxPipe > 0 ? sVtxPipe + 1 : sNrmPipe + 1);
        break;
    default:
        add_cycles(sNrmPipe + 1);
        break;
    }
    sNrmPipe = 0;
}

static s32 command_cycles(u8 c) {
    switch (c) {
    case 0x11: case 0x13: return 16;
    case 0x12: case 0x14: return 35;
    case 0x15: return sMtxModeForCost == 3 ? 0 : 18;
    case 0x16: return sMtxModeForCost == 3 ? 10 : 18;
    case 0x17: return sMtxModeForCost == 3 ? 10 : 18;
    case 0x18: return 53;
    case 0x19: return 53;
    case 0x1a: return 56;
    case 0x1b: return 32;
    case 0x1c: return 62;
    case 0x21: return 4;
    case 0x30: case 0x31: return 3;
    case 0x32: return 5;
    case 0x33: return 1;
    case 0x50: return 324;
    default: return 0;
    }
}
static void timing_after(u8 c) {
    if (is_vertex_cmd(c)) {
        add_cycles(c == 0x23 ? 4 : 3);
        sVtxPipe = 4;
        if (sPolyDone) {
            sPolyDone = 0;
            int strip = sPolyType >= 2, quad = sPolyNOut == 4;
            if (sPolyOutcome) { sPolyPipe = 5; sSlots = 0x1e; }
            else if (quad) { sPolyPipe = 32; sSlots = strip ? 0x1c : 0x1e; }
            else { sPolyPipe = 23; sSlots = strip ? 0x08 : 0x0e; }
            sSlotCtr = 4;
        }
        return;
    }
    if (c == 0x40) add_cycles(sPolyPipe);
    add_cycles(command_cycles(c));
    if (c == 0x21) sNrmPipe = 7;
}
static int pipelines_busy(void) { return sVtxPipe || sNrmPipe || sPolyPipe; }

static u32 sPackCmds;
static int sPackLeft, sPackParamsLeft;
static void pack_advance(void) {
    while (sPackLeft > 0) {
        u8 c = sPackCmds & 0xff;
        if (sPackCmds == 0) { sPackLeft = 0; break; }
        if (kParams[c]) { sPackParamsLeft = kParams[c]; return; }
        if (c) fifo_put(c, 0);
        sPackCmds >>= 8;
        sPackLeft--;
    }
    sPackParamsLeft = 0;
}
static void gxfifo_write(u32 v) {
    if (sPackParamsLeft == 0) {
        if (v == 0) { fifo_put(0, 0); return; }
        sPackCmds = v;
        sPackLeft = 4;
        pack_advance();
    } else {
        fifo_put(sPackCmds & 0xff, v);
        if (--sPackParamsLeft == 0) {
            sPackCmds >>= 8;
            sPackLeft--;
            pack_advance();
        }
    }
}

static void mtx_identity(Mtx *a) {
    memset(a, 0, sizeof *a);
    a->m[0] = a->m[5] = a->m[10] = a->m[15] = 0x1000;
}
static void mtx_mult(Mtx *cur, const Mtx *a) {
    Mtx r;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            s64 s = 0;
            for (int k = 0; k < 4; k++) s += (s64)a->m[i * 4 + k] * cur->m[k * 4 + j];
            r.m[i * 4 + j] = (s32)(s >> 12);
        }
    *cur = r;
}
static void mtx_scale(Mtx *cur, const s32 *p) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++) cur->m[i * 4 + j] = (s32)(((s64)p[i] * cur->m[i * 4 + j]) >> 12);
}
static void mtx_trans(Mtx *cur, const s32 *p) {
    for (int j = 0; j < 4; j++)
        cur->m[12 + j] += (s32)(((s64)p[0] * cur->m[j] + (s64)p[1] * cur->m[4 + j] + (s64)p[2] * cur->m[8 + j]) >> 12);
}
static u32 sClipSerial;   // bumped whenever the clip matrix is recomputed, so consumers can cache by identity
static void update_clip(void) {
    if (!sClipDirty) return;
    sClip = sProj;
    mtx_mult(&sClip, &sPos);
    sClipDirty = 0;
    sClipSerial++;
}

static void apply_mult(const Mtx *a) {
    switch (sMtxMode) {
    case 0: mtx_mult(&sProj, a); sClipDirty = 1; break;
    case 1: mtx_mult(&sPos, a); sClipDirty = 1; break;
    case 2: mtx_mult(&sPos, a); mtx_mult(&sVec, a); sClipDirty = 1; break;
    default: mtx_mult(&sTex, a); break;
    }
}
static void apply_load(const Mtx *a) {
    switch (sMtxMode) {
    case 0: sProj = *a; sClipDirty = 1; break;
    case 1: sPos = *a; sClipDirty = 1; break;
    case 2: sPos = *a; sVec = *a; sClipDirty = 1; break;
    default: sTex = *a; break;
    }
}

static void stack_error(void) {
    sGxStat |= 1u << 15;
    g3_feature(F_STACK_ERROR);
}
static void mtx_push(void) {
    if (sMtxMode == 0) {
        if (sProjSP) stack_error();
        sProjStack = sProj;
        sProjSP ^= 1;
    } else if (sMtxMode == 3) {
        if (sTexSP) stack_error();
        sTexStack = sTex;
        sTexSP ^= 1;
    } else {
        if (sPosSP >= 31) stack_error();
        sPosStack[sPosSP & 31] = sPos;
        sVecStack[sPosSP & 31] = sVec;
        sPosSP = (sPosSP + 1) & 63;
    }
}
static void mtx_pop(u32 p) {
    if (sMtxMode == 0) {
        if (!sProjSP) stack_error();
        sProjSP ^= 1;
        sProj = sProjStack;
        sClipDirty = 1;
    } else if (sMtxMode == 3) {
        if (!sTexSP) stack_error();
        sTexSP ^= 1;
        sTex = sTexStack;
    } else {
        s32 off = (s32)(p << 26) >> 26;
        sPosSP = (sPosSP - off) & 63;
        if (sPosSP >= 31) stack_error();
        sPos = sPosStack[sPosSP & 31];
        sVec = sVecStack[sPosSP & 31];
        sClipDirty = 1;
    }
}
static void mtx_store(u32 p) {
    if (sMtxMode == 0) sProjStack = sProj;
    else if (sMtxMode == 3) sTexStack = sTex;
    else {
        u32 i = p & 0x1f;
        if (i == 31) stack_error();
        sPosStack[i] = sPos;
        sVecStack[i] = sVec;
    }
}
static void mtx_restore(u32 p) {
    if (sMtxMode == 0) { sProj = sProjStack; sClipDirty = 1; }
    else if (sMtxMode == 3) sTex = sTexStack;
    else {
        u32 i = p & 0x1f;
        if (i == 31) stack_error();
        sPos = sPosStack[i];
        sVec = sVecStack[i];
        sClipDirty = 1;
    }
}

static s16 sx10(u32 v) { return (s16)((v & 0x3ff) << 6) >> 3; }
static void light_vector(u32 p) {
    int l = p >> 30;
    s32 d[3] = {sx10(p), sx10(p >> 10), sx10(p >> 20)};
    for (int i = 0; i < 3; i++)
        sLightDir[l][i] = (s16)((((s64)d[0] * gVecM->m[i] + (s64)d[1] * gVecM->m[4 + i] + (s64)d[2] * gVecM->m[8 + i]) >> 12) >> 3);
    sLightParam[l] = p;
    for (int i = 0; i < 9; i++) sLightVec[l][i] = gVecM->m[(i / 3) * 4 + i % 3];
    static int sLightLog = -1;
    if (sLightLog < 0) sLightLog = getenv("ACWW_GPU3D_LIGHTLOG") != 0;
    if (sLightLog)
        fprintf(stderr, "light %d param %08x dir %d %d %d vec %d %d %d / %d %d %d / %d %d %d -> %d %d %d\n", l, p, d[0], d[1],
                d[2], sVec.m[0], sVec.m[1], sVec.m[2], sVec.m[4], sVec.m[5], sVec.m[6], sVec.m[8], sVec.m[9], sVec.m[10],
                sLightDir[l][0], sLightDir[l][1], sLightDir[l][2]);
}
typedef struct { s32 n[3]; s16 raw[3]; u32 lights; s16 dir[4][3]; u8 lc[4][3], dif[3], amb[3], spe[3], emi[3], out[3]; int table; s32 vec[9]; u32 lparam[4]; s32 lvec[4][9]; } LightRec;
static LightRec *sLightRecs;
HOST_STATE_KEEP(sLightRecs)
static unsigned sLightRecN;
static int sCurLightRec = -1;
void gpu3d_debug_light(int id) {
    if (!sLightRecs || id < 0) return;
    const LightRec *r = &sLightRecs[id & 0xffff];
    fprintf(stderr, "      light: normal %d %d %d (raw %d %d %d) lights %x table %d dif %d,%d,%d amb %d,%d,%d spe %d,%d,%d emi %d,%d,%d out %d,%d,%d\n",
            r->n[0], r->n[1], r->n[2], r->raw[0], r->raw[1], r->raw[2], r->lights, r->table, r->dif[0], r->dif[1], r->dif[2],
            r->amb[0], r->amb[1], r->amb[2], r->spe[0], r->spe[1], r->spe[2], r->emi[0], r->emi[1], r->emi[2], r->out[0],
            r->out[1], r->out[2]);
    for (int l = 0; l < 4; l++)
        if (r->lights & (1u << l))
            fprintf(stderr, "        light %d dir %d %d %d colour %d,%d,%d\n", l, r->dir[l][0], r->dir[l][1], r->dir[l][2],
                    r->lc[l][0], r->lc[l][1], r->lc[l][2]);
}
static void lighting(void) {
    s32 n[3];
    for (int i = 0; i < 3; i++)
        n[i] = (s32)(((s64)sNormal[0] * gVecM->m[i] + (s64)sNormal[1] * gVecM->m[4 + i] + (s64)sNormal[2] * gVecM->m[8 + i]) >> 12);
    if (sFeaturesOn > 1) {
        if (!sLightRecs) sLightRecs = calloc(65536, sizeof *sLightRecs);
        LightRec *r = &sLightRecs[sLightRecN & 0xffff];
        sCurLightRec = (int)(sLightRecN++ & 0xffff);
        memcpy(r->n, n, sizeof n);
        memcpy(r->raw, sNormal, sizeof r->raw);
        r->lights = sCurPolyAttr & 15;
        memcpy(r->dir, sLightDir, sizeof r->dir);
        memcpy(r->lc, sLightColor, sizeof r->lc);
        memcpy(r->dif, sDiffuse, 3);
        memcpy(r->amb, sAmbient, 3);
        memcpy(r->spe, sSpecular, 3);
        memcpy(r->emi, sEmission, 3);
        r->table = sShininessOn;
        for (int i = 0; i < 9; i++) r->vec[i] = gVecM->m[(i / 3) * 4 + i % 3];
        memcpy(r->lparam, sLightParam, sizeof r->lparam);
        memcpy(r->lvec, sLightVec, sizeof r->lvec);
    }
    s32 n9[3] = {n[0] >> 3, n[1] >> 3, n[2] >> 3};
    s32 c[3] = {sEmission[0] << 14, sEmission[1] << 14, sEmission[2] << 14};
    g3_feature(F_LIGHT);
    if (sLightCoefDirty) {
        for (int l = 0; l < 4; l++)
            for (int k = 0; k < 3; k++) {
                s32 lc = sLightColor[l][k];
                sPSpec[l][k] = sSpecular[k] * lc;
                sPDif[l][k] = sDiffuse[k] * lc;
                sPAmb[l][k] = (sAmbient[k] * lc) << 9;
            }
        sLightCoefDirty = 0;
    }
    for (int l = 0; l < 4; l++) {
        if (!(sCurPolyAttr & (1u << l))) continue;
        const s16 *L = sLightDir[l];
        s32 diffuse = ((-L[0] * n9[0]) >> 9) + ((-L[1] * n9[1]) >> 9) + ((-L[2] * n9[2]) >> 9);
        if (diffuse < 0) diffuse = 0;
        s32 shine = -((L[0] >> 1) * n9[0] + (L[1] >> 1) * n9[1] + ((L[2] - 0x200) >> 1) * n9[2]) >> 10;
        if (shine < 0) shine = 0;
        if (shine > 255) shine = 255;
        shine = (shine * shine) >> 8;
        if (sShininessOn) { shine = sShininess[shine >> 1]; g3_feature(F_SHININESS_TABLE); }
        for (int k = 0; k < 3; k++) {
            c[k] += sPSpec[l][k] * shine << 1;
            c[k] += sPDif[l][k] * diffuse;
            c[k] += sPAmb[l][k];
        }
    }
    for (int k = 0; k < 3; k++) c[k] >>= 14;
    for (int k = 0; k < 3; k++) sVtxColor[k] = (u8)(c[k] > 31 ? 31 : c[k]);
    if (sLightRecs && sCurLightRec >= 0) memcpy(sLightRecs[sCurLightRec].out, sVtxColor, 3);
}

static s64 floor_div(s64 a, s64 b) {
    s64 q = a / b;
    return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q;
}
static int sScreenYofs;
static void screen_position(G3Vertex *v) {
    if (v->screenDone) return;
    s64 w = v->pos[3];
    int s = w >= 0x10000;
    s64 ws = w >> s;
    if (ws == 0) ws = 1;
    s64 nx = ((s64)v->pos[0] + w) >> s, ny = (-(s64)v->pos[1] + w) >> s;
    v->x = (s32)(floor_div(nx * sViewport[4], ws * 2) + sViewport[0]) & 0x1ff;
    {
        s64 yy = floor_div(ny * sViewport[5], ws * 2) + sViewport[3] + sScreenYofs;
        v->y = (s32)((sScreenYofs && yy < 0 ? 0 : yy) & 0x3ff);
    }
    v->hx = (s32)(floor_div(nx * sViewport[4] * 16, ws * 2) + sViewport[0] * 16) & 0x1fff;
    {
        s64 hy = floor_div(ny * sViewport[5] * 16, ws * 2) + (sViewport[3] + sScreenYofs) * 16;
        v->hy = (s32)((sScreenYofs && hy < 0 ? 0 : hy) & 0x3fff);
    }
    v->screenDone = 1;
}

static int sClipK = 96;
static int sTall;
int gG3Yofs = 192;
void gpu3d_set_tall(int on, int above) {
    extern int sTallWanted;
    gG3Yofs = above < 0 ? 0 : above > G3_MAX_ROWS - 192 ? G3_MAX_ROWS - 192 : above;
    sTall = on && g3_tall_supported() && gG3Yofs > 0;
    sTallWanted = sTall;
}
static G3Vertex sClipBuf[2][10];
static s64 plane_d(const s32 *pos, int comp, int sign) {
    s64 k = comp == 1 && sign > 0 ? sClipK : 96;
    return k * pos[3] / 96 - sign * (s64)pos[comp];
}
static void clip_interp(G3Vertex *o, const G3Vertex *in, const G3Vertex *out, int comp, int sign) {
    s64 din = plane_d(in->pos, comp, sign);
    s64 dout = plane_d(out->pos, comp, sign);
    s64 den = din - dout;
#define LERP(a, b) ((s32)((b) + (((s64)(a) - (b)) * -dout) / den))
    for (int i = 0; i < 4; i++) o->pos[i] = LERP(in->pos[i], out->pos[i]);
    o->pos[comp] = (s32)(sign * (comp == 1 && sign > 0 ? sClipK : 96) * (s64)o->pos[3] / 96);
    {
        const s32 *ic = in->color;
        s64 cdin = din, cden = den;
        if (in->farAxis == comp + 1) {
            ic = in->farColor;
            cdin = plane_d(in->farPos, comp, sign);
            cden = cdin - dout;
        }
        for (int i = 0; i < 3; i++) {
            s64 a = ic[i] >> 4, b = out->color[i] >> 4, base, diff, n;
            if (a >= b) base = a, diff = a - b, n = cdin;
            else base = b, diff = b - a, n = cden - cdin;
            s64 t = (n * 65536 + cden - 1) / cden;
            s64 v = base * 65536 - diff * t;
            s32 c = (s32)(-((-v) >> 16));
            o->color[i] = c ? c << 4 | 0xf : 0;
        }
    }
    o->farAxis = (u8)(comp + 1);
    memcpy(o->farPos, out->pos, sizeof o->farPos);
    memcpy(o->farColor, out->color, sizeof o->farColor);
    for (int i = 0; i < 2; i++) o->tex[i] = (s16)LERP(in->tex[i], out->tex[i]);
#undef LERP
    o->dbgClips = (u8)(in->dbgClips + out->dbgClips + 1);
    for (int i = 0; i < 3; i++) o->dbgClip[i] = in->color[i], o->dbgClip[3 + i] = out->color[i];
    o->dbgClip[6] = (s32)din;
    o->dbgClip[7] = (s32)dout;
    o->clipped = 1;
    o->screenDone = 0;
}
static int inside(const G3Vertex *v, int comp, int sign) {
    return sign > 0 ? (s64)v->pos[comp] <= (comp == 1 ? sClipK : 96) * (s64)v->pos[3] / 96 : v->pos[comp] >= -v->pos[3];
}
static int clip_plane(G3Vertex *dst, const G3Vertex *src, int n, int comp, int sign) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        const G3Vertex *cur = &src[i], *prev = &src[(i + n - 1) % n], *next = &src[(i + 1) % n];
        if (inside(cur, comp, sign)) { dst[m++] = *cur; continue; }
        if (inside(prev, comp, sign)) clip_interp(&dst[m++], prev, cur, comp, sign);
        if (inside(next, comp, sign)) clip_interp(&dst[m++], next, cur, comp, sign);
    }
    return m;
}
static int clip_polygon_buf(G3Vertex (*buf)[10], G3Vertex *in, int n, G3Vertex **out, int farClip, int *clipped) {
    G3Vertex *a = buf[0], *b = buf[1];
    memcpy(a, in, n * sizeof *a);
    *clipped = 0;
    for (int i = 0; i < n; i++)
        if (a[i].pos[2] > a[i].pos[3]) {
            *clipped = 1;
            if (!farClip) { g3_feature(F_FARCLIP_HIDE); sFarHidden = 1; return 0; }
        }
    static int kPlanes[6][2] = {{2, 1}, {2, -1}, {1, 1}, {1, -1}, {0, 1}, {0, -1}};
    {
        static int init;
        const char *o = init ? 0 : getenv("ACWW_GPU3D_CLIPORDER");
        if (!init++ && o && strlen(o) == 6)
            for (int i = 0; i < 6; i++) {
                char c = o[i];
                kPlanes[i][0] = (c | 0x20) == 'x' ? 0 : (c | 0x20) == 'y' ? 1 : 2;
                kPlanes[i][1] = c >= 'a' ? 1 : -1;
            }
    }
    for (int p = 0; p < 6; p++) {
        int comp = kPlanes[p][0], sign = kPlanes[p][1];
        int all = 1;
        for (int i = 0; i < n; i++) if (!inside(&a[i], comp, sign)) { all = 0; break; }
        if (all) continue;
        *clipped = 1;
        n = clip_plane(b, a, n, comp, sign);
        G3Vertex *t = a; a = b; b = t;
        if (n == 0) return 0;
    }
    *out = a;
    return n;
}

static int clip_polygon(G3Vertex *in, int n, G3Vertex **out, int farClip, int *clipped) {
    return clip_polygon_buf(sClipBuf, in, n, out, farClip, clipped);
}

static void debug_tex_poly(const char *what, const G3Vertex *v, int nv, int n) {
    static long want = -2;
    if (want == -2) want = getenv("ACWW_GPU3D_DEBUGTEX") ? (long)strtoul(getenv("ACWW_GPU3D_DEBUGTEX"), 0, 16) : -1;
    if (want < 0 || (u32)want != sTexParam) return;
    fprintf(stderr, "debugtex frame %llu %s type %d n %d -> %d:", (unsigned long long)sFrame, what, sPolyType, nv, n);
    for (int i = 0; i < nv; i++) fprintf(stderr, " (%d %d %d %d)", v[i].pos[0], v[i].pos[1], v[i].pos[2], v[i].pos[3]);
    fprintf(stderr, "\n");
}

int gG3ClipSpaceOut;   // set by a GPU backend (vita_gl3d) that consumes clip-space pos[] directly
int gG3GpuXform;       // set by vita_gl3d: the GPU does the vertex transform/clip/cull from raw verts + matrix
static void poly_screen(G3Polygon *poly, int n) {
    if (gG3ClipSpaceOut) {
        // The GPU backend feeds clip-space pos[] to the hardware, which does the perspective divide, viewport
        // map and depth itself - so the software screen mapping here (screen_position(): 4 integer divides per
        // vertex, plus a per-vertex z divide) is pure wasted work, and integer divide is expensive on the
        // Cortex-A9 (it has none in hardware). Skip it; keep only the cheap w[] normalisation. Translucent draw
        // order falls back to submission order + the GPU depth test (the software back-to-front sort is dropped).
        int ws = 0;
        for (int i = 0; i < n; i++) { u32 w = (u32)poly->v[i]->pos[3]; while ((w >> ws) && ws < 32) ws += 4; }
        for (int i = 0; i < n; i++) {
            u32 w = (u32)poly->v[i]->pos[3];
            poly->w[i] = (s32)(ws < 16 ? w << (16 - ws) : w >> (ws - 16));
            poly->z[i] = 0;
        }
        poly->vtop = poly->vbottom = 0;
        poly->ytop = poly->ybottom = poly->xtop = poly->xbottom = 0;
        poly->sortKey = poly->translucent ? 1u << 21 : 0;
        return;
    }
    s32 ytop = 0x7fff, ybot = 0, xtop = 0, xbot = 0;
    int vtop = 0, vbot = 0;
    int wsize = 0;
    for (int i = 0; i < n; i++) {
        G3Vertex *x = poly->v[i];
        screen_position(x);
        u32 w = (u32)x->pos[3];
        while ((w >> wsize) && wsize < 32) wsize += 4;
        s64 z;
        if (poly->wbuffer) z = 0;
        else if (x->pos[3]) z = ((s64)x->pos[2] << 23) / x->pos[3] + 0x7fffff;
        else z = 0;
        if (z < 0) z = 0;
        if (z > 0xffffff) z = 0xffffff;
        poly->z[i] = (s32)z;
        if (x->y < ytop) { ytop = x->y; xtop = x->x; vtop = i; }
        if (x->y > ybot) { ybot = x->y; xbot = x->x; vbot = i; }
    }
    for (int i = 0; i < n; i++) {
        u32 w = (u32)poly->v[i]->pos[3];
        poly->w[i] = (s32)(wsize < 16 ? w << (16 - wsize) : w >> (wsize - 16));
        if (poly->wbuffer) {
            s64 z = wsize > 16 ? (s64)(w >> (wsize - 16)) << (wsize - 16) : (s64)w;
            poly->z[i] = (s32)(z > 0xffffff ? 0xffffff : z);
        }
    }
    poly->vtop = vtop;
    poly->vbottom = vbot;
    poly->ytop = ytop;
    poly->ybottom = ybot;
    poly->xtop = xtop;
    poly->xbottom = xbot;
    poly->sortKey = (u32)(ybot << 10 | ytop) | (poly->translucent ? 1u << 21 : 0);
}

// GPU-transform fast path: store the de-stripped polygon with raw (model-space) verts + the DS clip matrix and
// let the GPU do the transform, clip and cull. Skips the CPU vertex transform, Sutherland-Hodgman clip and the
// facing/cull math entirely - the bulk of the per-frame geometry cost on the Cortex-A9.
static void store_poly_gpu(const G3Vertex *v, int nv, u32 attr) {
    if (sNumPolys >= MAX_POLYGONS || sNumVerts + nv > MAX_VERTICES) { sRamOverflow = 1; sStripValid = 0; return; }
    update_clip();
    G3Polygon *poly = &sPolyRam[sBank][sNumPolys];
    memset(poly, 0, sizeof *poly);
    poly->index = sNumPolys;
    G3Vertex *ram = sVertexRam[sBank];
    for (int i = 0; i < nv; i++) { G3Vertex *x = &ram[sNumVerts++]; *x = v[i]; poly->v[i] = x; }
    poly->nv = nv;
    poly->attr = attr;
    poly->texparam = sTexParam;
    poly->texpal = sTexPal;
    u32 alpha = attr >> 16 & 31;
    int fmt = sTexParam >> 26 & 7;
    poly->translucent = (alpha > 0 && alpha < 31) || fmt == 1 || fmt == 6;
    int mode = attr >> 4 & 3;
    if (mode == 3) { if ((attr >> 24 & 63) == 0) poly->shadowMask = 1; else poly->shadow = 1; }
    // Convert the clip matrix to float only when it actually changed (per object, not per poly) - the int->float
    // divides are slow on the A9 VFP and were eating the GPU-transform win. Cache by clip serial.
    static u32 sStoreSerial = ~0u;
    static float sStoreMtx[16];
    if (sClipSerial != sStoreSerial) { for (int i = 0; i < 16; i++) sStoreMtx[i] = (float)sClip.m[i] / 4096.0f; sStoreSerial = sClipSerial; }
    memcpy(poly->glMtx, sStoreMtx, sizeof sStoreMtx);
    poly->glMtxId = sClipSerial;
    sStripValid = 0;
    sPolyOutcome = 0;
    sNumPolys++;
    if (!poly->translucent) sNumOpaque++;
}

static void submit_polygon(void) {
    sPolyDone = 1;
    sPolySerial++;
    int nv = sPolyType & 1 ? 4 : 3;
    G3Vertex v[4];
    if (sPolyType == 2 && (sStripPolys & 1)) { v[0] = sTemp[1]; v[1] = sTemp[0]; v[2] = sTemp[2]; }
    else if (sPolyType == 3) { v[0] = sTemp[0]; v[1] = sTemp[1]; v[2] = sTemp[3]; v[3] = sTemp[2]; }
    else memcpy(v, sTemp, nv * sizeof *v);
    if (gG3GpuXform) { store_poly_gpu(v, nv, sCurPolyAttr); return; }

    s64 ax = v[1].pos[0] - v[0].pos[0], ay = v[1].pos[1] - v[0].pos[1], aw = v[1].pos[3] - v[0].pos[3];
    s64 bx = v[2].pos[0] - v[0].pos[0], by = v[2].pos[1] - v[0].pos[1], bw = v[2].pos[3] - v[0].pos[3];
    s64 nx = ay * bw - aw * by, ny = aw * bx - ax * bw, nw = ax * by - ay * bx;
    s64 dot = nx * v[0].pos[0] + ny * v[0].pos[1] + nw * v[0].pos[3];
    int facing = dot >= 0;
    u32 attr = sCurPolyAttr;
    int culled = facing ? !(attr & 0x80) : !(attr & 0x40);
    sPolyOutcome = 0;
    sPolyNOut = 0;
    if (culled) {
        debug_tex_poly("culled", v, nv, 0);
        g3_feature(F_CULLED);
        sStripValid = 0;
        sPolyOutcome = 1;
        return;
    }

    G3Vertex *cv;
    int clipped;
    sFarHidden = 0;
    int clipk = sTall && sViewport[0] == 0 && sViewport[3] == 0 && sViewport[4] == 256 && sViewport[5] == 192 ? 96 + gG3Yofs : 96;
    sClipK = clipk;
    sScreenYofs = G3_YOFS;
    // When a GPU backend consumes clip-space pos[] and clips in hardware (gG3ClipSpaceOut), skip the CPU
    // Sutherland-Hodgman clip entirely (~11ms on the A9) and hand the GPU the unclipped polygon - GL clips it to
    // the frustum (including w<=0 near-plane verts) before the perspective divide.
    int n;
    if (gG3ClipSpaceOut && !gG3GpuXform) { cv = v; n = nv; clipped = 0; }
    else n = clip_polygon(v, nv, &cv, attr >> 12 & 1, &clipped);
    sClipK = 96;
    sPolyOutcome = n == 0 ? (sFarHidden ? 3 : 2) : 0;
    debug_tex_poly(n ? "kept" : "clipped away", v, nv, n);
    sPolyNOut = n;
    sPolyClipped = clipped;
    { static int dbg = -1; if (dbg < 0) dbg = getenv("ACWW_GPU3D_DEBUGPOLY") ? atoi(getenv("ACWW_GPU3D_DEBUGPOLY")) : 0; if (dbg > 0) { dbg--; fprintf(stderr, "poly attr %08x n %d -> %d clipped %d:", attr, nv, n, clipped); for (int i = 0; i < nv; i++) fprintf(stderr, " (%d %d %d %d)", v[i].pos[0], v[i].pos[1], v[i].pos[2], v[i].pos[3]); fprintf(stderr, "\n"); } }
    if (clipped) g3_feature(F_CLIP);
    int phantom = 0;
    if (n == 0 && sFarHidden && (sPolyType == 2 || sPolyType == 3) && sStripValid) {
        phantom = 1;
        n = 2;
        sPolyOutcome = 0;
        sPolyNOut = 2;
    }
    if (n == 0) { sStripValid = 0; return; }

    int strip = sPolyType == 2 || sPolyType == 3;
    int shared = !clipped && strip && sStripValid ? 2 : 0;
    int need = phantom ? 0 : n - shared;
    if (sNumPolys >= MAX_POLYGONS || sNumVerts + need > MAX_VERTICES) {
        sRamOverflow = 1;
        g3_feature(F_RAM_OVERFLOW);
        sPolyOutcome = 5;
        sStripValid = 0;
        return;
    }
    G3Polygon *poly = &sPolyRam[sBank][sNumPolys];
    // The GL renderer (gG3ClipSpaceOut) never reads orig[]/glMtx (debug dumps / GPU-transform only): leave those
    // ~900 bytes alone instead of clearing and refilling them for every polygon.
    memset(poly, 0, gG3ClipSpaceOut ? offsetof(G3Polygon, orig) : sizeof *poly);
    poly->index = sNumPolys;
    G3Vertex *ram = sVertexRam[sBank];
    if (phantom) {
        int rev = sPolyType == 2 && (sStripPolys & 1);
        poly->v[0] = sStripRam[rev ? 1 : 0];
        poly->v[1] = sStripRam[rev ? 0 : 1];
        sStripValid = 0;
    } else if (!clipped) {
        static const u8 kOrder[5][4] = {{0, 1, 2}, {0, 1, 2, 3}, {0, 1, 2}, {0, 1, 3, 2}, {1, 0, 2}};
        const u8 *ord = kOrder[sPolyType == 2 && (sStripPolys & 1) ? 4 : sPolyType];
        G3Vertex *at[4];
        for (int k = 0; k < nv; k++) {
            if (k < shared) at[k] = sStripRam[k];
            else { at[k] = &ram[sNumVerts++]; *at[k] = sTemp[k]; }
        }
        for (int i = 0; i < n; i++) poly->v[i] = at[ord[i]];
        if (strip) {
            sStripRam[0] = at[nv - 2];
            sStripRam[1] = at[nv - 1];
            sStripValid = 1;
        }
    } else {
        for (int i = 0; i < n; i++) {
            G3Vertex *x = &ram[sNumVerts++];
            *x = cv[i];
            poly->v[i] = x;
        }
        sStripValid = 0;
    }
    poly->nv = n;
    poly->attr = attr;
    poly->texparam = sTexParam;
    poly->texpal = sTexPal;
    poly->facing = (u8)facing;
    u32 alpha = attr >> 16 & 31;
    int fmt = sTexParam >> 26 & 7;
    poly->translucent = (alpha > 0 && alpha < 31) || fmt == 1 || fmt == 6;
    int mode = attr >> 4 & 3;
    if (mode == 3) {
        if ((attr >> 24 & 63) == 0) poly->shadowMask = 1, g3_feature(F_SHADOW_MASK);
        else poly->shadow = 1, g3_feature(F_SHADOW);
    }
    g3_feature(F_POLYMODE + mode);
    if (sTexParam >> 26 & 7) g3_feature(F_TEXFMT + fmt);
    if (poly->translucent) g3_feature(F_TRANSLUCENT);
    if (alpha == 0) g3_feature(F_WIREFRAME);
    if (attr & (1u << 14)) g3_feature(F_DEPTH_EQUAL);
    if (attr & (1u << 15)) g3_feature(F_FOG_POLY);
    if (attr & (1u << 11)) g3_feature(F_TRANS_DEPTH_UPDATE);
    if (fmt) {
        if (sTexParam & (3u << 16)) g3_feature(F_TEX_REPEAT);
        if (sTexParam & (3u << 18)) g3_feature(F_TEX_FLIP);
        if ((sTexParam & (3u << 16)) != (3u << 16)) g3_feature(F_TEX_CLAMP);
        if (sTexParam & (1u << 29)) g3_feature(F_COLOR0_TRANSPARENT);
    }

    poly->wbuffer = (u8)(sFlushParam >> 1 & 1);
    poly_screen(poly, n);
    if (gG3ClipSpaceOut) poly->norig = 0;
    else {
        poly->norig = phantom ? 0 : nv;
        memcpy(poly->orig, v, sizeof(G3Vertex) * nv);
    }
    poly->yofs = (s16)sScreenYofs;
    if (!(attr & (1u << 13))) {
        int dot1 = 1;
        for (int i = 1; i < n; i++)
            if (poly->v[i]->x != poly->v[0]->x || poly->v[i]->y != poly->v[0]->y) { dot1 = 0; break; }
        if (dot1) {
            s32 wmin = 0x7fffffff;
            for (int i = 0; i < n; i++) if (poly->w[i] < wmin) wmin = poly->w[i];
            if (wmin > ((s32)sDot1Depth << 9) / 0x200 * 0x200 && wmin > (s32)sDot1Depth) {
                g3_feature(F_ONEDOT_HIDE);
                sPolyOutcome = 4;
                sNumVerts -= need;
                sStripValid = 0;
                return;
            }
        }
    }
    sNumPolys++;
    if (!poly->translucent) sNumOpaque++;
}

// GL fast path polygon: the same outcome as submit_polygon() with gG3ClipSpaceOut (no CPU clip, so never a
// phantom or split polygon), but vertices by pointer into the pool and only the fields the GL renderer reads.
static void submit_polygon_fast(void) {
    sPolyDone = 1;
    sPolySerial++;
    int nv = sPolyType & 1 ? 4 : 3;
    const G3VHot *v[4];
    if (sPolyType == 2 && (sStripPolys & 1)) { v[0] = sTempH[1]; v[1] = sTempH[0]; v[2] = sTempH[2]; }
    else if (sPolyType == 3) { v[0] = sTempH[0]; v[1] = sTempH[1]; v[2] = sTempH[3]; v[3] = sTempH[2]; }
    else { v[0] = sTempH[0]; v[1] = sTempH[1]; v[2] = sTempH[2]; v[3] = sTempH[3]; }

    s64 ax = v[1]->pos[0] - v[0]->pos[0], ay = v[1]->pos[1] - v[0]->pos[1], aw = v[1]->pos[3] - v[0]->pos[3];
    s64 bx = v[2]->pos[0] - v[0]->pos[0], by = v[2]->pos[1] - v[0]->pos[1], bw = v[2]->pos[3] - v[0]->pos[3];
    s64 nx = ay * bw - aw * by, ny = aw * bx - ax * bw, nw = ax * by - ay * bx;
    s64 dot = nx * v[0]->pos[0] + ny * v[0]->pos[1] + nw * v[0]->pos[3];
    int facing = dot >= 0;
    u32 attr = sCurPolyAttr;
    sPolyOutcome = 0;
    sPolyNOut = 0;
    if (facing ? !(attr & 0x80) : !(attr & 0x40)) { sStripValid = 0; sPolyOutcome = 1; return; }
    sPolyNOut = nv;
    sPolyClipped = 0;

    int strip = sPolyType == 2 || sPolyType == 3;
    int need = nv - (strip && sStripValid ? 2 : 0);
    int spill = 0;
    for (int i = 0; i < nv; i++) spill |= v[i] == &sPoolSpill;
    if (sNumPolys >= MAX_POLYGONS || sNumVerts + need > MAX_VERTICES || spill) {
        sRamOverflow = 1;
        sPolyOutcome = 5;
        sStripValid = 0;
        return;
    }
    if (strip) sStripValid = 1;
    // One-dot polygons: the slow GL path never computes screen x/y (they stay 0), so every polygon counts as one
    // dot there and only the depth test below decides. Same here, from the same normalised w.
    if (!(attr & (1u << 13))) {
        int ws = 0;
        for (int i = 0; i < nv; i++) { u32 w = (u32)v[i]->pos[3]; while ((w >> ws) && ws < 32) ws += 4; }
        s32 wmin = 0x7fffffff;
        for (int i = 0; i < nv; i++) {
            u32 w = (u32)v[i]->pos[3];
            s32 wn = (s32)(ws < 16 ? w << (16 - ws) : w >> (ws - 16));
            if (wn < wmin) wmin = wn;
        }
        if (wmin > ((s32)sDot1Depth << 9) / 0x200 * 0x200 && wmin > (s32)sDot1Depth) {
            sPolyOutcome = 4;
            sStripValid = 0;
            return;
        }
    }
    sNumVerts += need;
    G3Polygon *poly = &sPolyRam[sBank][sNumPolys];
    poly->hv[0] = v[0]; poly->hv[1] = v[1]; poly->hv[2] = v[2]; poly->hv[3] = nv == 4 ? v[3] : 0;
    poly->v[0] = poly->v[1] = poly->v[2] = poly->v[3] = 0;
    poly->nv = nv;
    poly->attr = attr;
    poly->texparam = sTexParam;
    poly->texpal = sTexPal;
    poly->facing = (u8)facing;
    u32 alpha = attr >> 16 & 31;
    int fmt = sTexParam >> 26 & 7;
    int tl = (alpha > 0 && alpha < 31) || fmt == 1 || fmt == 6;
    poly->translucent = (u8)tl;
    int mode = attr >> 4 & 3;
    poly->shadowMask = mode == 3 && (attr >> 24 & 63) == 0;
    poly->shadow = mode == 3 && (attr >> 24 & 63) != 0;
    poly->wbuffer = (u8)(sFlushParam >> 1 & 1);
    poly->sortKey = tl ? 1u << 21 : 0;
    poly->index = sNumPolys;
    poly->yofs = (s16)G3_YOFS;
    poly->norig = 0;
    poly->ytop = 0;
    sNumPolys++;
    if (!tl) sNumOpaque++;
}

static void submit_vertex_fast(void) {
    if (!sMT) update_clip();
    const s32 *cm = gClipM->m;
    G3VHot *v;
    if (sPoolN < G3_POOL_N) v = &sPool[sBank][sPoolN++];
    else { v = &sPoolSpill; sRamOverflow = 1; }
    s32 x = sVtx[0], y = sVtx[1], z = sVtx[2];
    for (int i = 0; i < 4; i++)
        v->pos[i] = (s32)(((s64)x * cm[i] + (s64)y * cm[4 + i] + (s64)z * cm[8 + i] + ((s64)cm[12 + i] << 12)) >> 12);
    for (int i = 0; i < 3; i++) v->color[i] = sVtxColor[i] ? sVtxColor[i] << 4 | 0xf : 0;
    if ((sTexParam >> 30) == 3) {
        const s32 *tm = gTexM->m;
        for (int i = 0; i < 2; i++)
            v->tex[i] = (s16)((((s64)x * tm[i] + (s64)y * tm[4 + i] + (s64)z * tm[8 + i]) >> 24) + sTexRaw[i]);
    } else {
        v->tex[0] = sTexCoord[0];
        v->tex[1] = sTexCoord[1];
    }
    sTempH[sVtxInPoly++] = v;
    sVtxCount++;
    switch (sPolyType) {
    case 0: if (sVtxInPoly == 3) { submit_polygon_fast(); sVtxInPoly = 0; } break;
    case 1: if (sVtxInPoly == 4) { submit_polygon_fast(); sVtxInPoly = 0; } break;
    case 2:
        if (sVtxInPoly == 3) {
            submit_polygon_fast();
            sStripPolys++;
            sTempH[0] = sTempH[1]; sTempH[1] = sTempH[2];
            sVtxInPoly = 2;
        }
        break;
    case 3:
        if (sVtxInPoly == 4) {
            submit_polygon_fast();
            sStripPolys++;
            sTempH[0] = sTempH[2]; sTempH[1] = sTempH[3];
            sVtxInPoly = 2;
        }
        break;
    }
}

static int fast_gl_decide(void) {
    if (sFastGL < 0) {
        const char *e = getenv("ACWW_G3FAST");
        sFastGL = gG3ClipSpaceOut && !gG3GpuXform && !(e && !atoi(e));
        if (sFastGL) {
            for (int b = 0; b < 2; b++)
                if (!sPool[b] && !(sPool[b] = malloc(sizeof(G3VHot) * G3_POOL_N))) sFastGL = 0;
            fprintf(stderr, "gpu3d: lean GL vertex path %s\n", sFastGL ? "on" : "off (no memory)");
        }
    }
    return sFastGL;
}

static void submit_vertex(void) {
    if (!sInBegin) return;
    if (sFastGL > 0 || (sFastGL < 0 && fast_gl_decide())) { submit_vertex_fast(); return; }
    if (!sMT) update_clip();
    const s32 *cm = gClipM->m, *tm = gTexM->m;
    G3Vertex *v = &sTemp[sVtxInPoly];
    s64 x = sVtx[0], y = sVtx[1], z = sVtx[2];
    v->raw[0] = (s32)x; v->raw[1] = (s32)y; v->raw[2] = (s32)z;   // keep model-space coords for the GPU transform
    if (!gG3GpuXform)
        for (int i = 0; i < 4; i++)
            v->pos[i] = (s32)((x * cm[i] + y * cm[4 + i] + z * cm[8 + i] + 0x1000 * (s64)cm[12 + i]) >> 12);
    for (int i = 0; i < 3; i++) v->color[i] = sVtxColor[i] ? sVtxColor[i] << 4 | 0xf : 0;
    v->dbgLight = sCurLightRec;
    if ((sTexParam >> 30) == 3) {
        for (int i = 0; i < 2; i++)
            v->tex[i] = (s16)(((x * tm[i] + y * tm[4 + i] + z * tm[8 + i]) >> 24) + sTexRaw[i]);
    } else {
        v->tex[0] = sTexCoord[0];
        v->tex[1] = sTexCoord[1];
    }
    v->clipped = 0;
    v->farAxis = 0;
    v->dbgClips = 0;
    v->screenDone = 0;
    sVtxInPoly++;
    sVtxCount++;
    switch (sPolyType) {
    case 0:
        if (sVtxInPoly == 3) { submit_polygon(); sVtxInPoly = 0; }
        break;
    case 1:
        if (sVtxInPoly == 4) { submit_polygon(); sVtxInPoly = 0; }
        break;
    case 2:
        if (sVtxInPoly == 3) {
            submit_polygon();
            sStripPolys++;
            sTemp[0] = sTemp[1];
            sTemp[1] = sTemp[2];
            sVtxInPoly = 2;
        }
        break;
    case 3:
        if (sVtxInPoly == 4) {
            submit_polygon();
            sStripPolys++;
            sTemp[0] = sTemp[2];
            sTemp[1] = sTemp[3];
            sVtxInPoly = 2;
        }
        break;
    }
}

static void pos_test(void) {
    update_clip();
    s64 x = sVtx[0], y = sVtx[1], z = sVtx[2];
    for (int i = 0; i < 4; i++)
        sPosResult[i] = (s32)((x * sClip.m[i] + y * sClip.m[4 + i] + z * sClip.m[8 + i] + 0x1000 * (s64)sClip.m[12 + i]) >> 12);
}
static void vec_test(u32 p) {
    s32 d[3] = {sx10(p), sx10(p >> 10), sx10(p >> 20)};
    for (int i = 0; i < 3; i++) {
        s32 r = (s32)(((s64)d[0] * sVec.m[i] + (s64)d[1] * sVec.m[4 + i] + (s64)d[2] * sVec.m[8 + i]) >> 12);
        sVecResult[i] = (s16)(r << 3) >> 3;
    }
}
static void box_test(const u32 *p) {
    update_clip();
    s16 bx = (s16)p[0], by = (s16)(p[0] >> 16), bz = (s16)p[1];
    s16 bw = (s16)(p[1] >> 16), bh = (s16)p[2], bd = (s16)(p[2] >> 16);
    G3Vertex c[8];
    for (int i = 0; i < 8; i++) {
        s64 x = bx + (i & 1 ? bw : 0), y = by + (i & 2 ? bh : 0), z = bz + (i & 4 ? bd : 0);
        x = (s16)x; y = (s16)y; z = (s16)z;
        memset(&c[i], 0, sizeof c[i]);
        for (int k = 0; k < 4; k++)
            c[i].pos[k] = (s32)((x * sClip.m[k] + y * sClip.m[4 + k] + z * sClip.m[8 + k] + 0x1000 * (s64)sClip.m[12 + k]) >> 12);
    }
    static const u8 kFaces[6][4] = {{0, 1, 3, 2}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 3, 7, 5}};
    int in = 0;
    for (int f = 0; f < 6 && !in; f++) {
        G3Vertex face[4], *out;
        for (int k = 0; k < 4; k++) face[k] = c[kFaces[f][k]];
        int clipped;
        // Own clip buffers: with the GX worker this runs on the game thread while the worker may be clipping.
        static G3Vertex sBoxClip[2][10];
        if (clip_polygon_buf(sBoxClip, face, 4, &out, 1, &clipped) > 0) in = 1;
    }
    sGxStat = (sGxStat & ~2u) | (in ? 2u : 0);
}

// Lit-colour cache: lighting() is a pure function of the raw normal, the vector matrix, the light/material state
// and the polygon's light-enable bits. Models repeat normals a lot (flat ground, faceted meshes), so remember the
// colour per raw normal until any of those inputs change (sLitGen bump) - same result, no recompute.
typedef struct { u32 key, gen; u8 col[3], attr; } LitEnt;
static LitEnt sLitCache[256];
static u32 sLitGen = 1;
static void lit_cached(u32 raw) {
    if (sFeaturesOn > 1) { lighting(); return; }
    u32 h = (raw ^ raw >> 10 ^ raw >> 20) & 255;
    LitEnt *c = &sLitCache[h];
    u8 attr = (u8)(sCurPolyAttr & 15);
    if (c->gen == sLitGen && c->key == raw && c->attr == attr) {
        sVtxColor[0] = c->col[0]; sVtxColor[1] = c->col[1]; sVtxColor[2] = c->col[2];
        sCurLightRec = -1;
        return;
    }
    lighting();
    c->gen = sLitGen; c->key = raw; c->attr = attr;
    c->col[0] = sVtxColor[0]; c->col[1] = sVtxColor[1]; c->col[2] = sVtxColor[2];
}

static u32 sExecParams[32];
static int sExecCount, sExecCmd = -1;

static struct { u8 cmd; u32 p0; int mode, sp; } sHist[64];
static unsigned sHistPos;
void gpu3d_debug_history(void) {
    for (unsigned i = 48; i < 64; i++) {
        unsigned k = (sHistPos + i) & 63;
        fprintf(stderr, " %02x(%x)m%d,sp%d", sHist[k].cmd, sHist[k].p0, sHist[k].mode, sHist[k].sp);
    }
    fprintf(stderr, "\n");
}
static void execute(u8 cmd, const u32 *p) {
    {
        static long f = -2;
        if (f == -2) f = getenv("ACWW_GPU3D_CMDLOG") ? atol(getenv("ACWW_GPU3D_CMDLOG")) : -1;
        if (f >= 0 && sFrame == (u64)f) fprintf(stderr, "cmdlog %02x %08x tex %08x vin %d\n", cmd, p[0], sTexParam, sVtxInPoly);
    }
    g3_feature(F_CMD + (cmd & 0x7f));
    // Lighting inputs changed. With the GX worker, matrix commands run on the game thread and the worker sees the
    // vector matrix only through its 0xf1 snapshot (which bumps there): the counter stays single-writer.
    if ((cmd >= 0x30 && cmd <= 0x34) || (!sMT && cmd >= 0x10 && cmd <= 0x1c)) sLitGen++;
    Mtx a;
    switch (cmd) {
    case 0x10: sMtxMode = p[0] & 3; break;
    case 0x11: mtx_push(); break;
    case 0x12: mtx_pop(p[0]); break;
    case 0x13: mtx_store(p[0]); break;
    case 0x14: mtx_restore(p[0]); break;
    case 0x15: mtx_identity(&a); apply_load(&a); break;
    case 0x16: for (int i = 0; i < 16; i++) a.m[i] = (s32)p[i]; apply_load(&a); break;
    case 0x17:
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 3; c++) a.m[r * 4 + c] = (s32)p[r * 3 + c];
            a.m[r * 4 + 3] = r == 3 ? 0x1000 : 0;
        }
        apply_load(&a);
        break;
    case 0x18: for (int i = 0; i < 16; i++) a.m[i] = (s32)p[i]; apply_mult(&a); break;
    case 0x19:
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 3; c++) a.m[r * 4 + c] = (s32)p[r * 3 + c];
            a.m[r * 4 + 3] = r == 3 ? 0x1000 : 0;
        }
        apply_mult(&a);
        break;
    case 0x1a:
        mtx_identity(&a);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) a.m[r * 4 + c] = (s32)p[r * 3 + c];
        apply_mult(&a);
        break;
    case 0x1b: {
        s32 s[3] = {(s32)p[0], (s32)p[1], (s32)p[2]};
        if (sMtxMode == 3) mtx_scale(&sTex, s);
        else if (sMtxMode == 0) { mtx_scale(&sProj, s); sClipDirty = 1; }
        else { mtx_scale(&sPos, s); sClipDirty = 1; }
        break;
    }
    case 0x1c: {
        s32 t[3] = {(s32)p[0], (s32)p[1], (s32)p[2]};
        switch (sMtxMode) {
        case 0: mtx_trans(&sProj, t); sClipDirty = 1; break;
        case 1: mtx_trans(&sPos, t); sClipDirty = 1; break;
        case 2: mtx_trans(&sPos, t); mtx_trans(&sVec, t); sClipDirty = 1; break;
        default: mtx_trans(&sTex, t); break;
        }
        break;
    }
    case 0x20:
        sVtxColor[0] = p[0] & 31;
        sVtxColor[1] = p[0] >> 5 & 31;
        sVtxColor[2] = p[0] >> 10 & 31;
        sCurLightRec = -1;
        break;
    case 0x21:
        sNormal[0] = sx10(p[0]);
        sNormal[1] = sx10(p[0] >> 10);
        sNormal[2] = sx10(p[0] >> 20);
        if ((sTexParam >> 30) == 2) {
            for (int i = 0; i < 2; i++)
                sTexCoord[i] = (s16)((((s64)sNormal[0] * gTexM->m[i] + (s64)sNormal[1] * gTexM->m[4 + i] +
                                       (s64)sNormal[2] * gTexM->m[8 + i]) >> 24) + sTexRaw[i]);
        }
        lit_cached(p[0] & 0x3fffffff);
        break;
    case 0x22:
        sTexRaw[0] = (s16)p[0];
        sTexRaw[1] = (s16)(p[0] >> 16);
        if ((sTexParam >> 30) == 1) {
            for (int i = 0; i < 2; i++)
                sTexCoord[i] = (s16)(((s64)sTexRaw[0] * gTexM->m[i] + (s64)sTexRaw[1] * gTexM->m[4 + i] + gTexM->m[8 + i] +
                                      gTexM->m[12 + i]) >> 12);
        } else {
            sTexCoord[0] = sTexRaw[0];
            sTexCoord[1] = sTexRaw[1];
        }
        break;
    case 0x23:
        sVtx[0] = (s16)p[0];
        sVtx[1] = (s16)(p[0] >> 16);
        sVtx[2] = (s16)p[1];
        submit_vertex();
        break;
    case 0x24:
        sVtx[0] = (s16)((p[0] & 0x3ff) << 6);
        sVtx[1] = (s16)((p[0] >> 10 & 0x3ff) << 6);
        sVtx[2] = (s16)((p[0] >> 20 & 0x3ff) << 6);
        submit_vertex();
        break;
    case 0x25: sVtx[0] = (s16)p[0]; sVtx[1] = (s16)(p[0] >> 16); submit_vertex(); break;
    case 0x26: sVtx[0] = (s16)p[0]; sVtx[2] = (s16)(p[0] >> 16); submit_vertex(); break;
    case 0x27: sVtx[1] = (s16)p[0]; sVtx[2] = (s16)(p[0] >> 16); submit_vertex(); break;
    case 0x28:
        sVtx[0] += (s16)((p[0] & 0x3ff) << 6) >> 6;
        sVtx[1] += (s16)((p[0] >> 10 & 0x3ff) << 6) >> 6;
        sVtx[2] += (s16)((p[0] >> 20 & 0x3ff) << 6) >> 6;
        submit_vertex();
        break;
    case 0x29: sPolyAttr = p[0]; break;
    case 0x2a: sTexParam = p[0]; g3_feature(F_TEXGEN + (p[0] >> 30)); break;
    case 0x2b: sTexPal = p[0] & 0x1fff; break;
    case 0x30:
        sDiffuse[0] = p[0] & 31;
        sDiffuse[1] = p[0] >> 5 & 31;
        sDiffuse[2] = p[0] >> 10 & 31;
        sAmbient[0] = p[0] >> 16 & 31;
        sAmbient[1] = p[0] >> 21 & 31;
        sAmbient[2] = p[0] >> 26 & 31;
        sLightCoefDirty = 1;
        if (p[0] & 0x8000) {
            for (int i = 0; i < 3; i++) sVtxColor[i] = sDiffuse[i];
            sCurLightRec = -1;
        }
        break;
    case 0x31:
        sSpecular[0] = p[0] & 31;
        sSpecular[1] = p[0] >> 5 & 31;
        sSpecular[2] = p[0] >> 10 & 31;
        sLightCoefDirty = 1;
        sEmission[0] = p[0] >> 16 & 31;
        sEmission[1] = p[0] >> 21 & 31;
        sEmission[2] = p[0] >> 26 & 31;
        sShininessOn = p[0] >> 15 & 1;
        break;
    case 0x32: light_vector(p[0]); break;
    case 0x33: {
        int l = p[0] >> 30;
        sLightColor[l][0] = p[0] & 31;
        sLightColor[l][1] = p[0] >> 5 & 31;
        sLightColor[l][2] = p[0] >> 10 & 31;
        sLightCoefDirty = 1;
        break;
    }
    case 0x34:
        for (int i = 0; i < 32; i++)
            for (int k = 0; k < 4; k++) sShininess[i * 4 + k] = (u8)(p[i] >> (8 * k));
        break;
    case 0x40:
        sPolyType = p[0] & 3;
        sCurPolyAttr = sPolyAttr;
        sInBegin = 1;
        sVtxInPoly = 0;
        sStripPolys = 0;
        sStripValid = 0;
        break;
    case 0x41: break;
    case 0x50:
        sFlushPending = 1;
        sFlushParam = p[0] & 3;
        break;
    case 0x60:
        sViewport[0] = p[0] & 0xff;
        sViewport[1] = p[0] >> 8 & 0xff;
        sViewport[2] = p[0] >> 16 & 0xff;
        sViewport[3] = (191 - (p[0] >> 24 & 0xff)) & 0xff;
        sViewport[4] = (sViewport[2] - sViewport[0] + 1) & 0x1ff;
        sViewport[5] = ((p[0] >> 24 & 0xff) - sViewport[1] + 1) & 0xff;
        break;
    case 0x70: g3_feature(F_BOX_TEST); box_test(p); break;
    case 0x71:
        g3_feature(F_POS_TEST);
        sVtx[0] = (s16)p[0];
        sVtx[1] = (s16)(p[0] >> 16);
        sVtx[2] = (s16)p[1];
        pos_test();
        break;
    case 0x72: g3_feature(F_VEC_TEST); vec_test(p[0]); break;
    default: break;
    }
}


// ---- GX worker ------------------------------------------------------------------------------------------------
// Profiling the town showed ~3/4 of the frame's CPU phase is GX command processing (FIFO emulation, vertex
// transform, lighting, polygon build) running serially on the game's core. With the worker on, the game thread
// only decodes the command stream and keeps the matrix stack; complete commands go into a single-producer /
// single-consumer ring and execute on another core, with no FIFO/pipe/cycle-timing emulation at all.
//
// What the game can read back is all answered on the game thread without waiting for the worker: GXSTAT (stack
// levels / error bit; the FIFO always reads empty and idle), the clip/vector matrices, and POS_TEST / VEC_TEST
// (pure functions of the matrices). BOX_TEST and the RAM-count register first wait for the worker to catch up.
//
// SWAP_BUFFERS parks the worker until the game thread's vblank commit swaps the polygon RAM. In async mode the
// commit does not wait for the worker, so geometry runs one frame pipelined and overlaps the render phase too.
enum { MTQ_N = 1 << 19, MTQ_M = MTQ_N - 1 };   // ring size in words (~16 heavy frames)
static u32 *sMtQ;
static int sRawOpen;   // raw GX FIFO run open in the ring (see "raw GX FIFO")
// Free-running word positions. The game thread fills up to sMtWLocal and publishes sMtW in batches (MTQ_BATCH
// words, or at any sync/commit) so the two cores do not trade a cache line on every command; the worker likewise
// publishes sMtR per batch. Each published index sits on its own cache line.
enum { MTQ_BATCH = 512 };
static u32 sMtWLocal;
static u32 sMtWDone;   // end of the last complete command: what gets published (never part of a command)
static u32 sMtRSeen;   // last worker position this thread read (re-read only when the ring looks full)
static u32 sMtW __attribute__((aligned(64)));
static u32 sMtR __attribute__((aligned(64)));
static int sMtPad __attribute__((aligned(64), unused));
static int sMtParked;             // worker executed SWAP_BUFFERS and waits for the commit
static int sMtAsync, sMtStarted;
static u32 sMtSwapsQueued, sMtSwapsDone;
static int sMtCvDirty, sMtTexDirty;
// Frameskip (pipelined mode): when two built frames are already waiting to be shown, the next frame's vertices
// and swap are not sent to the worker at all - the game runs on, state commands still apply, nothing is built.
static int sMtSkip;
static unsigned long long sMtSkipped;
static unsigned long long sMtBusyNs;
static pthread_t sMtThread;
HOST_STATE_KEEP(sMtQ) HOST_STATE_KEEP(sMtW) HOST_STATE_KEEP(sMtR) HOST_STATE_KEEP(sMtWLocal) HOST_STATE_KEEP(sMtWDone) HOST_STATE_KEEP(sMtRSeen) HOST_STATE_KEEP(sMtParked)
HOST_STATE_KEEP(sMtStarted) HOST_STATE_KEEP(sMtThread) HOST_STATE_KEEP(sMT) HOST_STATE_KEEP(sMtAsync)
HOST_STATE_KEEP(gClipM) HOST_STATE_KEEP(gVecM) HOST_STATE_KEEP(gTexM)
extern uint64_t host_plat_time_ns(void);

extern unsigned int gHostGxStageN;   // GX store queue (hw.c)
void HostGx_StageFlush(void);
static void mt_sync_body(void);
// The worker sleeps (instead of polling) when its ring is empty or it is parked on a swap: an idle worker no longer
// takes time from the render/2D threads that share its cores. The game thread wakes it only when it is asleep
// (the sleeping flag and the published index are a seq_cst store-then-load pair on each side).
static pthread_mutex_t sMtSleepMx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t sMtSleepCv = PTHREAD_COND_INITIALIZER;
static int sMtSleeping;
static inline void mt_wake(void) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    if (__atomic_load_n(&sMtSleeping, __ATOMIC_SEQ_CST)) {
        pthread_mutex_lock(&sMtSleepMx);
        __atomic_store_n(&sMtSleeping, 0, __ATOMIC_SEQ_CST);
        pthread_cond_signal(&sMtSleepCv);
        pthread_mutex_unlock(&sMtSleepMx);
    }
}
static inline void mt_publish_w(u32 w) { __atomic_store_n(&sMtW, w, __ATOMIC_RELEASE); mt_wake(); }
// Worker: sleep until the published index moves past r (parkWait = 0) or the park is released (parkWait = 1).
static void mt_sleep(u32 r, int parkWait) {
    pthread_mutex_lock(&sMtSleepMx);
    __atomic_store_n(&sMtSleeping, 1, __ATOMIC_SEQ_CST);
    for (;;) {
        int ready = parkWait ? !__atomic_load_n(&sMtParked, __ATOMIC_SEQ_CST) : __atomic_load_n(&sMtW, __ATOMIC_SEQ_CST) != r;
        if (ready || !__atomic_load_n(&sMtSleeping, __ATOMIC_SEQ_CST)) break;
        pthread_cond_wait(&sMtSleepCv, &sMtSleepMx);
    }
    __atomic_store_n(&sMtSleeping, 0, __ATOMIC_SEQ_CST);
    pthread_mutex_unlock(&sMtSleepMx);
}
static void mt_pause(int *spins) {
    if (++*spins < 32) sched_yield();
    else usleep(100);   // idle: sleep instead of spinning so the core goes to the render thread / 2D band
}
static void gpu3d_write16_now(uint32_t addr, uint16_t v);
// Worker side of the raw FIFO forwarding (see "raw GX FIFO" below): the packed-command decoder, run on the worker.
// Matrix commands and the tests were already executed by the game thread (the worker gets the matrices through
// the 0xf1/0xf2 snapshots placed in the stream); everything else executes here exactly as before.
static u32 wPackCmds, wParams[32];
static int wPackLeft, wParamsLeft, wCnt;
static const u32 kZeroParams[32];
static void w_exec(u8 c, const u32 *p) {
    if (c >= 0x10 && c <= 0x1c) return;
    if (c == 0x70 || c == 0x72) return;
    if (c == 0x71) { sVtx[0] = (s16)p[0]; sVtx[1] = (s16)(p[0] >> 16); sVtx[2] = (s16)p[1]; return; }   // result: game thread
    if (!is_command(c)) return;
    execute(c, p);
}
static void w_advance(void) {
    while (wPackLeft > 0) {
        u8 c = wPackCmds & 0xff;
        if (wPackCmds == 0) { wPackLeft = 0; break; }
        if (kParams[c]) { wParamsLeft = kParams[c]; wCnt = 0; return; }
        if (c) w_exec(c, kZeroParams);
        wPackCmds >>= 8;
        wPackLeft--;
    }
    wParamsLeft = 0;
}
static inline void w_word(u32 v) {
    if (wParamsLeft == 0) {
        if (v == 0) return;
        wPackCmds = v;
        wPackLeft = 4;
        w_advance();
        return;
    }
    wParams[wCnt++] = v;
    if (--wParamsLeft == 0) {
        w_exec(wPackCmds & 0xff, wParams);
        wPackCmds >>= 8;
        wPackLeft--;
        w_advance();
    }
}
static void *mt_worker(void *arg) {
    (void)arg;
#if HOST_VITA
    // Cores 1+2 only (USER_1|USER_2): core 0 belongs to the game thread. A two-core mask, never a single core
    // (see the note in gpu3d_raster.c's band_worker).
    extern int sceKernelChangeThreadCpuAffinityMask(int, int);
    sceKernelChangeThreadCpuAffinityMask(0, 0x60000);
#endif
    u32 tmp[64];
    int spins = 0;
    u32 r = 0;
    for (;;) {
        u32 w = __atomic_load_n(&sMtW, __ATOMIC_ACQUIRE);
        if (r == w) {
            if (++spins < 32) sched_yield();
            else { mt_sleep(r, 0); spins = 0; }
            continue;
        }
        spins = 0;
        uint64_t t0 = host_plat_time_ns();
        int park = 0;
        while (r != w && !park) {
            u32 h = sMtQ[r & MTQ_M], c = h & 0xff, n = h >> 8;
            if (c == 0xf3) {   // raw GX FIFO words: decode here (w_word)
                for (u32 i = 0; i < n; i++) w_word(sMtQ[(r + 1 + i) & MTQ_M]);
                r += 1 + n;
                park = sFlushPending;
                continue;
            }
            const u32 *p = &sMtQ[(r + 1) & MTQ_M];
            if (((r + 1) & MTQ_M) + n > MTQ_N) {
                for (u32 i = 0; i < n; i++) tmp[i] = sMtQ[(r + 1 + i) & MTQ_M];
                p = tmp;
            }
            switch (c) {
            case 0xf0: gpu3d_write16_now(p[0], (u16)p[1]); break;
            case 0xf1: memcpy(wClip.m, p, sizeof wClip.m); memcpy(wVec.m, p + 16, sizeof wVec.m); sLitGen++; break;
            case 0xf2: memcpy(wTex.m, p, sizeof wTex.m); break;
            case 0x71: sVtx[0] = (s16)p[0]; sVtx[1] = (s16)(p[0] >> 16); sVtx[2] = (s16)p[1]; break;   // result: game thread
            default: execute((u8)c, p); break;
            }
            r += 1 + n;
            park = sFlushPending;
        }
        __atomic_fetch_add(&sMtBusyNs, host_plat_time_ns() - t0, __ATOMIC_RELAXED);
        // Park before publishing R so a game thread that sees the queue drained also sees the pending swap.
        if (park) __atomic_store_n(&sMtParked, 1, __ATOMIC_SEQ_CST);
        __atomic_store_n(&sMtR, r, __ATOMIC_RELEASE);
        if (park) {
            { static int lag = -1; if (lag < 0) lag = getenv("ACWW_GX_LAG") ? atoi(getenv("ACWW_GX_LAG")) : 0; if (lag) usleep(lag * 1000); }
            int s = 0;
            while (__atomic_load_n(&sMtParked, __ATOMIC_ACQUIRE)) {
                if (++s < 32) sched_yield();
                else mt_sleep(0, 1);
            }
        }
    }
    return 0;
}
// Game thread: wait until the worker has executed everything queued (or is parked on a swap).
static inline void mt_publish(void) {
    if (sMtW != sMtWDone) mt_publish_w(sMtWDone);
}
extern volatile unsigned char gHostPhase;   // phase sampler (host.h): 6 = PH_WAIT
// Game-thread waits for the GX worker, by reason (0 box test, 1 RAM-count read, 2 vblank commit, 3 other), for
// the "gx waits" log line.
static unsigned long long sMtWaitNs[4], sMtWaitN[4], sBoxTests;
static void mt_sync_r(int why) {
    unsigned char ph = gHostPhase;
    gHostPhase = 6;
    uint64_t t0 = host_plat_time_ns();
    mt_sync_body();
    sMtWaitNs[why] += host_plat_time_ns() - t0;
    sMtWaitN[why]++;
    gHostPhase = ph;
}
static void mt_sync(void) { mt_sync_r(3); }
void gpu3d_wait_stats(char *buf, int n, int frames) {
    if (frames < 1) frames = 1;
    snprintf(buf, (size_t)n, "gx waits (avg/frame): box tests %llu (waited %llu x, %llu us), ram-count reads %llu x %llu us, "
             "commit %llu us, other %llu x %llu us", sBoxTests / frames, sMtWaitN[0] / frames, sMtWaitNs[0] / frames / 1000,
             sMtWaitN[1] / frames, sMtWaitNs[1] / frames / 1000, sMtWaitNs[2] / frames / 1000, sMtWaitN[3] / frames,
             sMtWaitNs[3] / frames / 1000);
    memset(sMtWaitNs, 0, sizeof sMtWaitNs); memset(sMtWaitN, 0, sizeof sMtWaitN); sBoxTests = 0;
}
static void raw_close(int publishNow);
static void mt_sync_body(void) {
    int s = 0;
    if (gHostGxStageN) HostGx_StageFlush();
    raw_close(1);
    mt_publish();
    while (__atomic_load_n(&sMtR, __ATOMIC_ACQUIRE) != sMtWDone && !__atomic_load_n(&sMtParked, __ATOMIC_ACQUIRE)) mt_pause(&s);
}
// Make room for `words` more words; 0 = the ring is full behind a pending swap (drop the command, as a full FIFO
// would).
static int mt_reserve(u32 words) {
    u32 w = sMtWLocal;
    if (w - sMtRSeen + words <= MTQ_N) return 1;
    int s = 0;
    unsigned char ph = gHostPhase;
    mt_publish();
    gHostPhase = 6;   // ring full: waiting for the worker
    while (w - (sMtRSeen = __atomic_load_n(&sMtR, __ATOMIC_ACQUIRE)) + words > MTQ_N) {
        if (__atomic_load_n(&sMtParked, __ATOMIC_ACQUIRE)) { gHostPhase = ph; return 0; }
        mt_pause(&s);
    }
    gHostPhase = ph;
    return 1;
}
static inline __attribute__((always_inline)) void mt_command_done(int publishNow) {
    sMtWDone = sMtWLocal;
    if (publishNow || sMtWDone - sMtW >= MTQ_BATCH) mt_publish_w(sMtWDone);
}
static void mt_push(u32 c, const u32 *p, u32 n) {
    if (!mt_reserve(n + 1)) return;
    u32 w = sMtWLocal;
    sMtQ[w & MTQ_M] = c | n << 8;
    for (u32 i = 0; i < n; i++) sMtQ[(w + 1 + i) & MTQ_M] = p[i];
    sMtWLocal = w + n + 1;
    mt_command_done(c == 0x50);
}
static void mt_dispatch(u8 c, const u32 *p, int n) {
    if (c >= 0x10 && c <= 0x1c) {                 // matrix commands: game thread only
        execute(c, p);
        if (c == 0x12 || c >= 0x14) { if (sMtxMode == 3) sMtTexDirty = 1; else sMtCvDirty = 1; }
        return;
    }
    // BOX_TEST depends only on the clip matrix, which the game thread keeps current in command order: answer it
    // here without waiting for the worker to drain (the game issues one per model for culling; each wait used to
    // serialise the game thread behind all queued geometry).
    if (c == 0x70) { static int syncBox = -1; if (syncBox < 0) syncBox = getenv("ACWW_BOXSYNC") && atoi(getenv("ACWW_BOXSYNC")); sBoxTests++; if (syncBox) mt_sync_r(0); box_test(p); return; }
    if (c == 0x72) { vec_test(p[0]); return; }
    if (!is_command(c)) return;
    if (sMtSkip && c >= 0x23 && c <= 0x28) return;
    if (c == 0x71) {                              // result here; the worker only latches the vertex registers
        update_clip();
        s64 x = (s16)p[0], y = (s16)(p[0] >> 16), z = (s16)p[1];
        for (int i = 0; i < 4; i++)
            sPosResult[i] = (s32)((x * sClip.m[i] + y * sClip.m[4 + i] + z * sClip.m[8 + i] + 0x1000 * (s64)sClip.m[12 + i]) >> 12);
    }
    if (sMtCvDirty) {
        u32 b[32];
        update_clip();
        memcpy(b, sClip.m, sizeof sClip.m);
        memcpy(b + 16, sVec.m, sizeof sVec.m);
        mt_push(0xf1, b, 32);
        sMtCvDirty = 0;
    }
    if (sMtTexDirty) { mt_push(0xf2, (const u32 *)sTex.m, 16); sMtTexDirty = 0; }
    if (c == 0x50) {
        if (sMtSkip) sMtSkipped++;
        else { sMtSwapsQueued++; mt_push(c, p, (u32)n); }
        sMtSkip = sMtAsync && sMtSwapsQueued - sMtSwapsDone >= 3;
        return;
    }
    mt_push(c, p, (u32)n);
}
// Pass-through commands (everything the game thread does not execute itself) are streamed straight into the ring:
// the header when the command starts, then one store per parameter word, instead of collecting the parameters
// and copying them again. The command is published only once complete (sMtWDone).
static u32 sStrLeft;   // parameters still to come for the command being streamed
static int sStrMode;   // 0: none, 1: streaming into the ring, 2: dropping (ring full behind a pending swap)
static inline int mt_streamable(u8 c) {
    // (Not while the pipelined mode is dropping a frame's vertices: mt_dispatch does the dropping.)
    return !sMtSkip && ((c >= 0x20 && c <= 0x2b) || (c >= 0x30 && c <= 0x34) || c == 0x40 || c == 0x60);
}
static void mt_stream_begin(u8 c, u32 n) {
    if (sMtCvDirty) {   // the worker needs the matrices current before this command (as mt_dispatch does)
        u32 b[32];
        update_clip();
        memcpy(b, sClip.m, sizeof sClip.m);
        memcpy(b + 16, sVec.m, sizeof sVec.m);
        mt_push(0xf1, b, 32);
        sMtCvDirty = 0;
    }
    if (sMtTexDirty) { mt_push(0xf2, (const u32 *)sTex.m, 16); sMtTexDirty = 0; }
    sStrLeft = n;
    if (!mt_reserve(n + 1)) { sStrMode = 2; return; }
    sMtQ[sMtWLocal++ & MTQ_M] = c | n << 8;
    sStrMode = 1;
}
static inline __attribute__((always_inline)) void mt_stream_param(u32 param) {
    if (sStrMode == 1) sMtQ[sMtWLocal++ & MTQ_M] = param;
    if (--sStrLeft == 0) {
        if (sStrMode == 1) mt_command_done(0);
        sStrMode = 0;
        sExecCmd = -1;
    }
}
static void mt_entry(u8 cmd, u32 param) {
    if (sRawOpen) raw_close(0);   // command-port write after raw FIFO words: keep the order
    if (sStrMode) { mt_stream_param(param); return; }
    if (sExecCmd < 0) {
        int need = kParams[cmd];
        if (need && mt_streamable(cmd)) {
            sExecCmd = cmd;
            if (!sMtCvDirty && !sMtTexDirty && sMtWLocal - sMtRSeen + (u32)need + 1 <= MTQ_N) {   // common case inline
                sStrLeft = (u32)need;
                sMtQ[sMtWLocal++ & MTQ_M] = cmd | (u32)need << 8;
                sStrMode = 1;
                mt_stream_param(param);
                return;
            }
            mt_stream_begin(cmd, (u32)need);
            mt_stream_param(param);   // this entry carries the first parameter
            return;
        }
        sExecCmd = cmd;
        sExecCount = 0;
    }
    int need = kParams[(u8)sExecCmd];
    if (need) sExecParams[sExecCount++] = param;
    if (sExecCount >= need) {
        u8 c = (u8)sExecCmd;
        sExecCmd = -1;
        mt_dispatch(c, sExecParams, need);
    }
}

// The common cases of mt_entry, inline: the next parameter of a command being streamed into the ring, and the
// first parameter of a pass-through command when nothing is pending (no matrix snapshot due, room in the ring).
// Same ring contents and state as mt_entry; returns 0 when mt_entry must handle the word.
static inline __attribute__((always_inline)) int mt_fast(u8 c, u32 v) {
    if (sStrMode == 1) {
        sMtQ[sMtWLocal++ & MTQ_M] = v;
        if (--sStrLeft == 0) { mt_command_done(0); sStrMode = 0; sExecCmd = -1; }
        return 1;
    }
    if (sStrMode == 0 && sExecCmd < 0) {
        u32 k = kParams[c];
        if (k && !sMtCvDirty && !sMtTexDirty && mt_streamable(c) && sMtWLocal - sMtRSeen + k + 1 <= MTQ_N) {
            u32 w = sMtWLocal;
            sMtQ[w & MTQ_M] = c | k << 8;
            sMtQ[(w + 1) & MTQ_M] = v;
            sMtWLocal = w + 2;
            if (k == 1) mt_command_done(0);
            else { sStrLeft = k - 1; sStrMode = 1; sExecCmd = c; }
            return 1;
        }
    }
    return 0;
}
// Worker mode: the GX FIFO decoder (gxfifo_write/pack_advance) calling mt_entry directly, so the whole per-word
// path inlines. Same logic: a word with no parameters pending holds up to four packed commands.
static inline __attribute__((always_inline)) void pack_advance_mt(void) {
    while (sPackLeft > 0) {
        u8 c = sPackCmds & 0xff;
        if (sPackCmds == 0) { sPackLeft = 0; break; }
        if (kParams[c]) { sPackParamsLeft = kParams[c]; return; }
        if (c) mt_entry(c, 0);
        sPackCmds >>= 8;
        sPackLeft--;
    }
    sPackParamsLeft = 0;
}
static inline __attribute__((always_inline)) void gxfifo_write_mt(u32 v) {
    if (sPackParamsLeft == 0) {
        if (v == 0) { mt_entry(0, 0); return; }
        sPackCmds = v;
        sPackLeft = 4;
        pack_advance_mt();
    } else {
        u8 c = sPackCmds & 0xff;
        if (!mt_fast(c, v)) mt_entry(c, v);
        if (--sPackParamsLeft == 0) {
            sPackCmds >>= 8;
            sPackLeft--;
            pack_advance_mt();
        }
    }
}
// Worker mode, a write to a GX command port (0x04000440-0x5ff: the game's immediate G3_* calls - most of the GX
// words). Same as fifo_put -> mt_entry (the non-worker drain() after it has nothing to do in worker mode), with
// the one-parameter pass-through case written straight into the ring.
static inline __attribute__((always_inline)) void gx_port_mt(u8 c, u32 v) {
    if (sRawOpen) raw_close(0);
    if (!mt_fast(c, v)) mt_entry(c, v);
}
// ---- raw GX FIFO ---------------------------------------------------------------------------------------------
// The game thread no longer decodes GX FIFO words into per-command ring records. It copies the raw words into
// the ring (runs of up to RAW_MAX words, record 0xf3) and only follows the packing to find the commands it must
// execute itself: matrix commands (it keeps the matrix stack: GXSTAT, matrix reads, tests) and BOX/POS/VEC_TEST
// (answered here). Before the first parameter of any other command, if a matrix changed, it ends the run and
// inserts the matrix snapshot (0xf1/0xf2) - so the worker, which decodes the same words in the same order
// (w_word), executes that command with the right matrices. A SWAP_BUFFERS ends its run, so the worker parks
// exactly after it. Synchronous commit only (the pipelined mode's frameskip keeps the record path).
enum { RAW_MAX = 256 };
static int sRawOn;
static u32 sRawHdr, sRawN;
static u32 gPackCmds, gParams[32];
static int gPackLeft, gParamsLeft, gCnt, gLocal;
static u8 gCmd;
HOST_STATE_KEEP(sRawOn) HOST_STATE_KEEP(sRawOpen) HOST_STATE_KEEP(sRawHdr) HOST_STATE_KEEP(sRawN) HOST_STATE_KEEP(gPackCmds)
HOST_STATE_KEEP(gPackLeft) HOST_STATE_KEEP(gParamsLeft) HOST_STATE_KEEP(wPackCmds) HOST_STATE_KEEP(wPackLeft) HOST_STATE_KEEP(wParamsLeft)
static void raw_close(int publishNow) {
    if (!sRawOpen) { if (publishNow) mt_publish(); return; }
    sMtQ[sRawHdr & MTQ_M] = 0xf3 | sRawN << 8;
    sRawOpen = 0;
    mt_command_done(publishNow);
}
static inline void raw_put(u32 v) {
    if (!sRawOpen) {
        if (!mt_reserve(RAW_MAX + 1)) return;   // (only when the ring is full behind a parked swap: async mode)
        sRawHdr = sMtWLocal++;
        sRawN = 0;
        sRawOpen = 1;
    }
    sMtQ[sMtWLocal++ & MTQ_M] = v;
    if (++sRawN == RAW_MAX) raw_close(0);
}
static void raw_snapshots(void) {
    raw_close(0);
    if (sMtCvDirty) {
        u32 b[32];
        update_clip();
        memcpy(b, sClip.m, sizeof sClip.m);
        memcpy(b + 16, sVec.m, sizeof sVec.m);
        mt_push(0xf1, b, 32);
        sMtCvDirty = 0;
    }
    if (sMtTexDirty) { mt_push(0xf2, (const u32 *)sTex.m, 16); sMtTexDirty = 0; }
}
static void raw_begin(u8 c) {
    gCmd = c;
    gLocal = (c >= 0x10 && c <= 0x1c) || (c >= 0x70 && c <= 0x72);
    if (!gLocal && (sMtCvDirty | sMtTexDirty)) raw_snapshots();
}
static void raw_cmd_done(u8 c, const u32 *p) {
    if (c >= 0x10 && c <= 0x1c) {
        execute(c, p);
        if (c == 0x12 || c >= 0x14) { if (sMtxMode == 3) sMtTexDirty = 1; else sMtCvDirty = 1; }
    } else if (c == 0x70) { sBoxTests++; box_test(p); }
    else if (c == 0x72) vec_test(p[0]);
    else if (c == 0x71) {
        update_clip();
        s64 x = (s16)p[0], y = (s16)(p[0] >> 16), z = (s16)p[1];
        for (int i = 0; i < 4; i++)
            sPosResult[i] = (s32)((x * sClip.m[i] + y * sClip.m[4 + i] + z * sClip.m[8 + i] + 0x1000 * (s64)sClip.m[12 + i]) >> 12);
    } else if (c == 0x50) { sMtSwapsQueued++; raw_close(1); }
}
static void raw_advance(void) {
    while (gPackLeft > 0) {
        u8 c = gPackCmds & 0xff;
        if (gPackCmds == 0) { gPackLeft = 0; break; }
        raw_begin(c);
        if (kParams[c]) { gParamsLeft = kParams[c]; gCnt = 0; return; }
        raw_cmd_done(c, kZeroParams);
        gPackCmds >>= 8;
        gPackLeft--;
    }
    gParamsLeft = 0;
}
static inline void gx_raw_word(u32 v) {
    if (gParamsLeft == 0) {
        raw_put(v);
        if (v == 0) return;
        gPackCmds = v;
        gPackLeft = 4;
        raw_advance();
        return;
    }
    if (gLocal) gParams[gCnt++] = v;
    raw_put(v);
    if (--gParamsLeft == 0) {
        if (gLocal || gCmd == 0x50) raw_cmd_done(gCmd, gParams);
        gPackCmds >>= 8;
        gPackLeft--;
        raw_advance();
    }
}

// Called once at start-up, before any command. async: the vblank commit does not wait for the worker.
void gpu3d_mt_enable(int on, int async) {
    if (!on || sMtStarted) return;
    sMtQ = malloc(sizeof(u32) * MTQ_N);
    if (!sMtQ) return;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setstacksize(&at, 1 << 20);
    if (pthread_create(&sMtThread, &at, mt_worker, 0) != 0) { free(sMtQ); sMtQ = 0; return; }
    sMtStarted = 1;
    sMtAsync = async;
    gClipM = &wClip; gVecM = &wVec; gTexM = &wTex;
    sMtCvDirty = sMtTexDirty = 1;
    // Raw FIFO forwarding: measured (host, -O3) to save the game thread ~1% while adding ~25% to the worker, so
    // opt-in (ACWW_GXRAW=1).
    sRawOn = !async && getenv("ACWW_GXRAW") && atoi(getenv("ACWW_GXRAW"));
    sMT = 1;
    fprintf(stderr, "gpu3d: GX worker thread on (%s commit%s)\n", async ? "pipelined" : "synchronous",
            sRawOn ? ", raw FIFO to the worker" : "");
}
unsigned long long gpu3d_mt_skipped(void) { unsigned long long n = sMtSkipped; sMtSkipped = 0; return n; }
unsigned long long gpu3d_mt_busy_ns(void) { return __atomic_exchange_n(&sMtBusyNs, 0, __ATOMIC_RELAXED); }

static void exec_entry(Entry e) {
    if (sExecCmd < 0) {
        sExecCmd = e.cmd;
        sExecCount = 0;
    }
    int need = kParams[(u8)sExecCmd];
    if (need) sExecParams[sExecCount++] = e.param;
    if (sExecCount >= need) {
        u8 c = (u8)sExecCmd;
        sExecCmd = -1;
        timing_before(c);
        execute(c, sExecParams);
        timing_after(c);
    } else {
        if (sExecCmd == 0x23 && sExecCount == 1) vertex_entry_first();
        else add_cycles(1);
    }
}

static void drain(void) {
    while (sGeomOn && !sFlushPending && sPipeN) exec_entry(fifo_get());
}

static int cmp_poly(const void *a, const void *b) {
    const G3Polygon *p = *(G3Polygon *const *)a, *q = *(G3Polygon *const *)b;
    if (p->sortKey != q->sortKey) return p->sortKey < q->sortKey ? -1 : 1;
    return p->index - q->index;
}
static void do_swap(void) {
    G3RenderState *r = &gG3Render;
    r->npoly = sNumPolys;
    r->wbuffer = sFlushParam >> 1 & 1;
    r->manualSort = sFlushParam & 1;
    r->yofs = G3_YOFS;
    if (r->wbuffer) g3_feature(F_WBUFFER);
    else if (r->npoly) g3_feature(F_ZBUFFER_POLYS);
    if (r->manualSort) g3_feature(F_MANUAL_SORT);
    for (int i = 0; i < sNumPolys; i++) {
        G3Polygon *p = &sPolyRam[sBank][i];
        if (r->manualSort && p->translucent) p->sortKey = 1u << 21;
        r->poly[i] = p;
    }
    if (gG3ClipSpaceOut) {
        // GL renderer: every sortKey is 0 (opaque) or 1 << 21 (translucent), so sorting by (sortKey, index) is a
        // stable partition - opaque polygons, then translucent ones, each in submission order. One linear pass.
        int n = 0;
        for (int i = 0; i < sNumPolys; i++)
            if (!sPolyRam[sBank][i].sortKey) r->poly[n++] = &sPolyRam[sBank][i];
        for (int i = 0; i < sNumPolys; i++)
            if (sPolyRam[sBank][i].sortKey) r->poly[n++] = &sPolyRam[sBank][i];
    } else
        qsort(r->poly, (size_t)sNumPolys, sizeof r->poly[0], cmp_poly);
    r->dispcnt = sDispCnt;
    r->clearColor = sRegs[(0x350 - 0x320) / 2];
    r->clearAttrHi = sRegs[(0x352 - 0x320) / 2];
    r->clearDepth = sRegs[(0x354 - 0x320) / 2] & 0x7fff;
    r->clearOffset = sRegs[(0x356 - 0x320) / 2];
    r->alphaRef = sRegs[(0x340 - 0x320) / 2] & 31;
    memcpy(r->toon, &sRegs[(0x380 - 0x320) / 2], sizeof r->toon);
    memcpy(r->edge, &sRegs[(0x330 - 0x320) / 2], sizeof r->edge);
    r->fogColor = sRegs[(0x358 - 0x320) / 2] | (u32)sRegs[(0x35a - 0x320) / 2] << 16;
    r->fogOffset = sRegs[(0x35c - 0x320) / 2] & 0x7fff;
    for (int i = 0; i < 32; i++) r->fogTable[i] = (u8)(sRegs[(0x360 - 0x320) / 2 + i / 2] >> (8 * (i & 1))) & 0x7f;
    {   // ACWW_G3HASH: per-frame hash of what the GL renderer reads (A/B check of the geometry paths)
        static int on = -1;
        if (on < 0) on = getenv("ACWW_G3HASH") != 0;
        if (on) {
            u64 h = 1469598103934665603ull;
#define H(x) (h = (h ^ (u64)(u32)(x)) * 1099511628211ull)
            for (int i = 0; i < r->npoly; i++) {
                const G3Polygon *p = r->poly[i];
                H(p->nv); H(p->attr); H(p->texparam); H(p->texpal); H(p->translucent); H(p->sortKey);
                for (int k = 0; k < p->nv && k < 4; k++) {
                    const s32 *pos = p->hv[0] ? p->hv[k]->pos : p->v[k]->pos;
                    const s32 *col = p->hv[0] ? p->hv[k]->color : p->v[k]->color;
                    const s16 *tx = p->hv[0] ? p->hv[k]->tex : p->v[k]->tex;
                    H(pos[0]); H(pos[1]); H(pos[2]); H(pos[3]); H(col[0]); H(col[1]); H(col[2]); H(tx[0]); H(tx[1]);
                }
            }
#undef H
            fprintf(stderr, "g3hash %llu %d %016llx\n", (unsigned long long)sFrame, r->npoly, (unsigned long long)h);
        }
    }
    sBank ^= 1;
    sNumPolys = sNumVerts = sNumOpaque = 0;
    sPoolN = 0;
    sStripValid = 0;
    sFlushPending = 0;
}

void gpu3d_reset(void) {
    sGeomOn = sRenderOn = 0;
    sDispCnt = sGxStat = 0;
    memset(sRegs, 0, sizeof sRegs);
    sMtxMode = 0;
    mtx_identity(&sProj); mtx_identity(&sPos); mtx_identity(&sVec); mtx_identity(&sTex);
    sProjSP = sTexSP = sPosSP = 0;
    sClipDirty = 1;
    sPipeHead = sPipeN = sFifoHead = sFifoN = 0;
    sPushPopQueued = sTestQueued = 0;
    sFlushPending = 0;
    sPackLeft = sPackParamsLeft = 0;
    sExecCmd = -1;
    sNumPolys = sNumVerts = sNumOpaque = 0;
    sPoolN = 0;
    if (getenv("ACWW_G3CLIPOUT")) gG3ClipSpaceOut = 1;   // host test of the GL geometry paths (no 3D drawn)
    memset(&gG3Render, 0, sizeof gG3Render);
    if (sFeaturesOn < 0) sFeaturesOn = getenv("ACWW_GPU3D_FEATURES") ? atoi(getenv("ACWW_GPU3D_FEATURES")) : 0;
}

void gpu3d_write8(uint32_t addr, uint8_t v) {
    if (addr >= 0x04000240 && addr < 0x04000247) gG3VramCnt[addr - 0x04000240] = v;
    else if (addr >= 0x04000248 && addr < 0x0400024a) gG3VramCnt[addr - 0x04000248 + 7] = v;
    else {
        u16 h = addr & 1 ? (u16)(v << 8) : v;
        (void)h;
    }
}
void gpu3d_write16(uint32_t addr, uint16_t v) {
    // With the GX worker, registers the swap latches travel in-stream so they stay ordered with the geometry.
    if (sMT && addr != 0x04000304) {
        if (addr == 0x04000060 || (addr >= 0x04000320 && addr < 0x040003c0) || addr == 0x04000610) {
            u32 b[2] = {addr, v};
            mt_push(0xf0, b, 2);
        }
        return;
    }
    gpu3d_write16_now(addr, v);
}
static void gpu3d_write16_now(uint32_t addr, uint16_t v) {
    if (addr == 0x04000060)
        sDispCnt = (sDispCnt & 0x3000u & ~(u32)(v & 0x3000)) | (v & 0x4fff);
    else if (addr == 0x04000304) { sRenderOn = v >> 2 & 1; sGeomOn = v >> 3 & 1; drain(); }
    else if (addr >= 0x04000320 && addr < 0x040003c0) sRegs[(addr - 0x04000320) / 2] = v;
    else if (addr == 0x04000610) sDot1Depth = v & 0x7fff;
    else if (addr >= 0x04000400 && addr < 0x04000600) { /* (the host writes these 32-bit) */ }
}
extern unsigned int gHostGxStageN;
extern void HostGx_StageFlush(void);
// The queued GX stores (hw.c / io_access.h), in order: the same as gpu3d_write32 per pair.
void gpu3d_write_pairs(const u32 *p, u32 n) {
    for (u32 i = 0; i < n; i++, p += 2) {
        u32 a = p[0], v = p[1];
        if (sMT && sGeomOn) {
            if (a - 0x04000400u < 0x40u) { if (sRawOn) gx_raw_word(v); else gxfifo_write_mt(v); continue; }
            if (a - 0x04000440u < 0x1c0u) { gx_port_mt((u8)((a - 0x04000400u) >> 2), v); continue; }
        }
        gpu3d_write32(a, v);
    }
}
void gpu3d_write32(uint32_t addr, uint32_t v) {
    if (gHostGxStageN) HostGx_StageFlush();   // queued stores go first (host-side callers)
    if (sMT && sGeomOn && addr - 0x04000400u < 0x40u) { if (sRawOn) gx_raw_word(v); else gxfifo_write_mt(v); return; }   // worker mode, GX FIFO
    if (sMT && sGeomOn && addr - 0x04000440u < 0x1c0u) { gx_port_mt((u8)((addr - 0x04000400u) >> 2), v); return; }   // command ports
    if (addr >= 0x04000400 && addr < 0x04000600 && !sGeomOn) { g3_feature(F_GEOM_OFF_WRITE); return; }
    if (addr >= 0x04000400 && addr < 0x04000440) gxfifo_write(v);
    else if (addr >= 0x04000440 && addr < 0x04000600) {
        u8 cmd = (u8)((addr - 0x04000400) >> 2);
        if (is_command(cmd)) fifo_put(cmd, v);
    } else if (addr == 0x04000600) {
        if (v & 0x8000) {
            sGxStat &= ~0x8000u;
            sProjSP = 0;
            sTexSP = 0;
        }
        sGxStat = (sGxStat & ~0xc0000000u) | (v & 0xc0000000u);
    } else {
        gpu3d_write16(addr, (u16)v);
        gpu3d_write16(addr + 2, (u16)(v >> 16));
        return;
    }
    drain();
}

static u32 gxstat(void) {
    u32 s = sGxStat & (2u | 0x8000u | 0xc0000000u);
    if (sTestQueued) s |= 1;
    s |= (u32)(sPosSP & 31) << 8;
    s |= (u32)(sProjSP & 1) << 13;
    if (sPushPopQueued) s |= 1u << 14;
    s |= (u32)sFifoN << 16;
    if (sFifoN < 128) s |= 1u << 25;
    if (sFifoN == 0) s |= 1u << 26;
    if (sPipeN || pipelines_busy()) s |= 1u << 27;
    return s;
}

// A block of words to one GX port (display-list DMA): the per-word loop runs here, fully inlined in worker mode.
// A block of GX FIFO words in worker mode (display-list DMA, NNS's MIi_CpuSend32): the packed-command decoder of
// gxfifo_write_mt + mt_fast with its state in locals, so the common words (headers, parameters of pass-through
// commands being streamed into the ring, parameters of a matrix command being collected) cost a few instructions.
// Anything else - a zero-parameter command, the start or end of a matrix command, a matrix snapshot due, the ring
// looking full - writes the state back and takes the general path for that word. Same ring contents and state.
static void fifo_block_mt(const u32 *src, u32 n) {
    u32 i = 0;
    while (i < n) {
        u32 cmds = sPackCmds, strLeft = sStrLeft, w = sMtWLocal;
        int left = sPackLeft, pleft = sPackParamsLeft, strMode = sStrMode, execCmd = sExecCmd;
        const u32 rseen = sMtRSeen;
        const int blocked = sMtCvDirty | sMtTexDirty | sMtSkip | sRawOpen;
        int slow = 0;
        for (; i < n; i++) {
            u32 v = src[i];
            if (pleft == 0) {                         // a header word
                if (v == 0) { if (strMode || execCmd >= 0) { slow = 1; break; } continue; }
                cmds = v;
                left = 4;
            } else {                                  // a parameter of command cmds & 0xff
                u8 c = cmds & 0xff;
                if (strMode == 1) {
                    sMtQ[w++ & MTQ_M] = v;
                    if (--strLeft == 0) {
                        strMode = 0;
                        execCmd = -1;
                        sMtWDone = w;
                        if (w - sMtW >= MTQ_BATCH) mt_publish_w(w);
                    }
                } else if (strMode == 0 && execCmd < 0) {
                    u32 k = kParams[c];
                    if (blocked || !((c >= 0x20 && c <= 0x2b) || (c >= 0x30 && c <= 0x34) || c == 0x40 || c == 0x60) ||
                        w - rseen + k + 1 > MTQ_N) { slow = 1; break; }
                    sMtQ[w & MTQ_M] = c | k << 8;
                    sMtQ[(w + 1) & MTQ_M] = v;
                    w += 2;
                    if (k == 1) {
                        sMtWDone = w;
                        if (w - sMtW >= MTQ_BATCH) mt_publish_w(w);
                    } else { strLeft = k - 1; strMode = 1; execCmd = c; }
                } else if (strMode == 0 && execCmd >= 0 && sExecCount + 1 < kParams[(u8)execCmd]) {
                    sExecParams[sExecCount++] = v;    // a parameter of a matrix/test command, not the last
                } else { slow = 1; break; }
                if (--pleft) continue;
                cmds >>= 8;
                left--;
            }
            // pack_advance: on to the next packed command
            while (left > 0) {
                u8 c = cmds & 0xff;
                if (cmds == 0) { left = 0; break; }
                if (kParams[c]) { pleft = kParams[c]; break; }
                if (c) { pleft = 0; slow = 2; break; }   // a zero-parameter command: general path
                cmds >>= 8;
                left--;
            }
            if (slow) { i++; break; }
            if (left <= 0) pleft = 0;
        }
        sPackCmds = cmds; sPackLeft = left; sPackParamsLeft = pleft;
        sStrMode = strMode; sStrLeft = strLeft; sExecCmd = execCmd; sMtWLocal = w;
        if (slow == 1) { gxfifo_write_mt(src[i]); i++; }
        else if (slow == 2) pack_advance_mt();   // runs the zero-parameter command(s) and continues the packing
    }
}
void gpu3d_write_block(uint32_t addr, const uint32_t *src, uint32_t n) {
    if (gHostGxStageN) HostGx_StageFlush();
    if (sMT && sGeomOn && addr - 0x04000400u < 0x40u) {
        static int fast = -1;
        if (fast < 0) fast = !(getenv("ACWW_GXBLOCK") && atoi(getenv("ACWW_GXBLOCK")) == 0);
        if (sRawOn) for (uint32_t i = 0; i < n; i++) gx_raw_word(src[i]);
        else if (fast) fifo_block_mt(src, n);
        else for (uint32_t i = 0; i < n; i++) gxfifo_write_mt(src[i]);
        return;
    }
    for (uint32_t i = 0; i < n; i++) gpu3d_write32(addr, src[i]);
}
uint32_t gpu3d_read32(uint32_t addr) {
    if (gHostGxStageN) HostGx_StageFlush();
    {   // ACWW_GXREADLOG=1: count GX register reads per address (diagnostics; printed every 600 reads)
        static int on = -1; static unsigned cnt[0x100], tot;
        if (on < 0) on = getenv("ACWW_GXREADLOG") != 0;
        if (on) {
            cnt[((addr - 0x04000000) >> 2) & 0xff]++;
            if (++tot % 20000 == 0) { fprintf(stderr, "gxreads:"); for (int i = 0; i < 0x100; i++) if (cnt[i]) fprintf(stderr, " %03x:%u", i * 4, cnt[i]); fprintf(stderr, "\n"); }
        }
    }
    if (sMT) { if (addr == 0x04000604) mt_sync_r(1); }
    else drain();
    if (addr == 0x04000600) return gxstat();
    if (addr == 0x04000604) { g3_feature(F_READ_RAMCOUNT); return (u32)sNumPolys | (u32)sNumVerts << 16; }
    if (addr == 0x04000320) { g3_feature(F_READ_RDLINES); return 46; }
    if (addr >= 0x04000620 && addr < 0x04000630) { g3_feature(F_READ_POSRESULT); return (u32)sPosResult[(addr - 0x04000620) / 4]; }
    if (addr == 0x04000630) { g3_feature(F_READ_VECRESULT); return (u16)sVecResult[0] | (u32)(u16)sVecResult[1] << 16; }
    if (addr == 0x04000634) { g3_feature(F_READ_VECRESULT); return (u16)sVecResult[2]; }
    if (addr >= 0x04000640 && addr < 0x04000680) {
        g3_feature(F_READ_CLIPMTX);
        update_clip();
        return (u32)sClip.m[(addr - 0x04000640) / 4];
    }
    if (addr >= 0x04000680 && addr < 0x040006a4) {
        g3_feature(F_READ_VECMTX);
        int i = (addr - 0x04000680) / 4;
        return (u32)sVec.m[(i / 3) * 4 + i % 3];
    }
    return 0;
}

// Commit the frame's geometry (main thread): bind the texture banks, run feature flags, and swap the double-
// buffered polygon RAM so gG3Render points at the finished bank. Executes no rasterization, so it can run on the
// main thread at the frame boundary while the render thread rasterizes the previously-committed geometry.
// Pipelined commit support (render.c): which geometry frame (1-based count of SWAP_BUFFERS) the last commit
// swapped in (0: none), how many SWAP_BUFFERS the game has issued, and re-binding the texture banks + VRAMCNT
// of the VRAM snapshot that belongs to that geometry frame.
static unsigned sCommitSwapped;
unsigned gpu3d_commit_swapped(void) { return sCommitSwapped; }
unsigned gpu3d_swaps_issued(void) { return sMtSwapsQueued; }
int gpu3d_pipelined(void) { return sMT && sMtAsync; }
void gpu3d_bind_banks(const uint8_t *const bank[9], const uint8_t cnt[9]) {
    gG3Banks = bank;
    memcpy(gG3VramCntFrame, cnt, sizeof gG3VramCntFrame);
}
void gpu3d_commit(const uint8_t *const bank[9]) {
    if (gHostGxStageN) HostGx_StageFlush();
    gG3Banks = bank;
    memcpy(gG3VramCntFrame, gG3VramCnt, sizeof gG3VramCntFrame);   // pairs with the bank snapshot
    gG3BankDirtyFrame = sBankDirtyNext;   // dirty flags belong to this snapshot only
    sBankDirtyNext = 0;
    sFrame++;
    sVtxPipe = sNrmPipe = sPolyPipe = sSlotCtr = 0;
    sSlots = 1;
    if (!(sDispCnt & 1)) g3_feature(F_TEXTURE_OFF);
    if (sDispCnt & 2) g3_feature(F_TOON_HIGHLIGHT);
    if (sDispCnt & 4) g3_feature(F_ALPHA_TEST);
    if (sDispCnt & 8) g3_feature(F_ALPHA_BLEND);
    if (sDispCnt & 16) g3_feature(F_ANTIALIAS);
    if (sDispCnt & 32) g3_feature(F_EDGE_MARK);
    if (sDispCnt & 64) g3_feature(F_FOG_ALPHA_ONLY);
    if (sDispCnt & 128) g3_feature(F_FOG);
    if (sDispCnt & 0x4000) g3_feature(F_REAR_BITMAP);
    if (sMT) {
        // Synchronous: wait for the worker, exactly like the single-threaded path. Pipelined: never wait - swap
        // if the worker has a finished frame parked, otherwise keep showing the previous one (frameskip above
        // keeps the backlog bounded).
        raw_close(1);
        mt_publish();
        if (!sMtAsync) mt_sync_r(2);
        sCommitSwapped = 0;
        if (__atomic_load_n(&sMtParked, __ATOMIC_ACQUIRE)) {
            do_swap();
            sMtSwapsDone++;
            sCommitSwapped = sMtSwapsDone;
            __atomic_store_n(&sMtParked, 0, __ATOMIC_RELEASE);
            mt_wake();
        }
        return;
    }
    if (sFlushPending) {
        do_swap();
        drain();
    }
}

// Rasterize the committed geometry (render thread). Reads gG3Render + the texture banks bound in gpu3d_commit().
void gpu3d_render_committed(void) { g3_render(); }

void gpu3d_frame(const uint8_t *const bank[9]) {
    gpu3d_commit(bank);
    gpu3d_render_committed();
}

const uint32_t *gpu3d_lines(void) { return g3_frame_lines() + 256 * g3_shown_yofs(); }
int gpu3d_top_rows(void) { return g3_shown_yofs(); }
const uint32_t *gpu3d_lines_top(void) { return g3_tall_supported() && g3_frame_tall() ? g3_frame_lines() : 0; }

void gpu3d_state(Gpu3dState *s) {
    if (sMT) mt_sync();
    s->pipe = (u32)sPipeN;
    s->fifo = (u32)sFifoN;
    s->flush = (u32)sFlushPending;
    s->npolys = (u32)sNumPolys;
    s->nverts = (u32)sNumVerts;
    s->mtxmode = (u32)sMtxMode;
    s->possp = (u32)sPosSP;
    s->projsp = (u32)sProjSP;
    s->vp = sVtxPipe; s->np = sNrmPipe; s->pp = sPolyPipe; s->slots = (s32)sSlots; s->slotctr = sSlotCtr;
    memcpy(s->proj, sProj.m, sizeof s->proj);
    memcpy(s->pos, sPos.m, sizeof s->pos);
    memcpy(s->vec, sVec.m, sizeof s->vec);
    memcpy(s->tex, sTex.m, sizeof s->tex);
    memcpy(s->light, sLightDir, sizeof s->light);
    memcpy(s->lightColor, sLightColor, sizeof s->lightColor);
    memcpy(s->material[0], sDiffuse, 3);
    memcpy(s->material[1], sAmbient, 3);
    memcpy(s->material[2], sSpecular, 3);
    memcpy(s->material[3], sEmission, 3);
    memcpy(s->normal, sNormal, sizeof s->normal);
    memcpy(s->vtxColor, sVtxColor, 3);
}

int gpu3d_poly_count(void) { return gG3Render.npoly; }
void gpu3d_debug_vertex(int i, int k, FILE *f) {
    if (i < 0 || i >= gG3Render.npoly) return;
    const G3Polygon *p = gG3Render.poly[i];
    const G3Vertex *v = p->v[k];
    fprintf(f, "clip %d %d %d %d vp %u %u %u %u", v->pos[0], v->pos[1], v->pos[2], v->pos[3], sViewport[0],
            sViewport[3], sViewport[4], sViewport[5]);
    G3Vertex t = *v;
    t.screenDone = 0;
    sScreenYofs = p->yofs;
    screen_position(&t);
    fprintf(f, " scr %d %d hi %d %d re %d %d", v->x, v->y, v->hx, v->hy, t.x, t.y);
    if (sLightRecs && v->dbgLight >= 0 && !v->clipped) {
        const LightRec *r = &sLightRecs[v->dbgLight];
        fprintf(f, " light %d %d %d %u %d", r->n[0], r->n[1], r->n[2], r->lights, r->table);
        for (int c = 0; c < 3; c++) fprintf(f, " %d %d %d %d", r->dif[c], r->amb[c], r->spe[c], r->emi[c]);
        for (int l = 0; l < 4; l++)
            fprintf(f, " %d %d %d %d %d %d", r->dir[l][0], r->dir[l][1], r->dir[l][2], r->lc[l][0], r->lc[l][1], r->lc[l][2]);
        const LightRec *a = &sLightRecs[(v->dbgLight - 1) & 0xffff], *b = &sLightRecs[(v->dbgLight + 1) & 0xffff];
        fprintf(f, " adj %d %d %d %d %d %d", a->n[0], a->n[1], a->n[2], b->n[0], b->n[1], b->n[2]);
        fprintf(f, " raw %d %d %d vec", r->raw[0], r->raw[1], r->raw[2]);
        for (int i = 0; i < 9; i++) fprintf(f, " %d", r->vec[i]);
        for (int l = 0; l < 4; l++) { fprintf(f, " lp %u", r->lparam[l]); for (int i = 0; i < 9; i++) fprintf(f, " %d", r->lvec[l][i]); }
    }
    if (v->clipped) {
        fprintf(f, " cl %d", v->dbgClips);
        for (int j = 0; j < 8; j++) fprintf(f, " %d", v->dbgClip[j]);
        fprintf(f, " orig %d", p->norig);
        for (int j = 0; j < p->norig; j++)
            fprintf(f, " %d %d %d %d %d %d %d %d %d", p->orig[j].pos[0], p->orig[j].pos[1], p->orig[j].pos[2],
                    p->orig[j].pos[3], p->orig[j].color[0], p->orig[j].color[1], p->orig[j].color[2], p->orig[j].tex[0],
                    p->orig[j].tex[1]);
    }
}
void gpu3d_debug_poly(int i) {
    if (i < 0 || i >= gG3Render.npoly) return;
    const G3Polygon *p = gG3Render.poly[i];
    fprintf(stderr, "    own polygon %d before clipping:", p->index);
    for (int k = 0; k < p->norig; k++)
        fprintf(stderr, " (%d %d %d %d c%d,%d,%d t%d,%d)", p->orig[k].pos[0], p->orig[k].pos[1], p->orig[k].pos[2],
                p->orig[k].pos[3], p->orig[k].color[0], p->orig[k].color[1], p->orig[k].color[2], p->orig[k].tex[0],
                p->orig[k].tex[1]);
    fprintf(stderr, "\n    after:");
    for (int k = 0; k < p->nv; k++)
        fprintf(stderr, " (%d %d %d %d)", p->v[k]->pos[0], p->v[k]->pos[1], p->v[k]->pos[2], p->v[k]->pos[3]);
    fprintf(stderr, "\n");
}
int gpu3d_poly_get(int i, Gpu3dVertexOut *v, uint32_t *attr, uint32_t *texparam, uint32_t *texpal, int *flags) {
    if (i < 0 || i >= gG3Render.npoly) return 0;
    const G3Polygon *p = gG3Render.poly[i];
    for (int k = 0; k < p->nv; k++) {
        for (int c = 0; c < 4; c++) v[k].clip[c] = p->v[k]->pos[c];
        v[k].x = p->v[k]->x;
        v[k].y = p->v[k]->y;
        v[k].z = p->z[k];
        v[k].w = p->w[k];
        v[k].r = p->v[k]->color[0];
        v[k].g = p->v[k]->color[1];
        v[k].b = p->v[k]->color[2];
        v[k].s = p->v[k]->tex[0];
        v[k].t = p->v[k]->tex[1];
        v[k].clipped = p->v[k]->clipped;
    }
    *attr = p->attr;
    *texparam = p->texparam;
    *texpal = p->texpal;
    *flags = p->facing | p->translucent << 1 | p->shadowMask << 2 | p->shadow << 3;
    return p->nv;
}

void gpu3d_print_features(void) {
    static const char *const kNames[] = {
        [F_LIGHT] = "lighting", [F_SHININESS_TABLE] = "shininess table", [F_CLIP] = "clipped polygons",
        [F_FARCLIP_HIDE] = "far-plane polygons hidden", [F_CULLED] = "culled polygons",
        [F_ONEDOT_HIDE] = "1-dot polygons hidden", [F_TRANSLUCENT] = "translucent polygons",
        [F_WIREFRAME] = "wireframe (alpha 0)", [F_DEPTH_EQUAL] = "depth test equal", [F_FOG_POLY] = "fog per polygon",
        [F_WBUFFER] = "W-buffering", [F_ZBUFFER_POLYS] = "Z-buffering with polygons", [F_MANUAL_SORT] = "manual translucent sort", [F_STACK_ERROR] = "matrix stack error",
        [F_RAM_OVERFLOW] = "polygon/vertex RAM overflow", [F_FIFO_QUEUED] = "commands queued behind a swap",
        [F_SWAP_QUEUED] = "a swap queued behind a swap", [F_FIFO_FULL] = "command FIFO full (writes dropped)", [F_TEX_REPEAT] = "texture repeat", [F_TEX_FLIP] = "texture flip",
        [F_TEX_CLAMP] = "texture clamp", [F_COLOR0_TRANSPARENT] = "palette colour 0 transparent",
        [F_TRANS_DEPTH_UPDATE] = "translucent depth update", [F_LINE_POLY] = "line polygons",
        [F_TEXTURE_OFF] = "texture mapping off", [F_TOON_HIGHLIGHT] = "highlight shading (DISP3DCNT bit 1)",
        [F_ALPHA_TEST] = "alpha test", [F_ALPHA_BLEND] = "alpha blending", [F_ANTIALIAS] = "anti-aliasing",
        [F_EDGE_MARK] = "edge marking", [F_FOG_ALPHA_ONLY] = "fog alpha only", [F_FOG] = "fog", [F_REAR_BITMAP] = "rear-plane bitmap",
        [F_BOX_TEST] = "box test", [F_POS_TEST] = "position test", [F_VEC_TEST] = "vector test",
        [F_READ_CLIPMTX] = "clip matrix read", [F_READ_VECMTX] = "vector matrix read", [F_READ_RAMCOUNT] = "RAM count read",
        [F_READ_POSRESULT] = "position test result read", [F_READ_VECRESULT] = "vector test result read",
        [F_READ_RDLINES] = "RDLINES_COUNT read", [F_GEOM_OFF_WRITE] = "commands while the geometry engine is off",
        [F_SHADOW_MASK] = "shadow mask polygons", [F_SHADOW] = "shadow polygons",
    };
    static const char *const kModes[] = {"modulation", "decal", "toon/highlight", "shadow"};
    static const char *const kFmts[] = {"none", "A3I5", "4-colour", "16-colour", "256-colour", "4x4 compressed", "A5I3", "direct"};
    static const char *const kGen[] = {"none", "texcoord", "normal", "vertex"};
    fprintf(stderr, "gpu3d features (the first frame that used each):\n");
    for (int i = 0; i < F_COUNT; i++) {
        if (!sFeatureFrame[i]) continue;
        char b[64];
        const char *n;
        if (i < 0x80) { snprintf(b, sizeof b, "command %02x", i); n = b; }
        else if (i < F_TEXFMT) { snprintf(b, sizeof b, "polygon mode %s", kModes[i - F_POLYMODE]); n = b; }
        else if (i < F_TEXGEN) { snprintf(b, sizeof b, "texture format %s", kFmts[i - F_TEXFMT]); n = b; }
        else if (i < F_LIGHT) { snprintf(b, sizeof b, "texture coordinates from %s", kGen[i - F_TEXGEN]); n = b; }
        else n = kNames[i] ? kNames[i] : "?";
        fprintf(stderr, "  %-44s frame %llu\n", n, (unsigned long long)sFeatureFrame[i] - 1);
    }
}

void gpu3d_debug_regs(FILE *f) {
    const G3RenderState *r = &gG3Render;
    fprintf(f, "regs %x %x %x %x %x %x %x", r->dispcnt, r->clearColor | (u32)r->clearAttrHi << 16, r->clearDepth, r->alphaRef,
            r->fogColor, r->fogOffset, r->wbuffer);
    for (int i = 0; i < 32; i++) fprintf(f, " %d", r->fogTable[i]);
    fprintf(f, "\n");
}
