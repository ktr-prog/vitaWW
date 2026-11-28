
#include <stdio.h>
#include <stdlib.h>

#include "gpu2d_internal.h"
#include "../plat.h"

static Engine sEngine[2];
static int sSingle, sSingleBlank, sA3dNow, sA3dPrev, sOutdoors;
static u8 sSkyA[SCREEN_W * SCREEN_H];
int gHostMenuPatternShown;
static u8 sBBack[SCREEN_W * SCREEN_H];
static u8 sATop[SCREEN_W * SCREEN_H];
static u8 sBKeep[SCREEN_W * SCREEN_H];
int gHostSingleScreen;
static u16 sPowCnt;
static u8 sVramCnt[9];
static u32 sScreen[2][SCREEN_W * SCREEN_H];
static u32 sFrame;
static int sDefer3d;
static int sGpu;
static Gpu2dGpuFrame sGpuFrame;
static u8 sPalettes[GPU2D_MAX_VERSIONS][0x800], sOams[GPU2D_MAX_VERSIONS][0x800];
static u32 sDeferred[2][SCREEN_W * SCREEN_H * 2], sDeferredLines[2][SCREEN_H];
static int sDiagnostics;

void gpu2d_set_diagnostics(int on) { sDiagnostics = on; }

static u32 sFeatureFrame[2][F_COUNT];
static const char *const kFeatureNames[F_COUNT] = {
    "BG mode 0", "BG mode 1", "BG mode 2", "BG mode 3", "BG mode 4", "BG mode 5", "BG mode 6", "BG mode 7",
    "3D layer", "text BG 16 colours", "text BG 256 colours", "text BG extended palette", "affine BG",
    "extended BG 16-bit tiles", "extended BG tiles, extended palette", "extended BG 256-colour bitmap",
    "extended BG direct-colour bitmap", "large bitmap BG", "BG mosaic", "OBJ", "affine OBJ", "double-size OBJ",
    "256-colour OBJ", "OBJ extended palette", "bitmap OBJ", "semi-transparent OBJ", "OBJ window", "OBJ mosaic",
    "OBJ 2D tile mapping", "window 0", "window 1", "OBJ window enabled", "alpha blending", "brightness up",
    "brightness down", "master brightness up", "master brightness down", "forced blank", "display off",
    "VRAM display", "main memory display (not shown)", "display capture (not done)", "engine B off (POWCNT1)",
    "engine A off (POWCNT1)",
};
void gpu2d_feature(const Engine *e, enum Feature f) {
    if (!sFeatureFrame[e->num][f]) sFeatureFrame[e->num][f] = sFrame + 1;
}
void gpu2d_print_features(void) {
    for (int n = 0; n < 2; n++)
        for (int f = 0; f < F_COUNT; f++)
            if (sFeatureFrame[n][f])
                fprintf(stderr, "gpu2d: engine %c: %s (first in frame %u)\n", 'A' + n, kFeatureNames[f],
                        sFeatureFrame[n][f] - 1);
}

void gpu2d_reset(void) {
    memset(sEngine, 0, sizeof sEngine);
    sEngine[1].num = 1;
    sPowCnt = 0;
    memset(sVramCnt, 0, sizeof sVramCnt);
    memset(sScreen, 0, sizeof sScreen);
    sFrame = 0;
}

static void engine_write8(Engine *e, u32 off, u8 value) {
    e->regs[off] = value;
    if ((off & ~0x17u) == 0x28) e->refWritten[off >> 4 & 1] |= (u8)(1 << (off >> 2 & 1));
}
void gpu2d_write8(uint32_t addr, uint8_t value) {
    if (addr >= 0x04000000 && addr < 0x04000000 + REG_SIZE) engine_write8(&sEngine[0], addr - 0x04000000, value);
    else if (addr >= 0x04001000 && addr < 0x04001000 + REG_SIZE) engine_write8(&sEngine[1], addr - 0x04001000, value);
    else if (addr == 0x04000304) sPowCnt = (u16)((sPowCnt & 0xff00) | value);
    else if (addr == 0x04000305) sPowCnt = (u16)((sPowCnt & 0x00ff) | value << 8);
    else if (addr >= 0x04000240 && addr <= 0x04000246) sVramCnt[addr - 0x04000240] = value;
    else if (addr == 0x04000248 || addr == 0x04000249) sVramCnt[addr - 0x04000248 + 7] = value;
}
void gpu2d_write16(uint32_t addr, uint16_t value) {
    gpu2d_write8(addr, (u8)value);
    gpu2d_write8(addr + 1, (u8)(value >> 8));
}
void gpu2d_write32(uint32_t addr, uint32_t value) {
    gpu2d_write16(addr, (u16)value);
    gpu2d_write16(addr + 2, (u16)(value >> 16));
}
uint32_t gpu2d_read32(uint32_t addr) {
    if (addr >= 0x04000000 && addr + 4 <= 0x04000000 + REG_SIZE) return reg32(&sEngine[0], addr - 0x04000000);
    if (addr >= 0x04001000 && addr + 4 <= 0x04001000 + REG_SIZE) return reg32(&sEngine[1], addr - 0x04001000);
    if (addr == 0x04000304) return sPowCnt;
    if (addr == 0x04000240) return sVramCnt[0] | sVramCnt[1] << 8 | sVramCnt[2] << 16 | (u32)sVramCnt[3] << 24;
    if (addr == 0x04000244) return sVramCnt[4] | sVramCnt[5] << 8 | sVramCnt[6] << 16;
    if (addr == 0x04000248) return sVramCnt[7] | sVramCnt[8] << 8;
    return 0;
}

static const u32 kBankSize[9] = {0x20000, 0x20000, 0x20000, 0x20000, 0x10000, 0x4000, 0x4000, 0x8000, 0x4000};
static const u8 kZeroGranule[GRANULE];
enum { AREA_A_BG, AREA_A_OBJ, AREA_B_BG, AREA_B_OBJ, AREA_A_BGEXT, AREA_A_OBJEXT, AREA_B_BGEXT, AREA_B_OBJEXT,
       AREA_NONE };
static u8 sOverlap[32 + 16 + 8 + 8 + 10][GRANULE];
static int sOverlapUsed;

static void map_piece(const u8 **slot, const u8 *src, u32 size) {
    if (!*slot) { *slot = src; return; }
    if (sOverlapUsed == (int)(sizeof sOverlap / sizeof sOverlap[0])) return;
    u8 *c = sOverlap[sOverlapUsed++];
    for (u32 i = 0; i < size; i++) c[i] = (*slot)[i] | src[i];
    *slot = c;
}

static const u32 kMirrorFG[] = {0x8000, 0}, kMirrorH[] = {0x10000, 0}, kMirrorIBg[] = {0x4000, 0x10000, 0x14000, 0},
                 kMirrorIObj[] = {0x4000, 0x8000, 0xc000, 0x10000, 0x14000, 0x18000, 0x1c000, 0}, kMirrorNone[] = {0};

static int bank_target(int i, u8 c, u32 *ofs, const u32 **mirrors) {
    if (!(c & 0x80)) return AREA_NONE;
    u32 o = c >> 3 & 3, mst = c & ((i < 2 || i > 6) ? 3 : 7);
    *ofs = 0;
    *mirrors = kMirrorNone;
    if ((i == 5 || i == 6) && (mst == 1 || mst == 2)) *mirrors = kMirrorFG;
    else if (i == 7 && mst == 1) *mirrors = kMirrorH;
    else if (i == 8 && mst == 1) *mirrors = kMirrorIBg;
    else if (i == 8 && mst == 2) *mirrors = kMirrorIObj;
    switch (i) {
    case 0: case 1:
        if (mst == 1) { *ofs = 0x20000 * o; return AREA_A_BG; }
        if (mst == 2) { *ofs = 0x20000 * (o & 1); return AREA_A_OBJ; }
        break;
    case 2: case 3:
        if (mst == 1) { *ofs = 0x20000 * o; return AREA_A_BG; }
        if (mst == 4) return i == 2 ? AREA_B_BG : AREA_B_OBJ;
        break;
    case 4:
        if (mst == 1) return AREA_A_BG;
        if (mst == 2) return AREA_A_OBJ;
        if (mst == 4) return AREA_A_BGEXT;
        break;
    case 5: case 6:
        if (mst == 1 || mst == 2) { *ofs = 0x4000 * (o & 1) + 0x10000 * (o >> 1); return mst == 1 ? AREA_A_BG : AREA_A_OBJ; }
        if (mst == 4) { *ofs = 0x4000 * (o & 1); return AREA_A_BGEXT; }
        if (mst == 5) return AREA_A_OBJEXT;
        break;
    case 7:
        if (mst == 1) return AREA_B_BG;
        if (mst == 2) return AREA_B_BGEXT;
        break;
    case 8:
        if (mst == 1) { *ofs = 0x8000; return AREA_B_BG; }
        if (mst == 2) return AREA_B_OBJ;
        if (mst == 3) return AREA_B_OBJEXT;
        break;
    }
    return AREA_NONE;
}

static void map_vram(const Gpu2dFrameInput *in) {
    Engine *a = &sEngine[0], *b = &sEngine[1];
    const u8 *gran[4][32] = {{0}};
    const u8 *ext[4][4] = {{0}};
    static const u32 kAreaSize[4] = {0x80000, 0x40000, 0x20000, 0x20000};
    sOverlapUsed = 0;
    for (int i = 0; i < 9; i++) {
        u32 ofs;
        const u32 *mirrors;
        int area = bank_target(i, sVramCnt[i], &ofs, &mirrors);
        if (area == AREA_NONE) continue;
        const u8 *bank = in->bank[i];
        if (area <= AREA_B_OBJ) {
            for (int k = -1; k < 0 || mirrors[k]; k++)
                for (u32 o = 0; o < kBankSize[i]; o += GRANULE) {
                    u32 at = ofs + (k < 0 ? 0 : mirrors[k]) + o;
                    if (at < kAreaSize[area]) map_piece(&gran[area][at >> GRANULE_SHIFT], bank + o, GRANULE);
                }
        } else if (area == AREA_A_BGEXT || area == AREA_B_BGEXT) {
            u32 first = ofs / 0x2000, n = kBankSize[i] >= 0x8000 ? 4 : 2;
            for (u32 s = 0; s < n && first + s < 4; s++)
                map_piece(&ext[area - AREA_A_BGEXT][first + s], bank + s * 0x2000, 0x2000);
        } else {
            map_piece(&ext[area - AREA_A_BGEXT][0], bank, 0x2000);
        }
    }
    for (int n = 0; n < 32; n++) {
        a->bgGran[n] = gran[0][n] ? gran[0][n] : kZeroGranule;
        if (n < 16) a->objGran[n] = gran[1][n] ? gran[1][n] : kZeroGranule;
        if (n < 16) b->bgGran[n] = gran[2][n & 7] ? gran[2][n & 7] : kZeroGranule;
        if (n < 16) b->objGran[n] = gran[3][n & 7] ? gran[3][n & 7] : kZeroGranule;
    }
    a->bgMask = 0x7ffff; a->objMask = 0x3ffff;
    b->bgMask = 0x1ffff; b->objMask = 0x1ffff;
    for (int s = 0; s < 4; s++) { a->bgExtPal[s] = ext[0][s]; b->bgExtPal[s] = ext[2][s]; }
    a->objExtPal = ext[1][0];
    b->objExtPal = ext[3][0];
}

static const u8 *lcdc_bank(const Gpu2dFrameInput *in, int i) {
    return (sVramCnt[i] & 0x87) == 0x80 ? in->bank[i] : NULL;
}

static void window_vertical(Engine *e, int line) {
    for (int w = 0; w < 2; w++) {
        u16 v = reg16(e, w ? REG_WIN1V : REG_WIN0V);
        if ((line & 0xff) == (v & 0xff)) e->winActive[w] &= ~1;
        else if ((line & 0xff) == v >> 8) e->winActive[w] |= 1;
    }
}
#if HOST_VITA
enum { kWideCanvas = 1 };   // the Vita composes a 16:9 canvas (vita_gl.c)
#else
enum { kWideCanvas = 0 };
#endif
static void window_line(Engine *e, const ObjLine *obj, u8 *mask) {
    u32 dispcnt = reg32(e, REG_DISPCNT);
    if (!(dispcnt & 0xe000)) { memset(mask, 0x3f, SCREEN_W); return; }
    u16 winin = reg16(e, REG_WININ), winout = reg16(e, REG_WINOUT);
    memset(mask, winout & 0x3f, SCREEN_W);
    if (dispcnt & 0x8000) {
        gpu2d_feature(e, F_WINOBJ);
        for (int x = 0; x < SCREEN_W; x++)
            if (obj->window[x]) mask[x] = winout >> 8 & 0x3f;
    }
    for (int w = 1; w >= 0; w--) {
        if (!(dispcnt & 0x2000u << w)) continue;
        gpu2d_feature(e, w ? F_WIN1 : F_WIN0);
        u16 h = reg16(e, w ? REG_WIN1H : REG_WIN0H);
        int x1 = h >> 8, x2 = h & 0xff;
        u8 in = (u8)(winin >> (8 * w) & 0x3f);
        for (int x = 0; x < SCREEN_W; x++) {
            // (x1 0, x2 255 - "the whole line" to the game, e.g. the open iris - leaves the last column out on the DS;
            // the tall single screen and the Vita's 16:9 canvas, where the picture continues past it, keep it in)
            if (x == x2 && !((sSingle || kWideCanvas) && x1 == 0 && x2 == 0xff)) e->winActive[w] &= ~2;
            else if (x == x1) e->winActive[w] |= 2;
            if (e->winActive[w] == 3) mask[x] = in;
        }
    }
}

static void window_advance(Engine *e) {
    u32 dispcnt = reg32(e, REG_DISPCNT);
    if (!(dispcnt & 0xe000)) return;
    for (int w = 0; w < 2; w++) {
        if (!(dispcnt & 0x2000u << w)) continue;
        u16 h = reg16(e, w ? REG_WIN1H : REG_WIN0H);
        if ((h & 0xff) >= h >> 8) e->winActive[w] &= ~2;
        else e->winActive[w] |= 2;
    }
}

static inline u32 rgb6(u16 c) {
    return (u32)(c & 0x1f) << 1 | (u32)(c >> 5 & 0x1f) << 9 | (u32)(c >> 10 & 0x1f) << 17;
}
static inline u32 chan(u32 c, int i) { return c >> (8 * i) & 0x3f; }
static inline u32 pack(u32 r, u32 g, u32 b) { return r | g << 8 | b << 16; }
static u32 blend(u32 a, u32 b, u32 eva, u32 evb) {
    u32 o[3];
    for (int i = 0; i < 3; i++) {
        u32 v = (chan(a, i) * eva + chan(b, i) * evb + 8) >> 4;
        o[i] = v > 63 ? 63 : v;
    }
    return pack(o[0], o[1], o[2]);
}
static u32 blend3d(u32 a, u32 b, u32 alpha) {
    u32 eva = alpha + 1, evb = 31 - alpha, o[3];
    for (int i = 0; i < 3; i++) o[i] = (chan(a, i) * eva + chan(b, i) * evb + 16) >> 5;
    return pack(o[0], o[1], o[2]);
}
static u32 brighten(u32 c, u32 ev, u32 bias) {
    u32 o[3];
    for (int i = 0; i < 3; i++) o[i] = chan(c, i) + (((63 - chan(c, i)) * ev + bias) >> 4);
    return pack(o[0], o[1], o[2]);
}
static u32 darken(u32 c, u32 ev, u32 bias) {
    u32 o[3];
    for (int i = 0; i < 3; i++) o[i] = chan(c, i) - ((chan(c, i) * ev + bias) >> 4);
    return pack(o[0], o[1], o[2]);
}

static void obj_mosaic_x(const Engine *e, ObjLine *obj) {
    int w = (reg16(e, REG_MOSAIC) >> 8 & 15) + 1;
    if (w == 1) return;
    int last = 0;
    for (int x = 1; x < SCREEN_W; x++) {
        if (!(obj->mosaic[last] && obj->mosaic[x]) || x % w == 0) { last = x; continue; }
        obj->value[x] = obj->value[last];
        obj->source[x] = obj->source[last];
        obj->prio[x] = obj->prio[last];
        obj->kind[x] = obj->kind[last];
        obj->alpha[x] = obj->alpha[last];
        obj->index[x] = obj->index[last];
    }
}
static void obj_mosaic_y(Engine *e, int line) {
    int h = reg16(e, REG_MOSAIC) >> 12 & 15;
    if (e->objMosaicCount == h) { e->objMosaicCount = 0; e->objMosaicLine = line + 1; }
    else e->objMosaicCount = (e->objMosaicCount + 1) & 15;
}

static u16 obj_color(const Engine *e, u32 dispcnt, const ObjLine *obj, int x) {
    switch (obj->source[x]) {
    case OBJ_SRC_DIRECT: return obj->value[x];
    case OBJ_SRC_EXTPAL:
        if (dispcnt & 1u << 31) return e->objExtPal ? pal16(e->objExtPal, obj->value[x] & 0xfffu) : 0;
        /* fall through */
    case OBJ_SRC_PALETTE: return pal16(e->palette + 0x200, obj->value[x] & 0xffu);
    }
    return 0;
}

enum { DBG_3D = 6 };
enum { EFF_NONE, EFF_ALPHA, EFF_UP, EFF_DOWN, EFF_SEMI_OBJ, EFF_BITMAP_OBJ, EFF_3D };

static inline u32 pack18(u32 c) { return (c & 0x3f) | (c >> 8 & 0x3f) << 6 | (c >> 16 & 0x3f) << 12; }
static u32 deferred_3d(const ObjLine *obj, int x, int at, u32 col0, u8 lay0, int effOk, u32 first, u32 second,
                       u32 effect, u32 eva, u32 evb, u32 evy) {
    enum { KIND_C = 1u << 30, KIND_BLEND = 2u << 30, KIND_TOP = 3u << 30 };
    int below3d = second & 1;
    if (at == 1) {
        if (lay0 == LAYER_OBJ && obj->kind[x] != OBJ_NORMAL) {
            if (!below3d) return KIND_C | pack18(col0);
            u32 a = obj->kind[x] == OBJ_BITMAP ? obj->alpha[x] + 1u : eva, b = obj->kind[x] == OBJ_BITMAP ? 15u - obj->alpha[x] : evb;
            return KIND_BLEND | pack18(col0) | a << 18 | b << 23;
        }
        if (effOk && ((first >> lay0) & 1)) {
            if (effect == 1 && below3d) return KIND_BLEND | pack18(col0) | eva << 18 | evb << 23;
            if (effect == 2) return KIND_C | pack18(brighten(col0, evy, 8));
            if (effect == 3) return KIND_C | pack18(darken(col0, evy, 7));
        }
        return KIND_C | pack18(col0);
    }
    if ((second >> lay0) & 1) return KIND_TOP | pack18(col0) | 1u << 28;
    if (effOk && (first & 1) && (effect == 2 || effect == 3)) return KIND_TOP | evy << 18 | effect << 28;
    return KIND_TOP;
}

int gpu2d_tall_3d_ok(void) {
    u32 d = reg32(&sEngine[0], REG_DISPCNT);
    return (d & 8) && (d & 0x100) && !(d & 0x80);
}
int gpu2d_main_on_bottom(void) { return 1; }
// Engine A shows a VRAM bank directly (display mode 2): the LCDC banks must be in the render snapshot.
int gpu2d_main_vram_display(void) { return (reg32(&sEngine[0], REG_DISPCNT) >> 16 & 3) == 2; }

// Engine-0 compose for the main-screen-only path (the Vita): no dual-screen bookkeeping, no deferred 3D, no
// diagnostics. Same result as the general loop in compose_line, organised layer-major: each layer is walked once
// over the line from back to front, and a hit pushes the current top down to second, so after all layers the top
// two are the first two opaque layers in front-to-back order - what the general loop finds per pixel. Then one
// pass blends. This replaces a per-pixel walk over the layer list with a type dispatch on every entry.

static void compose_line_fast(Engine *e, u32 dispcnt, const ObjLine *obj, const u8 *mask, const u8 *order, int layers,
                              int has3d, s32 hofs3d, const u32 *line3d, u32 backdrop, u32 effect, u32 first,
                              u32 second, u32 eva, u32 evb, u32 evy, u32 *dst, u32 *effectsUsed) {
    enum { OBJ_ORDER = 4 + 4, L3D = 0x80 };   // L3D: flag on a layer id = the 3D layer (BG0)
    // top / second candidates. The second slot starts as what the general loop uses when fewer than two layers
    // are found: none found -> backdrop colour with LAYER_NONE; one found -> the backdrop (pushed down from top).
    u32 objPrio = 0;   // sprite priorities that have a visible pixel on this line
    if (dispcnt & 0x1000)   // (only where a sprite wrote: obj->source is zero outside [lo, hi))
        for (int x = obj->lo < 0 ? 0 : obj->lo, xe = obj->hi > SCREEN_W ? SCREEN_W : obj->hi; x < xe; x++)
            if (obj->source[x] && (mask[x] & 0x10)) objPrio |= 1u << (obj->prio[x] & 3);
    // 3D-only line: no window, and every layer but the 3D one is empty here (transparent BG line, no sprite
    // pixels). Then each pixel is the 3D colour over the backdrop - the same outcome as the general loops below.
    if (has3d && !(dispcnt & 0xe000)) {
        int only3d = 1;
        for (int k = 0; k < layers && only3d; k++) {
            int n = order[k];
            if (n >= OBJ_ORDER) only3d = !(objPrio >> (n - OBJ_ORDER) & 1);
            else if (n != 0) only3d = !e->bgAny[n];
        }
        if (only3d) {
            int lo = hofs3d < 0 ? -hofs3d : 0, hi = SCREEN_W - (hofs3d > 0 ? hofs3d : 0);
            const u32 *l3 = line3d + hofs3d;
            int over = (second >> LAYER_BACKDROP) & 1;      // 3D over the backdrop blends (EFF_3D)
            int eff3d = (first & 1);                         // BG0 is a first target (brightness effects)
            int effBd = (first >> LAYER_BACKDROP) & 1;       // backdrop is a first target
            u32 bdOut = backdrop, bdEff = EFF_NONE;
            if (effBd && effect == 2) bdOut = brighten(backdrop, evy, 8), bdEff = EFF_UP;
            else if (effBd && effect == 3) bdOut = darken(backdrop, evy, 7), bdEff = EFF_DOWN;
            u32 used = 0;
            if (!(!over && eff3d && (effect == 2 || effect == 3))) {
                // No brightness effect on the 3D pixels: four at a time - alpha 0 -> the backdrop result, else the
                // 3D colour; the rare partly transparent 3D pixels over the backdrop are blended one by one.
                typedef u32 v4u __attribute__((vector_size(16)));
                u32 l3b[SCREEN_W] __attribute__((aligned(16)));
                const u32 *src = l3;
                if (lo != 0 || hi != SCREEN_W) {
                    memset(l3b, 0, sizeof l3b);
                    if (hi > lo) memcpy(l3b + lo, l3 + lo, (size_t)(hi - lo) * sizeof(u32));
                    src = l3b;
                }
                v4u anyZero = {0, 0, 0, 0}, anyPart = {0, 0, 0, 0}, anySet = {0, 0, 0, 0};
                const v4u bd = {bdOut, bdOut, bdOut, bdOut};
                for (int x = 0; x < SCREEN_W; x += 4) {
                    v4u c;
                    memcpy(&c, src + x, 16);
                    v4u a = c >> 24 & 0x1f;
                    v4u z = (v4u)(a == 0);
                    v4u out = (z & bd) | (~z & (c & 0x3f3f3f));
                    memcpy(dst + x, &out, 16);
                    anyZero |= z;
                    anySet |= ~z;
                    anyPart |= (v4u)(a != 0) & (v4u)(a != 31);
                }
                if ((anyZero[0] | anyZero[1] | anyZero[2] | anyZero[3])) used |= 1u << bdEff;
                if ((anySet[0] | anySet[1] | anySet[2] | anySet[3])) used |= 1u << (over ? EFF_3D : EFF_NONE);
                if (over && (anyPart[0] | anyPart[1] | anyPart[2] | anyPart[3]))
                    for (int x = 0; x < SCREEN_W; x++) {
                        u32 a = src[x] >> 24 & 0x1f;
                        if (a && a != 31) dst[x] = blend3d(src[x] & 0x3f3f3f, backdrop, a);
                    }
                *effectsUsed = used;
                return;
            }
            for (int x = 0; x < SCREEN_W; x++) {
                u32 c = x >= lo && x < hi ? l3[x] : 0, a = c >> 24 & 0x1f;
                if (!a) { dst[x] = bdOut; used |= 1u << bdEff; continue; }
                u32 col0 = c & 0x3f3f3f, out = col0, eff = EFF_NONE;
                if (over) {
                    if (a != 31) out = blend3d(col0, backdrop, a);
                    eff = EFF_3D;
                } else if (eff3d) {
                    if (effect == 2) { out = brighten(col0, evy, 8); eff = EFF_UP; }
                    else if (effect == 3) { out = darken(col0, evy, 7); eff = EFF_DOWN; }
                }
                used |= 1u << eff;
                dst[x] = out;
            }
            *effectsUsed = used;
            return;
        }
    }
    // General case, layer-major and four pixels at a time (GCC vector types: NEON on the Vita, SSE on a PC).
    // Each pixel keeps its top and second candidates packed in one word: colour (6 bits per channel in bytes
    // 0-2) | 3D alpha << 24 (0 = not a 3D pixel) | layer << 29.
    typedef u32 v4u __attribute__((vector_size(16)));
    typedef u16 v4h __attribute__((vector_size(8)));
    typedef u8 v4b __attribute__((vector_size(4)));
#define LD8(p) ({ v4b b_; memcpy(&b_, (p), 4); __builtin_convertvector(b_, v4u); })
#define LD16(p) ({ v4h h_; memcpy(&h_, (p), 8); __builtin_convertvector(h_, v4u); })
#define LD32(p) ({ v4u w_; memcpy(&w_, (p), 16); w_; })
#define ST32(p, v) do { v4u w_ = (v); memcpy((p), &w_, 16); } while (0)
    u32 tc[SCREEN_W] __attribute__((aligned(16))), sc[SCREEN_W] __attribute__((aligned(16)));
    {
        u32 t0 = backdrop | (u32)LAYER_BACKDROP << 29, s0 = backdrop | (u32)LAYER_NONE << 29;
        for (int x = 0; x < SCREEN_W; x++) { tc[x] = t0; sc[x] = s0; }
    }
    u32 objc[SCREEN_W] __attribute__((aligned(16)));
    u8 objp[SCREEN_W] __attribute__((aligned(16)));
    if (objPrio) {
        for (int x = 0; x < SCREEN_W; x++) {
            if ((mask[x] & 0x10) && obj->source[x]) {
                objp[x] = obj->prio[x];
                objc[x] = rgb6(obj_color(e, dispcnt, obj, x)) | (u32)LAYER_OBJ << 29;
            } else { objp[x] = 0xff; objc[x] = 0; }
        }
    }
    for (int k = layers - 1; k >= 0; k--) {
        int n = order[k];
        if (n >= OBJ_ORDER) {
            u32 p = (u32)(n - OBJ_ORDER);
            if (!(objPrio >> p & 1)) continue;
            for (int x = 0; x < SCREEN_W; x += 4) {
                v4u c = (v4u)(LD8(objp + x) == p);
                v4u t = LD32(tc + x), s2 = LD32(sc + x);
                ST32(sc + x, (c & t) | (~c & s2));
                ST32(tc + x, (c & LD32(objc + x)) | (~c & t));
            }
        } else if (n == 0 && has3d) {
            u32 l3b[SCREEN_W] __attribute__((aligned(16)));
            int lo = hofs3d < 0 ? -hofs3d : 0, hi = SCREEN_W - (hofs3d > 0 ? hofs3d : 0);
            memset(l3b, 0, sizeof l3b);
            if (hi > lo) memcpy(l3b + lo, line3d + hofs3d + lo, (size_t)(hi - lo) * sizeof(u32));
            for (int x = 0; x < SCREEN_W; x += 4) {
                v4u c3 = LD32(l3b + x), a = c3 >> 24 & 0x1f;
                v4u c = (v4u)(a != 0) & (v4u)((LD8(mask + x) & 1) != 0);
                v4u nc = (c3 & 0x3f3f3f) | a << 24;
                v4u t = LD32(tc + x), s2 = LD32(sc + x);
                ST32(sc + x, (c & t) | (~c & s2));
                ST32(tc + x, (c & nc) | (~c & t));
            }
        } else {
            if (!e->bgAny[n]) continue;   // fully transparent on this line
            const u16 *bg = e->bg[n];
            u32 bit = 1u << n, lay = (u32)n << 29;
            for (int x = 0; x < SCREEN_W; x += 4) {
                v4u b = LD16(bg + x);
                v4u c = (v4u)((b & PIX_OPAQUE) != 0) & (v4u)((LD8(mask + x) & bit) != 0);
                v4u nc = (b & 0x1f) << 1 | (b >> 5 & 0x1f) << 9 | (b >> 10 & 0x1f) << 17 | lay;
                v4u t = LD32(tc + x), s2 = LD32(sc + x);
                ST32(sc + x, (c & t) | (~c & s2));
                ST32(tc + x, (c & nc) | (~c & t));
            }
        }
    }
#undef LD8
#undef LD16
#undef LD32
#undef ST32
    u32 used = 0;
    for (int x = 0; x < SCREEN_W; x++) {
        u32 t = tc[x], s2 = sc[x];
        u32 col0 = t & 0x3f3f3f, out = col0, eff = EFF_NONE, lay0 = t >> 29, a3 = t >> 24 & 0x1f;
        u32 scol = s2 & 0x3f3f3f;
        int below2nd = (second >> (s2 >> 29)) & 1;
        if (lay0 == LAYER_OBJ && obj->kind[x] != OBJ_NORMAL && below2nd) {
            if (obj->kind[x] == OBJ_BITMAP) {
                out = blend(col0, scol, obj->alpha[x] + 1u, 15u - obj->alpha[x]);
                eff = EFF_BITMAP_OBJ;
            } else {
                out = blend(col0, scol, eva, evb);
                eff = EFF_SEMI_OBJ;
            }
        } else if (a3 && below2nd) {
            if (a3 != 31) out = blend3d(col0, scol, a3);   // alpha 31: blend3d returns col0 exactly
            eff = EFF_3D;
        } else if ((mask[x] & 0x20) && ((first >> lay0) & 1)) {
            if (effect == 1 && below2nd) { out = blend(col0, scol, eva, evb); eff = EFF_ALPHA; }
            else if (effect == 2) { out = brighten(col0, evy, 8); eff = EFF_UP; }
            else if (effect == 3) { out = darken(col0, evy, 7); eff = EFF_DOWN; }
        }
        used |= 1u << eff;
        dst[x] = out;
    }
    *effectsUsed = used;
}

static void compose_line(Engine *e, int line, const u32 *line3d, u32 *dst) {
    u32 dispcnt = reg32(e, REG_DISPCNT);
    const ObjLine *obj = &e->obj;
    u8 mask[SCREEN_W];
    window_line(e, obj, mask);
    obj_mosaic_x(e, &e->obj);
    int top3d = e->num == 1 && sSingle && line3d && sA3dPrev && !(sPowCnt & 0x8000);
    int has3d = e->num == 0 && (dispcnt & 8) && (dispcnt & 0x100) && line3d;
    int defer3d = e->num == 0 && (dispcnt & 8) && (dispcnt & 0x100) && sDefer3d;
    if (has3d || defer3d) gpu2d_feature(e, F_3D);
    u32 *defer = e->defer + line * SCREEN_W;
    s32 hofs3d = reg16(e, REG_BGHOFS) & 0x1ff;
    if (hofs3d & 0x100) hofs3d -= 0x200;
    enum { OBJ_ORDER = 4 + 4, TOP3D = 7 };
    u8 order[4 * 5 + 1];
    int layers = 0;
    for (u32 p = 0; p < 4; p++) {
        if (dispcnt & 0x1000) order[layers++] = (u8)(OBJ_ORDER + p);
        for (int n = 0; n < 4; n++)
            if ((reg16(e, REG_BGCNT + 2 * n) & 3) == p && (dispcnt & 0x100u << n)) order[layers++] = (u8)n;
    }
    if (top3d) {
        u32 back = 0;
        for (int k = 0; k < layers; k++)
            if (order[k] < 4 && (reg16(e, REG_BGCNT + 2 * order[k]) & 3) > back) back = reg16(e, REG_BGCNT + 2 * order[k]) & 3;
        {
            extern unsigned Scene_GetCurrent(void);
            if (Scene_GetCurrent() != 0x2c) back = 0;
        }
        u8 o2[4 * 5 + 1];
        int m2 = 0;
        for (int k = 0; k < layers; k++)
            if (order[k] < 4 && (reg16(e, REG_BGCNT + 2 * order[k]) & 3) < back) o2[m2++] = order[k];
        o2[m2++] = TOP3D;
        for (int k = 0; k < layers; k++)
            if (order[k] >= OBJ_ORDER) o2[m2++] = order[k];
        for (int k = 0; k < layers; k++)
            if (order[k] < 4 && (reg16(e, REG_BGCNT + 2 * order[k]) & 3) >= back) o2[m2++] = order[k];
        memcpy(order, o2, (size_t)m2);
        layers = m2;
    }
    u16 bldcnt = reg16(e, REG_BLDCNT), bldalpha = reg16(e, REG_BLDALPHA);
    u32 effect = bldcnt >> 6 & 3, first = bldcnt & 0x3f, second = bldcnt >> 8 & 0x3f;
    u32 eva = bldalpha & 0x1f, evb = bldalpha >> 8 & 0x1f, evy = reg16(e, REG_BLDY) & 0x1f;
    if (eva > 16) eva = 16;
    if (evb > 16) evb = 16;
    if (evy > 16) evy = 16;
    u32 backdrop = rgb6(pal16(e->palette, 0));
    u32 effectsUsed = 0;
    // Engine B (the menus over the scene) takes the fast path too unless the sprite window is on: the per-pixel
    // loop below was most of a menu redraw's cost on the Vita. What it records besides the colour - which pixels
    // show only the backdrop (sBBack) - comes from the layers' opaque pixels; sBKeep needs the sprite window, so 0.
    if ((e->num == 0 || (line < SCREEN_H && !(dispcnt & 0x8000))) && !sSingle && !defer3d && !top3d && !sDiagnostics) {
        compose_line_fast(e, dispcnt, obj, mask, order, layers, has3d, hofs3d, line3d, backdrop, effect, first, second,
                          eva, evb, evy, dst, &effectsUsed);
        if (e->num == 1) {
            u8 *bb = sBBack + line * SCREEN_W;
            memset(sBKeep + line * SCREEN_W, 0, SCREEN_W);
            for (int x = 0; x < SCREEN_W; x++) bb[x] = 1;
            for (int n = 0; n < 4; n++)
                if (dispcnt & 0x100u << n)
                    for (int x = 0; x < SCREEN_W; x++)
                        if ((mask[x] & 1 << n) && (e->bg[n][x] & PIX_OPAQUE)) bb[x] = 0;
            if (dispcnt & 0x1000)
                for (int x = obj->lo < 0 ? 0 : obj->lo, xe = obj->hi > SCREEN_W ? SCREEN_W : obj->hi; x < xe; x++)
                    if ((mask[x] & 0x10) && obj->source[x]) bb[x] = 0;
        }
        goto features;
    }
    for (int x = 0; x < SCREEN_W; x++) {
        u32 col[2] = {backdrop, backdrop};
        u8 lay[2] = {LAYER_BACKDROP, LAYER_BACKDROP};
        int is3d[2] = {0, 0}, found = 0, at3d = -1;
        u32 alpha3d = 0;
        u8 m = mask[x];
        for (int k = 0; k < layers && found < 2; k++) {
            int n = order[k];
            if (n >= OBJ_ORDER) {
                if ((m & 0x10) && obj->source[x] && obj->prio[x] == n - OBJ_ORDER) {
                    col[found] = rgb6(obj_color(e, dispcnt, obj, x));
                    lay[found++] = LAYER_OBJ;
                }
            } else if (n == TOP3D) {
                if (!(m & 1)) continue;
                u32 c = line3d[x];
                if (!(c >> 24 & 0x1f)) continue;
                if (!found) alpha3d = c >> 24 & 0x1f;
                is3d[found] = 1;
                col[found] = c & 0x3f3f3f;
                lay[found++] = LAYER_BG0;
            } else if (!(m & 1 << n)) {
                continue;
            } else if (n == 0 && defer3d) {
                s32 sx = x + hofs3d;
                if (sx >= 0 && sx < SCREEN_W && at3d < 0) at3d = found;
            } else if (n == 0 && has3d) {
                s32 sx = x + hofs3d;
                if (sx < 0 || sx >= SCREEN_W) continue;
                u32 c = line3d[sx];
                if (!(c >> 24 & 0x1f)) continue;
                if (!found) alpha3d = c >> 24 & 0x1f;
                is3d[found] = 1;
                col[found] = c & 0x3f3f3f;
                lay[found++] = LAYER_BG0;
            } else if (e->bg[n][x] & PIX_OPAQUE) {
                col[found] = rgb6(e->bg[n][x]);
                lay[found++] = (u8)n;
            }
        }
        if (sSingle && e->num == 0 && is3d[0]) sA3dNow = 1;
        if (sSingle && e->num == 0) sSkyA[line * SCREEN_W + x] = (u8)(sOutdoors && is3d[0] && (line3d[x + hofs3d] >> 31));
        if (e->num == 1 && line < SCREEN_H) sBBack[line * SCREEN_W + x] = (u8)(!found);
        if (e->num == 0 && line < SCREEN_H) sATop[line * SCREEN_W + x] = (u8)(is3d[0] ? 6 : lay[0]);
        if (e->num == 1 && line < SCREEN_H)
            sBKeep[line * SCREEN_W + x] = (u8)((dispcnt & 0x8000) && lay[0] == LAYER_OBJ && !(m & 0x20) && effect == 3);
        if (!found) lay[1] = LAYER_NONE;
        u32 out = col[0], eff = EFF_NONE;
        int effOk = m & 0x20, below2nd = (second >> lay[1]) & 1;
        if (lay[0] == LAYER_OBJ && obj->kind[x] != OBJ_NORMAL && below2nd) {
            if (obj->kind[x] == OBJ_BITMAP) {
                out = blend(col[0], col[1], obj->alpha[x] + 1u, 15u - obj->alpha[x]);
                eff = EFF_BITMAP_OBJ;
            } else {
                out = blend(col[0], col[1], eva, evb);
                eff = EFF_SEMI_OBJ;
            }
        } else if (is3d[0] && below2nd) {
            out = blend3d(col[0], col[1], alpha3d);
            eff = EFF_3D;
        } else if (effOk && ((first >> lay[0]) & 1)) {
            if (effect == 1 && below2nd) { out = blend(col[0], col[1], eva, evb); eff = EFF_ALPHA; }
            else if (effect == 2) { out = brighten(col[0], evy, 8); eff = EFF_UP; }
            else if (effect == 3) { out = darken(col[0], evy, 7); eff = EFF_DOWN; }
        }
        effectsUsed |= 1u << eff;
        dst[x] = out;
        if (defer3d) defer[x] = at3d < 0 ? 0 : deferred_3d(obj, x, at3d, col[0], lay[0], effOk, first, second, effect,
                                                         eva, evb, evy);
        if (sDiagnostics) {
            int at = line * SCREEN_W + x;
            e->dbgTop[at] = is3d[0] ? DBG_3D : lay[0];
            e->dbgBelow[at] = is3d[1] ? DBG_3D : lay[1];
            e->dbgEffect[at] = (u8)eff;
            e->dbgObj[at] = obj->index[x];
            e->dbgAlpha[at] = (u8)(is3d[0] ? alpha3d : obj->alpha[x]);
            e->dbgTopColor[at] = col[0];
            e->dbgBelowColor[at] = col[1];
        }
    }
features:
    if (effectsUsed & (1u << EFF_ALPHA | 1u << EFF_SEMI_OBJ | 1u << EFF_BITMAP_OBJ | 1u << EFF_3D)) gpu2d_feature(e, F_ALPHA);
    if (effectsUsed & 1u << EFF_UP) gpu2d_feature(e, F_BRIGHT_UP);
    if (effectsUsed & 1u << EFF_DOWN) gpu2d_feature(e, F_BRIGHT_DOWN);
}

static void render_line(Engine *e, const Gpu2dFrameInput *in, int line) {
    u32 dispcnt = reg32(e, REG_DISPCNT), *dst = e->out + line * SCREEN_W;
    if (sGpu == 1) {
        e->lineInfo[line] = reg16(e, REG_MASTER_BRIGHT) | (u32)(reg16(e, REG_BGHOFS) & 0x1ff) << 16;
        if (e->num == 1 && !(sPowCnt & 0x200)) return;
        if (!(dispcnt & 0x80)) window_advance(e);
        obj_mosaic_y(e, line);
        return;
    }
    if (sDiagnostics) {
        memset(e->dbgTop + line * SCREEN_W, LAYER_NONE, SCREEN_W);
        memset(e->dbgEffect + line * SCREEN_W, EFF_NONE, SCREEN_W);
    }
    u32 *defer = e->defer + line * SCREEN_W;
    if (sDefer3d) {
        memset(defer, 0, SCREEN_W * sizeof *defer);
        e->lineInfo[line] = 0;
    }
    if (e->num == 1 && !(sPowCnt & 0x200)) {
        gpu2d_feature(e, F_ENGINE_B_OFF);
        if (sSingle) {
            int r3 = line + in->topRows - SCREEN_H;
            const u32 *l3 = in->lines3dTop && r3 >= 0 ? in->lines3dTop + r3 * SCREEN_W : NULL;
            for (int x = 0; x < SCREEN_W; x++) {
                u32 c = l3 ? l3[x] : 0, al = c >> 24 & 31;
                dst[x] = !al ? 0 : al == 31 ? c & 0x3f3f3f : blend3d(c & 0x3f3f3f, 0, al);
            }
            return;
        }
        for (int x = 0; x < SCREEN_W; x++) dst[x] = 0x3f3f3f;
        return;
    }
    if (e->num == 0 && !(sPowCnt & 2)) gpu2d_feature(e, F_ENGINE_A_OFF);
    if (e->num == 0 && (reg32(e, REG_DISPCAPCNT) & 1u << 31)) gpu2d_feature(e, F_CAPTURE);
    gpu2d_feature(e, (enum Feature)(F_MODE0 + (dispcnt & 7)));
    u32 composed[SCREEN_W];
    if (dispcnt & 0x80) {
        gpu2d_feature(e, F_FORCED_BLANK);
        for (int x = 0; x < SCREEN_W; x++) composed[x] = 0x3f3f3f;
    } else {
        for (int n = 0; n < 4; n++) gpu2d_render_bg_line(e, n, line);
        const u32 *l3 = e->num == 0 && in->lines3d ? in->lines3d + line * SCREEN_W
                      : e->num == 1 && sSingle && in->lines3dTop && line + in->topRows - SCREEN_H >= 0
                          ? in->lines3dTop + (line + in->topRows - SCREEN_H) * SCREEN_W : NULL;
        compose_line(e, line, l3, composed);
    }
    obj_mosaic_y(e, line);
    u32 mode = dispcnt >> 16 & (e->num ? 1 : 3);
    if (sDefer3d && mode != 1) memset(defer, 0, SCREEN_W * sizeof *defer);
    if (mode == 1) {
        memcpy(dst, composed, sizeof composed);
    } else if (mode == 0) {
        gpu2d_feature(e, F_DISPLAY_OFF);
        for (int x = 0; x < SCREEN_W; x++) dst[x] = 0x3f3f3f;
        return;
    } else if (mode == 2) {
        gpu2d_feature(e, F_DISPLAY_VRAM);
        const u8 *bank = lcdc_bank(in, dispcnt >> 18 & 3);
        for (int x = 0; x < SCREEN_W; x++) dst[x] = bank ? rgb6(pal16(bank, (u32)(line * SCREEN_W + x))) : 0;
    } else {
        gpu2d_feature(e, F_DISPLAY_FIFO);
        memset(dst, 0, SCREEN_W * sizeof *dst);
    }
    u16 mb = reg16(e, REG_MASTER_BRIGHT);
    if (sSingle && e->num == 1) {
        const Engine *a = &sEngine[0];
        u32 da = reg32(a, REG_DISPCNT);
        mb = reg16(a, REG_MASTER_BRIGHT);
        if ((da & 0x80) || !(da >> 16 & 3))
            for (int x = 0; x < SCREEN_W; x++) dst[x] = 0x3f3f3f;
    }
    u32 f = mb & 0x1f;
    if (f > 16) f = 16;
    if (sDefer3d) {
        s32 hofs3d = reg16(e, REG_BGHOFS) & 0x1ff;
        e->lineInfo[line] = mb | (u32)hofs3d << 16;
        for (int x = 0; x < SCREEN_W; x++)
            if (defer[x]) composed[x] = dst[x];
    }
    if ((mb >> 14) == 1 && f) {
        gpu2d_feature(e, F_MASTER_UP);
        for (int x = 0; x < SCREEN_W; x++) dst[x] = brighten(dst[x], f, 0);
    } else if ((mb >> 14) == 2 && f) {
        gpu2d_feature(e, F_MASTER_DOWN);
        for (int x = 0; x < SCREEN_W; x++) dst[x] = darken(dst[x], f, 15);
    }
    if (sDefer3d)
        for (int x = 0; x < SCREEN_W; x++)
            if (defer[x]) dst[x] = composed[x];
}

static s32 ref28(const Engine *e, u32 off) { return (s32)(reg32(e, off) << 4) >> 4; }

static u32 gpu_offset(const Gpu2dFrameInput *in, const u8 *p) {
    static const u32 kOffset[9] = GPU2D_BANK_OFFSET;
    if (!p) return GPU2D_VRAM_ZERO;
    for (int i = 0; i < 9; i++)
        if (p >= in->bank[i] && p < in->bank[i] + kBankSize[i]) return kOffset[i] + (u32)(p - in->bank[i]);
    if (p >= sOverlap[0] && p < sOverlap[0] + sizeof sOverlap) return GPU2D_VRAM_OVERLAP + (u32)(p - sOverlap[0]);
    return GPU2D_VRAM_ZERO;
}
static void gpu_tables(const Gpu2dFrameInput *in) {
    Gpu2dGpuFrame *f = &sGpuFrame;
    for (int n = 0; n < 2; n++) {
        const Engine *e = &sEngine[n];
        u32 *t = f->tables[n];
        memset(t, 0, 64 * sizeof *t);
        for (int k = 0; k < 32; k++) t[G2T_BG + k] = gpu_offset(in, n == 0 || k < 16 ? e->bgGran[k] : NULL);
        for (int k = 0; k < 16; k++) t[G2T_OBJ + k] = gpu_offset(in, e->objGran[k]);
        for (int k = 0; k < 4; k++) t[G2T_BGEXT + k] = gpu_offset(in, e->bgExtPal[k]);
        t[G2T_OBJEXT] = gpu_offset(in, e->objExtPal);
        for (int k = 0; k < 4; k++) t[G2T_LCDC + k] = lcdc_bank(in, k) ? gpu_offset(in, in->bank[k]) : ~0u;
    }
    f->overlap = sOverlap[0];
    f->overlapSize = (u32)sOverlapUsed * GRANULE;
}
static u32 pack_obj(const ObjLine *o, int x) {
    return o->value[x] | (u32)o->source[x] << 16 | (u32)o->prio[x] << 18 | (u32)o->kind[x] << 20 |
           (u32)o->alpha[x] << 22 | (u32)(o->mosaic[x] & 1) << 26 | (u32)(o->window[x] & 1) << 27;
}
static void gpu_line(int n, int line) {
    const Engine *e = &sEngine[n];
    u32 *l = sGpuFrame.lines[n][line];
    memset(l, 0, GPU2D_LINE_WORDS * sizeof *l);
    for (int k = 0; k < REG_SIZE / 4; k++) l[G2L_REGS + k] = reg32(e, (u32)k * 4);
    for (int i = 0; i < 2; i++) l[G2L_REF + 2 * i] = (u32)e->refX[i], l[G2L_REF + 2 * i + 1] = (u32)e->refY[i];
    l[G2L_WIN] = (u32)(e->winActive[0] & 3) | (u32)(e->winActive[1] & 3) << 2;
    l[G2L_OBJMOSAIC] = (u32)e->objMosaicLine;
    l[G2L_PALETTE] = sGpuFrame.paletteCount - 1;
    l[G2L_OAM] = sGpuFrame.oamCount - 1;
    l[G2L_POWCNT] = sPowCnt;
}
static void gpu_versions(const Gpu2dFrameInput *in) {
    Gpu2dGpuFrame *f = &sGpuFrame;
    if (memcmp(sPalettes[f->paletteCount - 1], in->palette, 0x800) && f->paletteCount < GPU2D_MAX_VERSIONS)
        memcpy(sPalettes[f->paletteCount++], in->palette, 0x800);
    if (memcmp(sOams[f->oamCount - 1], in->oam, 0x800) && f->oamCount < GPU2D_MAX_VERSIONS)
        memcpy(sOams[f->oamCount++], in->oam, 0x800);
}
const Gpu2dGpuFrame *gpu2d_gpu_frame(void) { return &sGpuFrame; }

enum { CANVAS_MAX = 640 };
static u32 sCanvas[SCREEN_W * CANVAS_MAX];
static int sCanvasRows;
int gHostIrisRadius = -1;
static int sIrisSnap = -1;
// The Vita's title screen: the top screen's logo - the sub engine's BG2, just that layer, rendered on the game
// thread in the prepass - drawn 1:1 at the same place on the main screen by the GPU compose. RGBA8 (R,G,B,A
// bytes), transparent where BG2 is. Not while BG2 is a full-screen picture (the black wipe before the logo).
static int sLogoOn, sLogoReady, sLogoFull, sLogoDx, sLogoDy, sPromptSnap;
// The boot screens (scene 0x3f): the Vita shows the top screen (the Nintendo logo) instead of the bottom one (the
// online-play notice) - the sub engine's whole picture, rendered on the game thread, in sLogo.
static int sMirrorOn, sMirrorReady;
static uint32_t sLogo[SCREEN_W * SCREEN_H];
static int sOverlayOn, sOvlReady;   // (Vita) a menu over the scene: engine B drawn by the bands into sLogo
// (Vita) The sky, one screen, as City Folk has it: outdoors with no menu the DS shows the sky on the top screen
// (engine B: its gradient, clouds, stars, sun, moon, fireworks, the presents' balloons) and the main screen only a
// flat backdrop colour above the town's horizon. The Vita shows the top screen's whole sky behind the main screen's
// picture instead, wherever the main screen's backdrop colour would be. It is drawn by the present at the screen's
// own resolution (vita_gl.c), each sky pixel exactly 2 x 2 screen pixels (pixel-perfect: no stretching), 480 sky
// columns across the 960-pixel screen: the DS's 256 in the middle and, either side, SKY_SIDE more drawn as the DS
// draws columns beside its screen (sky_side_lines: the same line's registers, everything moved 256 columns). At 2x
// its 192 lines are 384 screen rows: the camera looking up (City Folk's look-up, Camera::onExecute) puts the horizon
// at screen row 377, so the sky's bottom sits SKY_DOWN_PX below the horizon (behind the ground) and looking up shows
// all of it, line 0 at the screen's top (sky_horizon_row: the horizon from the camera; sSkyTop: line 0's screen row).
// The bands draw the changed lines of the part shown into sSkyTex (without engine B's own master brightness).
// sSkyTex: row 1 + l = sky line l; rows 0 and SKY_TH - 1 the plain colours of lines 0 and 191 (the texture's edge
// rows: above the sky and below it).
enum { SKY_SIDE = GPU2D_SKY_SIDE, SKY_TW = GPU2D_SKY_W, SKY_TH = GPU2D_SKY_H };
enum { SKY_SCREEN_H = 544, SKY_DOWN_PX = 2 * SCREEN_H - 377 };   // (377: the looked-up horizon, row 133.1 of 192)
static int sSkyOn, sSkyReady, sSkyEdges, sSkyHold, sSkyWas;
static int sSkyBack;   // the sky was held (sky_hold): every line again when it is drawn again
static int sSkyTop = 110 + SKY_DOWN_PX - 2 * SCREEN_H, sSkyTopSnap = 110 + SKY_DOWN_PX - 2 * SCREEN_H;
static u32 sSkyTex[SKY_TW * SKY_TH];
static u8 sSkyRows[SKY_TH];    // rows written since the present last took them
static u8 sSkyValid[SCREEN_H]; // the sky line in sSkyTex is this frame's (else drawn when it is shown)
static unsigned sLogoGen;   // counts the frames that wrote sLogo (the present uploads it only then)
unsigned gpu2d_gc_logo_gen(void) { return sLogoGen; }
// The rows of sLogo written since the present last took them (it uploads only those): set by whoever writes a row
// (the bands, the title prepass), taken and cleared by the present.
static u8 sLogoRows[SCREEN_H];
void gpu2d_gc_logo_take_rows(unsigned char *rows) {
    for (int y = 0; y < SCREEN_H; y++) rows[y] = __atomic_exchange_n(&sLogoRows[y], 0, __ATOMIC_ACQ_REL);
}
int gHostTitlePrompt;   // TitleBlinkText (Vita): the "touch to start" BG3 is shown
// The logo (NULL when none) and how far to move it to centre it on the screen (dx, dy); prompt: BG3 is the title's
// "touch to start", to be drawn in the screen's bottom-right corner.
// (Vita) The HUD clock (HudClock::draw sets gHostClockDrawn each frame it draws, gHostClockX/Y where its panel
// cell goes): the GPU compose moves the sprites of the clock's box - from the panel's outline to the screen's right
// and bottom edges - right by the side bar's width, so the clock sits at the 16:9 screen's right edge.
#define CLOCK_BOX_DX 36   // the box measured on the field HUD: panel point (128, 96) -> box x 164..255, y 148..191
#define CLOCK_BOX_DY 52
int gHostClockDrawn, gHostClockX, gHostClockY;
static int sClockSnap, sClockXSnap, sClockYSnap;
int gpu2d_gc_objmove(int rect[4]) {
    rect[0] = sClockXSnap + CLOCK_BOX_DX, rect[1] = sClockYSnap + CLOCK_BOX_DY, rect[2] = 256, rect[3] = 192;
    return sClockSnap;
}
int gpu2d_mirror_shown(void) { return sMirrorOn; }   // (Vita) the picture shown is engine B's (render.c)
int gpu2d_sub_shown(void) { return (sPowCnt & 0x8000) != 0; }   // (Vita) engine B is the screen shown
int gpu2d_gc_logo(const uint32_t **px, int *dx, int *dy, int *prompt, int *mirror) {
    // mirror: 1 engine B's picture instead (boot screens), 2 the menu over the scene (alpha: 0 none, 128 over the
    // 3D or the backdrop, 255 always - single_compose's rule)
    *mirror = sMirrorReady ? 1 : sOvlReady ? 2 : 0;
    if (*mirror) { *px = sLogo; *dx = *dy = *prompt = 0; return 1; }
    *prompt = sLogoOn && sPromptSnap;
    *dx = sLogoDx;
    *dy = sLogoDy;
    *px = sLogoReady ? sLogo : NULL;
    return sLogoReady;
}   // gHostIrisRadius as the frame's prepass saw it (the GPU compose's wide-screen iris)
// (Vita) The sky (sSkyTex, GPU2D_SKY_W x GPU2D_SKY_H) this frame, and its line 0's top screen row (544 rows; 2 each).
int gpu2d_gc_sky(const uint32_t **px, int *ofs) {
    *px = sSkyTex;
    *ofs = sSkyTopSnap;
    return sSkyReady;
}
void gpu2d_gc_sky_take_rows(unsigned char *rows) {   // (as gpu2d_gc_logo_take_rows: the rows written since)
    for (int y = 0; y < SKY_TH; y++) rows[y] = __atomic_exchange_n(&sSkyRows[y], 0, __ATOMIC_ACQ_REL);
}
const uint32_t *gpu2d_canvas(int *rows) {
    *rows = sCanvasRows;
    return sCanvasRows ? sCanvas : NULL;
}
static u32 argb(u32 c) {
    u32 r = c & 0x3f, g = c >> 8 & 0x3f, b = c >> 16 & 0x3f;
    r = r << 2 | r >> 4; g = g << 2 | g >> 4; b = b << 2 | b >> 4;
    return 0xff000000u | r << 16 | g << 8 | b;
}
static u32 sExtSky[SCREEN_W * CANVAS_MAX];
static int sExtRows;
static void render_ext_sky(const Gpu2dFrameInput *in) {
    extern int gHostCanvasRows;
    Engine *b = &sEngine[1];
    int rows = gHostCanvasRows < SCREEN_H ? SCREEN_H : gHostCanvasRows > CANVAS_MAX ? CANVAS_MAX : gHostCanvasRows;
    int t = rows - SCREEN_H, ext = t - SCREEN_H;
    sExtRows = 0;
    extern int MenuCtrl_IsIdle(void);
    if (ext <= 0 || !sA3dPrev || (sPowCnt & 0x8000) || !MenuCtrl_IsIdle() || !(sPowCnt & 0x200) || (reg32(b, REG_DISPCNT) & 0x80) || !(reg32(b, REG_DISPCNT) >> 16 & 1)) return;
    const u32 *top = in->topRows == t ? in->lines3dTop : NULL;
    ObjLine *saved = malloc(sizeof *saved);
    if (!saved) return;
    *saved = b->obj;
    memset(&b->obj, 0, sizeof b->obj);
    u8 dc[4], bc[8];
    memcpy(dc, b->regs + REG_DISPCNT, 4);
    memcpy(bc, b->regs + REG_BGCNT, 8);
    {
        u32 d = reg32(b, REG_DISPCNT);
        int back = -1, bp = -1;
        for (int n = 0; n < 4; n++) {
            if (!(d & 0x100u << n)) continue;
            int p = reg16(b, REG_BGCNT + 2 * n) & 3;
            if (p >= bp) bp = p, back = n;
        }
        extern unsigned Scene_GetCurrent(void);
        if (Scene_GetCurrent() == 0x2c) {
            d &= ~0x1000u;
            d &= ~0x0f00u;
            if (back >= 0) d |= 0x100u << back;
        }
        for (int k = 0; k < 4; k++) b->regs[REG_DISPCNT + k] = (u8)(d >> 8 * k);
    }
    static u8 stars[0x400];
    const u8 *oam = b->oam;
    memcpy(stars, oam, sizeof stars);
    for (int i = 0; i < 128; i++) {
        u8 *o = stars + i * 8;
        if ((o[1] >> 6) != 0 || (o[3] >> 6) > 1 || (o[1] & 1)) o[0] = 0xc0, o[1] = 0x02;
    }
    b->oam = stars;
    extern int Sky_HostExtLine(int line, s32 *out);
    s32 sl[5];
    u8 regs[REG_SIZE], back[2], rw[2];
    s32 rx[2], ry[2], rl = b->refLine;
    memcpy(rx, b->refX, sizeof rx), memcpy(ry, b->refY, sizeof ry), memcpy(rw, b->refWritten, sizeof rw);
    memcpy(regs, b->regs, REG_SIZE);
    memcpy(back, in->palette + 0x400, 2);
    for (int r = 0; r < ext; r++) {
        if (Sky_HostExtLine(r - ext, sl)) {
            u8 *pal = (u8 *)in->palette + 0x400;
            pal[0] = (u8)sl[0], pal[1] = (u8)(sl[0] >> 8);
            b->regs[0x30] = (u8)sl[1], b->regs[0x31] = (u8)(sl[1] >> 8);
            for (int k = 0; k < 4; k++) b->regs[0x38 + k] = (u8)((u32)sl[2] >> 8 * k), b->regs[0x3c + k] = (u8)((u32)sl[3] >> 8 * k);
            b->regs[0x52] = (u8)sl[4], b->regs[0x53] = (u8)(sl[4] >> 8);
            b->regs[REG_BGCNT + 7] |= 0x20;
            for (int i = 0; i < 2; i++) {
                b->refX[i] = ref28(b, 0x28 + 0x10 * (u32)i);
                b->refY[i] = ref28(b, 0x2c + 0x10 * (u32)i);
            }
            b->refLine = r - ext;
        }
        for (int n = 0; n < 4; n++) gpu2d_render_bg_line(b, n, r - ext);
        gpu2d_render_obj_line(b, ((r - ext) % SCREEN_H + SCREEN_H) % SCREEN_H, &b->obj);
        compose_line(b, 0, top ? top + r * SCREEN_W : NULL, sExtSky + r * SCREEN_W);
    }
    memcpy(b->regs, regs, REG_SIZE);
    memcpy((u8 *)in->palette + 0x400, back, 2);
    memcpy(b->refX, rx, sizeof rx), memcpy(b->refY, ry, sizeof ry), memcpy(b->refWritten, rw, sizeof rw);
    b->refLine = rl;
    memcpy(b->regs + REG_DISPCNT, dc, 4);
    memcpy(b->regs + REG_BGCNT, bc, 8);
    b->oam = oam;
    b->obj = *saved;
    free(saved);
    sExtRows = ext;
}
static void single_compose(const Gpu2dFrameInput *in, int blank) {
    static u32 ref[SCREEN_W];
    static int haveRef;
    const Engine *a = &sEngine[0], *b = &sEngine[1];
    extern int gHostCanvasRows;
    int want = gHostCanvasRows < SCREEN_H ? SCREEN_H : gHostCanvasRows > CANVAS_MAX ? CANVAS_MAX : gHostCanvasRows;
    int t = want - SCREEN_H;
    int rows = want;
    const u32 *top3d = in->topRows == t ? in->lines3dTop : NULL;
    u16 mb = reg16(a, REG_MASTER_BRIGHT);
    u32 f = mb & 0x1f;
    if (f > 16) f = 16;
    if (blank) {
        u32 c = 0;
        if ((mb >> 14) == 1 && f) c = brighten(c, f, 0);
        else if ((mb >> 14) == 2 && f) c = darken(c, f, 15);
        for (int i = 0; i < SCREEN_W * rows; i++) sCanvas[i] = argb(c);
        sCanvasRows = rows;
        return;
    }
    int sw = (sPowCnt & 0x8000) != 0;
    extern int MenuCtrl_IsIdle(void);
    extern int MenuCtrl_IsMenuOpen(void);
    int menu = sw || (!MenuCtrl_IsIdle() && MenuCtrl_IsMenuOpen());
    static u32 skyRow[CANVAS_MAX], skyPix[CANVAS_MAX * SCREEN_W];
    static int skyRowHave;
    int iris = gHostIrisRadius >= 0 && (reg32(b, REG_DISPCNT) & 0x2000);
    int bLast = 0;
    for (int x = 0; x < SCREEN_W; x++) bLast |= !sBBack[(SCREEN_H - 1) * SCREEN_W + x];
    if (!sw && bLast && ((!menu && !iris && !(reg16(b, REG_MASTER_BRIGHT) & 0x1f) && !(mb & 0x1f)) || !haveRef)) {
        memcpy(ref, b->out + (SCREEN_H - 1) * SCREEN_W, sizeof ref);
        haveRef = 1;
    }
    static u32 skyTop, skyCand;
    static int skyHave, skyRun;
    {
        u32 dom = b->out[0];
        int best = 0;
        for (int x = 0; x < SCREEN_W; x += 4) {
            int n = 0;
            for (int k = 0; k < SCREEN_W; k += 4) n += b->out[k] == b->out[x];
            if (n > best) best = n, dom = b->out[x];
        }
        if (!skyHave) skyTop = dom, skyHave = 1;
        if (dom == skyTop) skyRun = 0;
        else if (dom == skyCand) { if (++skyRun >= 30) skyTop = dom, skyRun = 0; }
        else skyCand = dom, skyRun = 1;
    }
    double cx = SCREEN_W / 2.0, cy = rows / 2.0, rr = (double)gHostIrisRadius * gHostIrisRadius;
    for (int y = 0; y < rows; y++) {
        u32 *d = sCanvas + y * SCREEN_W;
        if (y >= t) {
            const u32 *src = a->out + (y - t) * SCREEN_W;
            const u8 *sky = sSkyA + (y - t) * SCREEN_W;
            const u32 *bm = b->out + (y - t) * SCREEN_W;
            const u8 *bb = sBBack + (y - t) * SCREEN_W;
            const u8 *at = sATop + (y - t) * SCREEN_W;
            const u8 *bk = sBKeep + (y - t) * SCREEN_W;
            for (int x = 0; x < SCREEN_W; x++)
                d[x] = argb(sw && !bb[x] && (at[x] >= LAYER_BACKDROP || bk[x]) ? bm[x] : sky[x] ? ref[x] : src[x]);
            continue;
        }
        if (menu && skyRowHave) {
            const u32 *l3 = top3d && sA3dPrev ? top3d + y * SCREEN_W : NULL;
            for (int x = 0; x < SCREEN_W; x++) {
                u32 o = skyPix[y * SCREEN_W + x], c = l3 ? l3[x] : 0, al = c >> 24 & 31;
                if (al) o = al == 31 ? c & 0x3f3f3f : blend3d(c & 0x3f3f3f, o, al);
                if ((mb >> 14) == 1 && f) o = brighten(o, f, 0);
                else if ((mb >> 14) == 2 && f) o = darken(o, f, 15);
                d[x] = argb(o);
            }
            continue;
        }
        int by = y - (t - SCREEN_H);
        if (by >= 0 && !sw) {
            for (int x = 0; x < SCREEN_W; x++) d[x] = argb(b->out[by * SCREEN_W + x]);
            continue;
        }
        if (y < sExtRows) {
            const u32 *src = sExtSky + y * SCREEN_W;
            for (int x = 0; x < SCREEN_W; x++) {
                u32 o = src[x];
                if ((mb >> 14) == 1 && f) o = brighten(o, f, 0);
                else if ((mb >> 14) == 2 && f) o = darken(o, f, 15);
                if (iris) {
                    double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
                    if (dx * dx + dy * dy > rr) o = 0;
                }
                d[x] = argb(o);
            }
            continue;
        }
        const u32 *l3 = top3d ? top3d + y * SCREEN_W : NULL;
        for (int x = 0; x < SCREEN_W; x++) {
            u32 o = sA3dPrev ? skyTop : 0;
            u32 c = l3 && sA3dPrev ? l3[x] : 0, al = c >> 24 & 31;
            if (al) {
                o = al == 31 ? c & 0x3f3f3f : blend3d(c & 0x3f3f3f, o, al);
                if ((mb >> 14) == 1 && f) o = brighten(o, f, 0);
                else if ((mb >> 14) == 2 && f) o = darken(o, f, 15);
            }
            if (iris) {
                double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
                if (dx * dx + dy * dy > rr) o = 0;
            }
            d[x] = argb(o);
        }
    }
    if (!menu && !iris && sA3dPrev) {
        for (int y = 0; y < t; y++) {
            int by = y - (t - SCREEN_H);
            const u32 *src = by >= 0 ? b->out + by * SCREEN_W : y < sExtRows ? sExtSky + y * SCREEN_W : NULL;
            if (!src) {
                skyRow[y] = skyTop;
                for (int x = 0; x < SCREEN_W; x++) skyPix[y * SCREEN_W + x] = skyTop;
                continue;
            }
            u32 dom = src[0];
            int best = 0;
            for (int x = 0; x < SCREEN_W; x += 8) {
                int n = 0;
                for (int k = 0; k < SCREEN_W; k += 8) n += src[k] == src[x];
                if (n > best) best = n, dom = src[x];
            }
            skyRow[y] = dom;
            for (int x = 0; x < SCREEN_W; x++) skyPix[y * SCREEN_W + x] = src[x];
        }
        skyRowHave = 1;
    }
    if (iris)
        for (int y = 0; y < rows; y++) {
            u32 *d = sCanvas + y * SCREEN_W;
            double dy = y + 0.5 - cy;
            for (int x = 0; x < SCREEN_W; x++) {
                double dx = x + 0.5 - cx;
                if (dx * dx + dy * dy > rr) d[x] = argb(0);
            }
        }
    sCanvasRows = rows;
}

static int sTouchTop;
int gpu2d_touch_on_top(void) { return 0; }
extern int gHostCanvasRows;
static void classic_compose(const Gpu2dFrameInput *in) {
    int rows = gHostCanvasRows < SCREEN_H ? SCREEN_H : gHostCanvasRows > CANVAS_MAX ? CANVAS_MAX : gHostCanvasRows;
    int aTop = (sPowCnt & 0x8000) != 0;
    const Engine *top = &sEngine[1], *bottom = &sEngine[0], *a = &sEngine[0];
    sTouchTop = aTop;
    aTop = 0;
    int ext = rows - 2 * SCREEN_H;
    u16 mb = reg16(a, REG_MASTER_BRIGHT);
    u32 f = mb & 0x1f;
    if (f > 16) f = 16;
    for (int y = 0; y < rows; y++) {
        u32 *d = sCanvas + y * SCREEN_W;
        int sy = y - ext;
        if (sy >= SCREEN_H) {
            for (int x = 0; x < SCREEN_W; x++) d[x] = argb(bottom->out[(sy - SCREEN_H) * SCREEN_W + x]);
        } else if (sy >= 0) {
            for (int x = 0; x < SCREEN_W; x++) d[x] = argb(top->out[sy * SCREEN_W + x]);
        } else {
            const u32 *l3 = aTop && in->lines3dTop && y < in->topRows ? in->lines3dTop + y * SCREEN_W : NULL;
            for (int x = 0; x < SCREEN_W; x++) {
                u32 c = l3 ? l3[x] : 0, al = c >> 24 & 31, o = 0;
                if (al) {
                    o = al == 31 ? c & 0x3f3f3f : blend3d(c & 0x3f3f3f, 0, al);
                    if ((mb >> 14) == 1 && f) o = brighten(o, f, 0);
                    else if ((mb >> 14) == 2 && f) o = darken(o, f, 15);
                }
                d[x] = argb(o);
            }
        }
    }
    sCanvasRows = rows;
}

// ---- Multi-threaded engine-0 line rendering ------------------------------------------------------------------
// The per-pixel 2D compositing (compose_line) is the dominant 2D cost and runs single-threaded per scanline. The
// 3D raster worker cores are now idle (3D moved to the GPU), so render engine-0's scanlines across threads.
// The only cross-line state is cumulative: the affine ref points (refX/refY), the vertical-window toggles
// (winActive) and the object-mosaic counters. A cheap sequential pre-pass advances exactly that state (no pixel
// work) and snapshots it at band boundaries; each band then renders its line range from its snapshot into a
// private Engine copy, and the lines are gathered back. hblank callbacks and both-engine ref advance stay in the
// sequential pre-pass. Gated by ACWW_2DMT (default on); the sequential path remains for sGpu/diagnostics/engine-1.
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#define N2DBANDS 3
static Engine s2dBandEng[N2DBANDS], s2dBandEngB[N2DBANDS];
static u8 s2dRegSnap[SCREEN_H][REG_SIZE];   // per-line register state (hblank can change scroll/etc. per scanline)
static u8 s2dPalSnap[SCREEN_H][0x400];       // per-line palette (hblank can write palette per scanline: sky/water)
static const u8 *s2dPalLine[SCREEN_H];       // each line's palette: its own snapshot, or the previous line's if equal
typedef struct { s32 refX[2], refY[2]; int refLine; u8 refWritten[2], winActive[2]; int objMosaicLine, objMosaicCount; } Snap2d;
typedef struct {
    const Gpu2dFrameInput *in; Engine *src; Engine *be; int lo, hi; Snap2d snap; u32 *canvas;
    int mirror;                                 // engine B into sLogo: 1 whole, 2 menu overlay; 3 the sky (sSkyTex)
    const u8 *dirty;                                                // (mirror) lines to draw; the others keep theirs
    const u8 (*regSnap)[REG_SIZE], (*palSnap)[0x400];               // other per-line snapshots (engine B's), or NULL
    int part, nparts;   // (menu overlay) every band walks all lines and draws every nparts-th changed one from part
} Band2d;
static u8 s2dRegSnapB[SCREEN_H][REG_SIZE];   // (Vita) engine B's per-line registers and palettes, for the menu overlay
static u8 s2dPalSnapB[SCREEN_H][0x400];

static void ref_advance(Engine *e, int line) {
    for (int i = 0; i < 2; i++) {
        u32 r = REG_BG2PA + 0x10 * (u32)i;
        e->refX[i] = e->refWritten[i] & 1 ? ref28(e, 0x28 + 0x10 * (u32)i)
                                          : e->refX[i] + (s16)reg16(e, r + 2) * (line + 1 - e->refLine);
        e->refY[i] = e->refWritten[i] & 2 ? ref28(e, 0x2c + 0x10 * (u32)i)
                                          : e->refY[i] + (s16)reg16(e, r + 6) * (line + 1 - e->refLine);
        e->refWritten[i] = 0;
    }
    e->refLine = line + 1;
}

// ---- GPU compose: per-line layer output (see gpu2d.h) ----
static Gpu2dGc sGcSets[2], *sGcCur = &sGcSets[0];
#define sGc (*sGcCur)
void gpu2d_set_gc_buffers(int set, uint16_t *const bg[4], uint16_t *obj, uint16_t *aux, uint8_t *par) {
    Gpu2dGc *g = &sGcSets[set & 1];
    for (int n = 0; n < 4; n++) g->bg[n] = bg[n];
    g->obj = obj; g->aux = aux; g->par = par; g->set = set & 1;
}
static void gc_buffers_default(Gpu2dGc *g, int set) {
    if (g->par) return;
    for (int n = 0; n < 4; n++) g->bg[n] = calloc(SCREEN_W * SCREEN_H, 2);
    g->obj = calloc(SCREEN_W * SCREEN_H, 2);
    g->aux = calloc(SCREEN_W * SCREEN_H, 2);
    g->par = calloc(SCREEN_H, GPU2D_GC_PAR * 4);
    g->set = set;
}
static void gc_test_dump(void);
static int sGcOn, sGcFrame, sGcFrameDone;
void gpu2d_set_gpu_compose(int on) { sGcOn = on; }
#if !HOST_VITA
__attribute__((constructor)) static void gc_test_init(void) {
    if (getenv("ACWW_GCTEST") && getenv("ACWW_GCMODE") && atoi(getenv("ACWW_GCMODE"))) sGcOn = 1;
}
#endif
const Gpu2dGc *gpu2d_gc_frame(void) { return sGcFrameDone ? sGcCur : NULL; }
// Parameter texels per line: 0 = BG priorities (4 = not shown), 1 = first/second target layers, effect | sprites
// on << 2 | 3D on << 3, EVY; 2 = EVA, EVB, master-brightness mode, factor; 3 = backdrop (6-bit r, g, b);
// 4 = 3D horizontal offset + 256 (low, high byte).
static void gc_line(Engine *e, const Gpu2dFrameInput *in, int line) {
    u32 dispcnt = reg32(e, REG_DISPCNT);
    for (int n = 0; n < 4; n++) gpu2d_render_bg_line(e, n, line);
    ObjLine *obj = &e->obj;
    u8 mask[SCREEN_W];
    window_line(e, obj, mask);
    obj_mosaic_x(e, obj);
    const int has3d = (dispcnt & 8) && (dispcnt & 0x100) && in->lines3d;
    u8 p[GPU2D_GC_PAR * 4] = {0};
    u8 uni = 0;
    for (int n = 0; n < 4; n++) {
        int on = (dispcnt & 0x100u << n) && (n == 0 && (dispcnt & 8) ? has3d : e->bgAny[n]);
        p[n] = on ? (u8)(reg16(e, REG_BGCNT + 2 * n) & 3) : 4;
        if (on && !(n == 0 && (dispcnt & 8))) {
            memcpy(sGc.bg[n] + line * SCREEN_W, e->bg[n], sizeof e->bg[n]);
            __atomic_fetch_or(&sGc.used, 1u << n, __ATOMIC_RELAXED);
            // a layer that is one opaque colour across the whole line (a black curtain, a solid background):
            // a wider canvas continues it beside the 2D area (p[23] bit n)
            if (kWideCanvas && (e->bg[n][0] & PIX_OPAQUE)) {
                int x = 1;
                while (x < SCREEN_W && e->bg[n][x] == e->bg[n][0]) x++;
                if (x == SCREEN_W) uni |= (u8)(1u << n);
            }
        }
    }
    u16 bldcnt = reg16(e, REG_BLDCNT), bldalpha = reg16(e, REG_BLDALPHA), mb = reg16(e, REG_MASTER_BRIGHT);
    u32 eva = bldalpha & 0x1f, evb = bldalpha >> 8 & 0x1f, evy = reg16(e, REG_BLDY) & 0x1f, mbf = mb & 0x1f;
    p[4] = (u8)(bldcnt & 0x3f);
    p[5] = (u8)(bldcnt >> 8 & 0x3f);
    p[6] = (u8)((bldcnt >> 6 & 3) | ((dispcnt & 0x1000) ? 4 : 0) | (has3d ? 8 : 0));
    p[7] = (u8)(evy > 16 ? 16 : evy);
    p[8] = (u8)(eva > 16 ? 16 : eva);
    p[9] = (u8)(evb > 16 ? 16 : evb);
    p[10] = (u8)(mb >> 14);
    p[11] = (u8)(mbf > 16 ? 16 : mbf);
    u32 bd = rgb6(pal16(e->palette, 0));   // (the sky, when shown, takes its place in the compose: uSkyOn)
    p[12] = (u8)(bd & 0x3f); p[13] = (u8)(bd >> 8 & 0x3f); p[14] = (u8)(bd >> 16 & 0x3f); p[15] = 0;
    s32 hofs3d = reg16(e, REG_BGHOFS) & 0x1ff;
    if (hofs3d & 0x100) hofs3d -= 0x200;
    p[16] = (u8)((hofs3d + 256) & 0xff); p[17] = (u8)((hofs3d + 256) >> 8); p[18] = p[19] = 0;
    // The iris wipe's circle (radius in pixels from the screen centre) while window 0 is on: a wider canvas than
    // the DS's 256 columns continues the circle beside the 2D layers.
    if (sIrisSnap >= 0 && (dispcnt & 0x2000)) { p[20] = (u8)(sIrisSnap & 0xff); p[21] = (u8)(sIrisSnap >> 8); p[22] = 1; }
    p[23] = uni;
    // (built here and copied out whole: the destination may be uncached GPU memory)
    u16 oc[SCREEN_W], ax[SCREEN_W];
    int anyObj = 0;
    // Coverage for the compose (it skips the layers of pixels without any): bit 7 of the low byte = an opaque pixel
    // of a shown 2D BG 0-2 or a sprite, bit 6 = of BG3 (the title moves BG3 sideways: it is looked up apart). The
    // window mask uses bits 0-5.
    const u16 *cb[3];
    int ncb = 0;
    for (int n = 0; n < 3; n++)
        if (p[n] != 4 && !(n == 0 && (dispcnt & 8))) cb[ncb++] = e->bg[n];
    const int cov3 = p[3] != 4;
    // BG1/BG2 that, when opaque, are in front of the 3D (a lower priority number; at equal priority BG0 wins)
    const u16 *fr[2];
    int nfr = 0;
    if (dispcnt & 8)
        for (int n = 1; n < 3; n++)
            if (p[n] != 4 && p[n] < p[0]) fr[nfr++] = e->bg[n];
    u8 behind[SCREEN_W];
    for (int x = 0; x < SCREEN_W; x++) {
        u16 a = mask[x];
        for (int i = 0; i < ncb; i++)
            if (cb[i][x] & PIX_OPAQUE) { a |= 0x80; break; }
        if (cov3 && (e->bg[3][x] & PIX_OPAQUE)) a |= 0x40;
        ax[x] = a;
        int front = 0;
        for (int i = 0; i < nfr; i++) front |= fr[i][x] & PIX_OPAQUE;
        behind[x] = (dispcnt & 8) && (a & 0x80) && !front;
    }
    // A pixel whose only 2D pixels (of BG 0-2) are behind an opaque 3D pixel: the obj texture's 0x4000 where there is
    // no sprite (a sprite's own entry has 0x8000) - the compose's fast path may take it when the 3D there is opaque.
    for (int x = 0; x < SCREEN_W; x++) oc[x] = behind[x] ? 0x4000 : 0;
    if (dispcnt & 0x1000) {
        for (int x = obj->lo < 0 ? 0 : obj->lo, xe = obj->hi > SCREEN_W ? SCREEN_W : obj->hi; x < xe; x++) {
            if (!obj->source[x]) continue;
            oc[x] = (u16)(obj_color(e, dispcnt, obj, x) | 0x8000);
            ax[x] |= (u16)((obj->prio[x] & 3) | (obj->kind[x] & 3) << 2 | (obj->alpha[x] & 15) << 4) << 8 | 0x80;
            anyObj = 1;
        }
    }
    if (anyObj) __atomic_fetch_or(&sGc.used, 1u << 4, __ATOMIC_RELAXED);
    {   // Where on this line the compose may show just the 3D or the backdrop (gpu2d_compose.h gc_tile_quads): bit k of
        // p[24..25] = some pixel of columns 16k..16k+15 has a 2D pixel or a sprite or hides the 3D by its window;
        // p[26] bit 0/1: the left/right side beside the 2D area can't (a layer carried there, or the edge column's
        // window hides the 3D).
        u32 g = 0;
        for (int x = 0; x < SCREEN_W; x++)
            if ((ax[x] & 0xc0) || (oc[x] & 0x8000) || !(ax[x] & 1)) g |= 1u << (x >> 4);
        p[24] = (u8)g; p[25] = (u8)(g >> 8);
        p[26] = (u8)((uni || !(ax[0] & 1) ? 1 : 0) | (uni || !(ax[SCREEN_W - 1] & 1) ? 2 : 0));
        u32 g3 = 0;   // p[27..28]: the groups with a BG3 pixel (the title moves BG3 sideways: the tiles follow it)
        for (int x = 0; x < SCREEN_W; x++)
            if (ax[x] & 0x40) g3 |= 1u << (x >> 4);
        p[27] = (u8)g3; p[28] = (u8)(g3 >> 8);
    }
    memcpy(sGc.obj + line * SCREEN_W, oc, sizeof oc);
    memcpy(sGc.aux + line * SCREEN_W, ax, sizeof ax);
    memcpy(sGc.par + line * GPU2D_GC_PAR * 4, p, sizeof p);
    gpu2d_feature(e, (enum Feature)(F_MODE0 + (dispcnt & 7)));
    obj_mosaic_y(e, line);
}
// The GPU composes a frame only when every line is a plain composited line (display mode 1, no forced blank).
static int gc_frame_ok(void) {
    for (int line = 0; line < SCREEN_H; line++) {
        u32 d = (u32)(s2dRegSnap[line][0] | s2dRegSnap[line][1] << 8 | s2dRegSnap[line][2] << 16 | s2dRegSnap[line][3] << 24);
        if ((d & 0x80) || (d >> 16 & 3) != 1) return 0;
    }
    return 1;
}

static void band2d_run(Band2d *w);
static Band2d sMtWB[N2DBANDS];   // (Vita) engine B's bands, run after engine A's by the same thread (menu overlay)
static void *band2d_worker(void *arg) {
    Band2d *w = arg;
    band2d_run(w);
    {
        extern Band2d sMtW[];
        ptrdiff_t b = w - sMtW;
        if (b >= 0 && b < N2DBANDS && sMtWB[b].lo < sMtWB[b].hi) band2d_run(&sMtWB[b]);
    }
    return 0;
}
unsigned long long gBProfNs, gBProfLines, gBProfRuns;   // (host profiling: engine B overlay lines)
static void band2d_run_(Band2d *w);
static void band2d_run(Band2d *w) {
    if (w->mirror != 2) { band2d_run_(w); return; }
    uint64_t t0 = host_plat_time_ns();
    band2d_run_(w);
    __atomic_fetch_add(&gBProfNs, host_plat_time_ns() - t0, __ATOMIC_RELAXED);
    unsigned n = 0, k = 0;
    for (int l = w->lo; l < w->hi; l++)
        if (w->dirty && w->dirty[l] && (w->nparts <= 1 || k++ % (unsigned)w->nparts == (unsigned)w->part)) n++;
    __atomic_fetch_add(&gBProfLines, n, __ATOMIC_RELAXED);
    __atomic_fetch_add(&gBProfRuns, 1, __ATOMIC_RELAXED);
}
// (the sky) n pixels of an engine's line into the sky texture's RGBA
static void sky_put(u32 *d, const u32 *o, int n) {
    for (int x = 0; x < n; x++) {
        u32 c = argb(o[x]);
        d[x] = (c >> 16 & 0xff) | (c & 0xff00) | (c & 0xff) << 16 | 0xff000000u;
    }
}
// (the sky) The DS's columns beside its screen on this sky line - x -SKY_SIDE..-1 and 256..256+SKY_SIDE-1 - as the
// line's registers draw them: the line drawn twice more with everything moved 256 columns, each time keeping the
// columns next to the screen. Text BGs' horizontal scroll and sprites' x are 9 bits (512 columns: 256 either way is
// the same move); the affine BGs' start moves by 256 steps of their x step (pa, pc), and the clouds' BG3 wraps
// round (as the tall PC sky's lines above the screen: render_ext_sky) - its picture goes on, not stopping at the
// edge of its map. Everything the line's own drawing uses is put back afterwards.
static void sky_side_lines(Engine *e, const Band2d *w, int line, u32 *d) {
    static ObjLine objSide[N2DBANDS], objKeep[N2DBANDS];
    static u8 oamSide[N2DBANDS][0x400];
    const int b = w->part < N2DBANDS ? w->part : 0;
    u8 *oam = oamSide[b];
    memcpy(oam, e->oam, 0x400);
    for (int i = 0; i < 128; i++) oam[i * 8 + 3] ^= 1;   // (attribute 1 bit 8: x + 256)
    const u8 *oamKeep = e->oam;
    const int ml = e->objMosaicLine, mc = e->objMosaicCount;
    u8 regs[REG_SIZE];
    s32 rx[2], ry[2];
    memcpy(regs, e->regs, REG_SIZE);
    memcpy(rx, e->refX, sizeof rx), memcpy(ry, e->refY, sizeof ry);
    objKeep[b] = e->obj;
    e->oam = oam;
    gpu2d_render_obj_line(e, line, &objSide[b]);
    e->oam = oamKeep;
    e->obj = objSide[b];
    for (int n = 0; n < 4; n++) e->regs[REG_BGHOFS + 4 * n + 1] ^= 1;
    e->regs[REG_BGCNT + 7] |= 0x20;
    for (int side = 0; side < 2; side++) {
        const s32 sh = side ? SCREEN_W : -SCREEN_W;
        for (int i = 0; i < 2; i++) {
            const s32 pa = (s16)reg16(e, REG_BG2PA + 0x10 * (u32)i), pc = (s16)reg16(e, REG_BG2PA + 4 + 0x10 * (u32)i);
            e->refX[i] = rx[i] + pa * sh;
            e->refY[i] = ry[i] + pc * sh;
        }
        e->objMosaicLine = ml, e->objMosaicCount = mc;
        render_line(e, w->in, line);
        const u32 *o = e->out + line * SCREEN_W;
        if (side) sky_put(d + SKY_SIDE + SCREEN_W, o, SKY_SIDE);
        else sky_put(d, o + SCREEN_W - SKY_SIDE, SKY_SIDE);
    }
    memcpy(e->regs, regs, REG_SIZE);
    memcpy(e->refX, rx, sizeof rx), memcpy(e->refY, ry, sizeof ry);
    e->objMosaicLine = ml, e->objMosaicCount = mc;
    objSide[b] = e->obj;
    e->obj = objKeep[b];
}
static void band2d_run_(Band2d *w) {
    Engine *e = w->be;
    // e already holds a copy of the engine's inputs (regs, VRAM/palette/oam ptrs, objNextLine0 ...), taken on the
    // main thread at the end of the prepass - the live engine may be written by the game while this band runs.
    e->refX[0] = w->snap.refX[0]; e->refX[1] = w->snap.refX[1];
    e->refY[0] = w->snap.refY[0]; e->refY[1] = w->snap.refY[1];
    e->refLine = w->snap.refLine; e->refWritten[0] = w->snap.refWritten[0]; e->refWritten[1] = w->snap.refWritten[1];
    e->winActive[0] = w->snap.winActive[0]; e->winActive[1] = w->snap.winActive[1];
    e->objMosaicLine = w->snap.objMosaicLine; e->objMosaicCount = w->snap.objMosaicCount;
    // The band's OAM is fixed for the frame: list the entries that are not disabled once, instead of decoding
    // all 128 on every line (entries are walked in the same order, so priority ties resolve the same way).
    e->objCandN = 0;
    for (int i = 0; i < 128; i++) {
        u16 a0 = (u16)(e->oam[i * 8] | e->oam[i * 8 + 1] << 8);
        if ((!(a0 & 0x100) && (a0 & 0x200)) || (a0 >> 14) == 3) continue;
        {   // y and bounding height (as gpu2d_render_obj_line computes them) for a quick per-line reject
            static const u8 kW[3][4] = {{8, 16, 32, 64}, {16, 32, 32, 64}, {8, 8, 16, 32}};
            static const u8 kH[3][4] = {{8, 16, 32, 64}, {8, 8, 16, 32}, {16, 32, 32, 64}};
            (void)kW;
            u16 a1 = (u16)(e->oam[i * 8 + 2] | e->oam[i * 8 + 3] << 8);
            int bh = kH[a0 >> 14][a1 >> 14];
            if ((a0 & 0x100) && (a0 & 0x200)) bh *= 2;
            e->objCandY[e->objCandN] = (u8)(a0 & 0xff);
            e->objCandH[e->objCandN] = (u8)(bh > 255 ? 255 : bh);
        }
        e->objCand[e->objCandN++] = (u8)i;
    }
    e->objCandOn = 1;
    unsigned dirtyK = 0;
    for (int line = w->lo; line < w->hi; line++) {
        memcpy(e->regs, w->regSnap ? w->regSnap[line] : s2dRegSnap[line], REG_SIZE);   // per-line registers
        if (w->mirror == 3) e->regs[REG_DISPCNT + 1] &= (u8)~0x04;   // (the sky never shows BG2: the title's logo)
        e->palette = w->palSnap ? w->palSnap[line] : s2dPalLine[line];                  // per-line palette
        window_vertical(e, line);
        int mine = 1;   // (menu overlay: the changed lines are dealt out to the bands in turn, wherever they are)
        if (w->mirror && w->dirty && w->dirty[line] && w->nparts > 1) mine = dirtyK++ % (unsigned)w->nparts == (unsigned)w->part;
        if (w->mirror && w->dirty && (!w->dirty[line] || !mine)) {   // this line of engine B's picture did not change
            obj_mosaic_y(e, line);
            ref_advance(e, line);
            if (line + 1 == w->hi) e->objCandOn = 0;
            continue;
        }
        if (line == 0 && e->haveNextLine0) e->obj = e->objNextLine0;
        else if (line == 0) memset(&e->obj, 0, sizeof e->obj);
        else gpu2d_render_obj_line(e, line, &e->obj);
        if (w->mirror == 3) {   // (Vita) the sky: the line, and the DS's columns either side of its screen
            u32 *d = sSkyTex + (line + 1) * SKY_TW;
            render_line(e, w->in, line);
            sky_put(d + SKY_SIDE, e->out + line * SCREEN_W, SCREEN_W);
            sky_side_lines(e, w, line, d);
            ref_advance(e, line);
            if (line + 1 == w->hi) e->objCandOn = 0;
            __atomic_store_n(&sSkyRows[line + 1], 1, __ATOMIC_RELEASE);
            continue;
        }
        if (w->mirror) {   // (Vita) engine B's whole picture, for the mirror (the menus, the boot screens)
            render_line(e, w->in, line);
            ref_advance(e, line);
            if (line + 1 == w->hi) e->objCandOn = 0;
            const u32 *o = e->out + line * SCREEN_W;
            u32 *d = sLogo + line * SCREEN_W;
            const u8 *bb = sBBack + line * SCREEN_W, *bk = sBKeep + line * SCREEN_W;
            for (int x = 0; x < SCREEN_W; x++) {
                u32 c = argb(o[x]);
                u32 a = w->mirror != 2 ? 0xffu : bb[x] ? 0u : bk[x] ? 0xffu : 0x80u;   // (1 the boot screens: all)
                d[x] = (c >> 16 & 0xff) | (c & 0xff00) | (c & 0xff) << 16 | a << 24;
            }
            __atomic_store_n(&sLogoRows[line], 1, __ATOMIC_RELEASE);
            continue;
        }
        if (sGcFrame) {
            gc_line(e, w->in, line);    // layers for the GPU compose instead of the CPU compose
            ref_advance(e, line);
            if (line + 1 == w->hi) e->objCandOn = 0;
            continue;
        }
        render_line(e, w->in, line);     // writes e->out[line] (+defer/side effects), advances obj mosaic
        ref_advance(e, line);
        if (line + 1 == w->hi) e->objCandOn = 0;
        if (w->canvas) {   // straight to the canvas: no gather into the engine and no second pass over the frame
            const u32 *o = e->out + line * SCREEN_W;
            u32 *d = w->canvas + line * SCREEN_W;
            for (int x = 0; x < SCREEN_W; x++) d[x] = argb(o[x]);
        }
    }
}

// Engine-0 multi-threaded render, split into a main-thread prepass and a worker-thread bands pass so the bands
// (pure compute over the snapshotted VRAM/registers) can overlap the next frame's CPU on a render thread.
// The prepass runs game code (the hblank IRQ replay) and must stay on the main thread; the bands run no game
// code and read only the per-line snapshots (s2dRegSnap/s2dPalSnap) plus the engine's VRAM pointers.
Band2d sMtW[N2DBANDS];
static int sMtBnd[N2DBANDS + 1];
static int s2dNB = N2DBANDS;   // bands in use (2 on the Vita with the render thread: the render thread + one helper)
static int sMtPending;   // a prepass ran in deferred mode and is waiting for its bands to be run by the worker

// The sub screen's pictures the Vita draws on the CPU (the title logo, the mirrored engine B) are redrawn only when
// what they are drawn from changed: the frame's start state (registers, palettes, OAM, the VRAM banks behind it)
// and, one frame late, the registers the hblank IRQs set line by line. On the title the logo cost 4.3 ms of the
// game thread every frame for a picture that rarely changes.
// Two murmur3-style lanes (rotations carry every bit into every other; a plain multiply only carries upwards, so
// the same change seen twice - one VRAM bank behind two granules - could cancel out).
static inline u32 mix32(u32 h, u32 k) {
    k *= 0xcc9e2d51u; k = k << 15 | k >> 17; k *= 0x1b873593u;
    h ^= k; h = h << 13 | h >> 19;
    return h * 5u + 0xe6546b64u;
}
static u32 hash_words(u32 h, u32 g, const void *p, u32 n, u32 *g_out) {
    const u32 *q = (const u32 *)p;
    for (u32 i = 0; i + 1 < n / 4; i += 2) {
        h = mix32(h, q[i]);
        g = mix32(g, q[i + 1]);
    }
    *g_out = g;
    return h;
}
// The registers a picture is drawn from: engine B's all for the mirror; for the logo only BG2's (DISPCNT, BG2CNT,
// BG2 scroll, BG2 affine parameters and reference point, MOSAIC) - the title's clouds scroll BG3 every frame.
static u32 sub_regs_hash(const Engine *e1, int mirror, u32 h) {
    u32 g;
    if (mirror) {   // (engine B's line also takes engine A's DISPCNT and master brightness: render_line, sSingle)
        h = mix32(mix32(h, reg32(&sEngine[0], REG_DISPCNT)), reg16(&sEngine[0], REG_MASTER_BRIGHT));
        h = hash_words(h, 0, e1->regs, REG_SIZE, &g);
        return mix32(h, g);
    }
    u32 v[12] = {reg32(e1, 0x00), reg16(e1, 0x0c), reg32(e1, 0x18), reg32(e1, 0x20), reg32(e1, 0x24), reg32(e1, 0x28),
                 reg32(e1, 0x2c), reg16(e1, 0x4c), 0, 0, 0, 0};
    h = hash_words(h, 0, v, sizeof v, &g);
    return mix32(h, g);
}
// The 16-colour palettes BG2 (a text BG in 16 colours) draws with: one bit per palette, from its map entries.
// 0xffff when it is not that kind of BG (all 256 colours count). The title animates other colours a few times a
// second, which alone redrew the logo each time.
static u32 bg2_palette_mask(const Engine *e1) {
    u16 cnt = reg16(e1, 0x0c);
    u32 mode = reg32(e1, REG_DISPCNT) & 7;
    if ((cnt & 0x80) || !(mode <= 1 || mode == 3)) return 0xffff;   // 256 colours, or BG2 not a text BG
    static const u8 kBlocks[4] = {1, 2, 2, 4};
    u32 base = (u32)(cnt >> 8 & 31) * 0x800, mask = 0;
    for (u32 b = 0; b < kBlocks[cnt >> 14]; b++)
        for (u32 i = 0; i < 0x800; i += 2) {
            u32 a = (base + b * 0x800 + i) & e1->bgMask;
            const u8 *gp = e1->bgGran[a >> 14];
            if (gp) mask |= 1u << (gp[(a & 0x3fff) + 1] >> 4);
        }
    return mask;
}
static u32 sub_state_hash(const Engine *e1, int mirror) {
    u32 g = 0x9e3779b9u, h = mix32(2166136261u, (u32)mirror);
    h = sub_regs_hash(e1, mirror, h);
    if (mirror)
        h = hash_words(h, g, e1->palette, 0x400, &g);
    else {
        u32 pm = bg2_palette_mask(e1);
        h = mix32(h, pm);
        for (int i = 0; i < 16; i++)
            if (pm >> i & 1) h = hash_words(h, g, e1->palette + i * 32, 32, &g);
    }
    for (int i = 0; i < 8; i++) {   // engine B: 128 KB of BG VRAM (and of OBJ VRAM), 16 KB granules, each once
        int seen = 0;
        for (int j = 0; j < i; j++) seen |= e1->bgGran[j] == e1->bgGran[i];
        h = mix32(h, e1->bgGran[i] != NULL);   // (contents, not where the snapshot copy happens to be)
        if (e1->bgGran[i] && !seen) h = hash_words(h, g, e1->bgGran[i], 0x4000, &g);
    }
    for (int i = 0; i < 4; i++)
        if (e1->bgExtPal[i]) h = hash_words(h, g, e1->bgExtPal[i], 0x2000, &g);
    if (mirror) {
        h = hash_words(h, g, e1->oam, 0x400, &g);
        for (int i = 0; i < 8; i++) {
            int seen = 0;
            for (int j = 0; j < i; j++) seen |= e1->objGran[j] == e1->objGran[i];
            h = mix32(h, e1->objGran[i] != NULL);
            if (e1->objGran[i] && !seen) h = hash_words(h, g, e1->objGran[i], 0x4000, &g);
        }
        if (e1->objExtPal) h = hash_words(h, g, e1->objExtPal, 0x2000, &g);
    }
    return mix32(h, g);
}
// Engine B's whole picture (the menus, the boot screens) is redrawn line by line: only the lines whose registers or
// palette changed, the lines a changed OAM entry covers (before or after), or all of them when its VRAM changed.
// The menus animate a cursor and scroll their background every frame; drawing all of engine B each frame cost the
// menus ~18 ms per band (30 fps).
static u32 bline_sig(const u8 *regs, u32 palh, u32 ignore) {   // a line's registers, without the scroll of BGs not
    u8 t[REG_SIZE];                                                // shown (or in ignore: BGs with nothing to show)
    memcpy(t, regs, REG_SIZE);
    u32 dc = (u32)(t[0] | t[1] << 8 | t[2] << 16 | (u32)t[3] << 24), g;
    for (int n = 0; n < 4; n++)
        if (!(dc >> (8 + n) & 1) || (ignore >> n & 1)) {
            memset(t + 0x10 + 4 * n, 0, 4);
            if (n >= 2) memset(t + 0x20 + 0x10 * (n - 2), 0, 0x10);
        }
    u32 h = hash_words(palh, 0, t, REG_SIZE, &g);
    return mix32(h, g);
}
static void oam_mark(u8 *dirty, const u8 *ent) {   // the lines an OAM entry (attr0/1) covers
    u16 a0 = (u16)(ent[0] | ent[1] << 8), a1 = (u16)(ent[2] | ent[3] << 8);
    if (!(a0 & 0x100) && (a0 & 0x200)) return;   // disabled
    static const u8 kH[3][4] = {{8, 16, 32, 64}, {8, 8, 16, 32}, {16, 32, 32, 64}};
    if ((a0 >> 14) == 3) return;
    int h = kH[a0 >> 14][a1 >> 14];
    if ((a0 & 0x300) == 0x300) h *= 2;
    for (int i = 0; i < h; i++) {
        int y = ((a0 & 0xff) + i) & 0xff;
        if (y < SCREEN_H) dirty[y] = 1;
    }
}
// Engine B's VRAM against last frame's, only where its picture reads it: the map entries of the BGs shown, the
// tiles they point to, the tiles of the sprites shown. Each 32-byte piece is compared (and its copy refreshed) at
// most once a frame, when first needed. A change marks only the lines drawn from it. (Comparing all of engine B's
// VRAM every frame cost 3.8 ms of the game thread on the Vita.)
static u8 sPrevV[2][0x20000];
static u16 sChkStamp[2][0x20000 / 32];
static u8 sChkRes[2][0x20000 / 32];
static u16 sChkFrame;
static const u8 *vram_ptr(const Engine *e1, int k, u32 a) {
    a &= 0x1ffff;
    const u8 *g = k ? e1->objGran[a >> 14] : e1->bgGran[a >> 14];
    return g ? g + (a & 0x3fff) : NULL;
}
static int piece_changed(const Engine *e1, int k, u32 a) {   // the 32-byte piece at a (wrapping in 128 KB)
    u32 c = (a & 0x1ffff) >> 5;
    if (sChkStamp[k][c] == sChkFrame) return sChkRes[k][c];
    static const u8 zero[32];
    const u8 *src = vram_ptr(e1, k, c << 5);
    if (!src) src = zero;
    u8 *q = sPrevV[k] + (c << 5);
    int d = memcmp(src, q, 32) != 0;
    if (d) memcpy(q, src, 32);
    sChkStamp[k][c] = sChkFrame;
    sChkRes[k][c] = (u8)d;
    return d;
}
static int range_changed(const Engine *e1, int k, u32 a, u32 b) {
    int d = 0;
    for (u32 x = a & ~31u; x < b; x += 32) d |= piece_changed(e1, k, x);   // (all of it: keep the copies current)
    return d;
}
static void oam_mark(u8 *dirty, const u8 *ent);
// (the sky) Whether engine B's BG3 - the clouds, an affine BG - has no pixel to show at all: every tile its map uses
// is all colour 0. Then its scroll (it drifts every frame) does not redraw the sky's lines (bline_sig). Worked out
// again when its VRAM changed (vram_mark: any change in what an affine BG reads) or its mode / BG3CNT did.
static int sSkyBg3Empty;
static void sky_bg3_check(const Engine *e1, int changed) {
    static u32 key = ~0u;
    const u32 dc = reg32(e1, REG_DISPCNT), mode = dc & 7, cnt = reg16(e1, REG_BGCNT + 6);
    const u32 k = mode | cnt << 8;
    if (!changed && k == key) return;
    key = k;
    sSkyBg3Empty = 0;
    // affine (modes 1, 2) or extended with a 16-bit map (modes 3-5, BG3CNT bit 7 clear); else (text, bitmap) unknown
    const int ext = mode >= 3 && mode <= 5;
    if (!(mode == 1 || mode == 2 || (ext && !(cnt & 0x80)))) return;
    const u32 side = 128u << (cnt >> 14), n = (side / 8) * (side / 8);
    const u32 cb = (cnt >> 2 & 15) * 0x4000, sb = (cnt >> 8 & 31) * 0x800;
    static u8 seen[1024];
    memset(seen, 0, sizeof seen);
    for (u32 i = 0; i < n; i++) {
        const u8 *mp = vram_ptr(e1, 0, sb + (ext ? i * 2 : i));
        if (!mp) return;
        u32 t = ext ? (u32)(mp[0] | mp[1] << 8) & 0x3ff : mp[0];
        if (seen[t]) continue;
        seen[t] = 1;
        for (u32 b = 0; b < 64; b += 32) {   // (vram_ptr's granules are 16 KB: 32 bytes never cross one)
            const u8 *tp = vram_ptr(e1, 0, cb + t * 64 + b);
            if (!tp) return;
            for (u32 j = 0; j < 32; j++) if (tp[j]) return;
        }
    }
    sSkyBg3Empty = 1;
}
// Marks the lines drawn from changed VRAM; returns 1 when that cannot be narrowed (all lines then).
static int vram_mark(const Engine *e1, const u8 (*regs)[REG_SIZE], u8 *dirty) {
    if (++sChkFrame == 0) { memset(sChkStamp, 0, sizeof sChkStamp); sChkFrame = 1; }
    const u8 *r0 = regs[0];
    u32 dc = (u32)(r0[0] | r0[1] << 8 | r0[2] << 16 | (u32)r0[3] << 24);
    int all = 0;
    if (dc & 0x1000) {   // sprites: the ones whose tiles changed, or whose rotation/scaling group changed
        static const u8 kW[3][4] = {{8, 16, 32, 64}, {16, 32, 32, 64}, {8, 8, 16, 32}};
        static const u8 kH[3][4] = {{8, 16, 32, 64}, {8, 8, 16, 32}, {16, 32, 32, 64}};
        static u8 prevRot[32][8];
        u32 rotChg = 0;
        for (int g = 0; g < 32; g++) {
            u8 cur[8];
            for (int j = 0; j < 4; j++) cur[j * 2] = e1->oam[(g * 4 + j) * 8 + 6], cur[j * 2 + 1] = e1->oam[(g * 4 + j) * 8 + 7];
            if (memcmp(cur, prevRot[g], 8)) rotChg |= 1u << g, memcpy(prevRot[g], cur, 8);
        }
        for (int i = 0; i < 128; i++) {
            const u8 *o = e1->oam + i * 8;
            u16 a0 = (u16)(o[0] | o[1] << 8), a1 = (u16)(o[2] | o[3] << 8), a2 = (u16)(o[4] | o[5] << 8);
            if ((!(a0 & 0x100) && (a0 & 0x200)) || (a0 >> 14) == 3) continue;
            int chg = (a0 & 0x100) && (rotChg >> (a1 >> 9 & 31) & 1);
            if ((a0 >> 10 & 3) == 3) {   // bitmap sprite
                u32 w = kW[a0 >> 14][a1 >> 14], h = kH[a0 >> 14][a1 >> 14];
                u32 start = (dc & 0x40) ? (u32)(a2 & 0x3ff) * (128u << (dc >> 22 & 1)) : (u32)(a2 & 0x3ff) * 16;
                chg |= range_changed(e1, 1, start, start + w * h * 2 + ((dc & 0x40) ? 0 : 512 * h));
            } else {
                u32 w = kW[a0 >> 14][a1 >> 14], h = kH[a0 >> 14][a1 >> 14], tb = (a0 & 0x2000) ? 64 : 32;
                if (dc & 0x10) {
                    u32 start = (u32)(a2 & 0x3ff) * (32u << (dc >> 20 & 3));
                    chg |= range_changed(e1, 1, start, start + (w / 8) * (h / 8) * tb);
                } else {
                    for (u32 ty = 0; ty < h / 8; ty++) {   // 2D mapping: rows of 32 tiles
                        u32 start = ((u32)(a2 & 0x3ff) + ty * 32) * 32;
                        chg |= range_changed(e1, 1, start, start + (w / 8) * tb);
                    }
                }
            }
            if (chg) oam_mark(dirty, o);
        }
    }
    for (int n = 0; n < 4; n++) {   // BG lines showing changed map entries or tiles
        if (!(dc >> (8 + n) & 1)) continue;
        u16 cnt = (u16)(r0[8 + 2 * n] | r0[9 + 2 * n] << 8);
        u32 mode = dc & 7;
        int text = n <= 1 || (n == 2 && (mode == 0 || mode == 1 || mode == 3)) || (n == 3 && mode == 0);
        u32 cb = (u32)(cnt >> 2 & 15) * 0x4000, sb = (u32)(cnt >> 8 & 31) * 0x800;
        if (!text) {   // affine / extended / bitmap BGs: any change in what they can read redraws all
            int sz = cnt >> 14;
            if (!(cnt & 0x80) || (mode != 3 && mode != 4 && mode != 5) || n < 2) {
                u32 side = 128u << sz, mapb = (side / 8) * (side / 8);   // affine: 8-bit map, 64-byte tiles
                if ((mode == 3 || mode == 4 || mode == 5) && n >= 2) mapb *= 2;   // extended: 16-bit map entries
                all |= range_changed(e1, 0, sb, sb + mapb);
                for (u32 m = 0; m < mapb && !all; m += (mapb == (side / 8) * (side / 8) ? 1 : 2)) {
                    const u8 *mp = vram_ptr(e1, 0, sb + m);
                    u32 t = mp ? (mapb == (side / 8) * (side / 8) ? mp[0] : (u32)(mp[0] | mp[1] << 8) & 0x3ff) : 0;
                    all |= range_changed(e1, 0, cb + t * 64, cb + t * 64 + 64);
                }
            } else {   // bitmap BG
                static const u16 kBw[4] = {128, 256, 512, 512}, kBh[4] = {128, 256, 256, 512};
                u32 bytes = (u32)kBw[sz] * kBh[sz] * ((cnt & 4) ? 2 : 1);
                all |= range_changed(e1, 0, sb * 8, sb * 8 + bytes);
            }
            if (all) break;
            continue;
        }
        u32 tb = (cnt & 0x80) ? 64 : 32;
        int sz = cnt >> 14, bw = (sz == 1 || sz == 3) ? 2 : 1, bh = (sz >= 2) ? 2 : 1;
        uint64_t rows = 0;
        for (int b = 0; b < bw * bh; b++)
            for (int rr = 0; rr < 32; rr++) {
                int rowChg = 0;
                for (int c = 0; c < 32; c++) {
                    u32 a = sb + (u32)b * 0x800 + (u32)(rr * 32 + c) * 2;
                    rowChg |= piece_changed(e1, 0, a);
                    const u8 *mp = vram_ptr(e1, 0, a);
                    u32 t = mp ? (u32)(mp[0] | mp[1] << 8) & 0x3ff : 0;
                    rowChg |= range_changed(e1, 0, cb + t * tb, cb + t * tb + tb);
                }
                if (rowChg) rows |= 1ull << ((b / bw) * 32 + rr);
            }
        if (!rows) continue;
        u32 mh = bh * 256 - 1;
        for (int y = 0; y < SCREEN_H; y++) {
            u32 vofs = (u32)(regs[y][0x12 + 4 * n] | regs[y][0x13 + 4 * n] << 8) & 0x1ff;
            if (rows >> (((y + vofs) & mh) >> 3) & 1) dirty[y] = 1;
        }
    }
    return all;
}
static u32 ext_pal_hash(const Engine *e1) {   // the extended palettes engine B has switched on
    u32 g = 0x9e3779b9u, h = 0x85ebca6bu, dc = reg32(e1, REG_DISPCNT);
    if (dc & 0x40000000u)
        for (int i = 0; i < 4; i++)
            if (e1->bgExtPal[i]) h = hash_words(h, g, e1->bgExtPal[i], 0x2000, &g);
    if ((dc & 0x80000000u) && e1->objExtPal) h = hash_words(h, g, e1->objExtPal, 0x2000, &g);
    return mix32(h, g);
}
static u8 sBDirty[SCREEN_H];
static u32 sBLineSig[SCREEN_H], sBVramSig;
static u8 sBOam[0x400];
static int sBValid, sBMode;
static u32 sSubSig, sSubLines;
static int sSubDirty = 1;
unsigned gSubFrames, gSubDraws;   // (render.c's log) frames with a sub-screen picture, and how many redrew it
unsigned long long gSubCheckNs;   // the time the check above took

static void render_engine0_mt_prepass(const Gpu2dFrameInput *in) {
    Engine *e = &sEngine[0];
    // the sub screen's picture: redrawn only when its inputs changed (above)
    int subDraw = 0;
    u32 subLines = 2166136261u;
    const int whole = sMirrorOn || sOverlayOn || sSkyOn;   // engine B's picture is drawn (boot screens, menus, the sky)
    int bAll = 0;   // (whole) every line of engine B is redrawn
    if (whole) {
        uint64_t t0 = host_plat_time_ns();
        const Engine *e1 = &sEngine[1];
        int mode = sMirrorOn ? 1 : sOverlayOn ? 2 : 3;
        u32 vs = ext_pal_hash(e1);   // (the VRAM itself: vram_diff after the lines)
        bAll = (sSkyOn && sSkyBack) || !sBValid || mode != sBMode || vs != sBVramSig || (sMirrorOn && !sMirrorReady) ||
               (sOverlayOn && !sOvlReady) || (sSkyOn && !sSkyReady);
        sBVramSig = vs;
        sBMode = mode;
        if (sSkyOn) sSkyBack = 0;
        memset(sBDirty, 0, sizeof sBDirty);
        if (!bAll)   // OAM entries that changed: the lines they covered and cover now
            for (int i = 0; i < 128; i++) {
                const u8 *o = e1->oam + i * 8, *p = sBOam + i * 8;
                if (memcmp(o, p, 6)) oam_mark(sBDirty, p), oam_mark(sBDirty, o);
            }
        memcpy(sBOam, e1->oam, 0x400);
        gSubCheckNs += host_plat_time_ns() - t0;
        gSubFrames++;
    } else
        sBValid = 0;
    if (sLogoOn) {   // (the title: its logo, the top screen's BG2 - with the sky drawn too, or not)
        uint64_t t0 = host_plat_time_ns();
        u32 sig = sub_state_hash(&sEngine[1], 0);
        gSubCheckNs += host_plat_time_ns() - t0;
        subDraw = sSubDirty || sig != sSubSig || (!sLogoReady && !sLogoFull);
        sSubSig = sig;
        gSubFrames++;
        gSubDraws += subDraw;
    } else
        sSubDirty = 1;
    if (subDraw) sLogoFull = 0;
    // Engine B's whole picture (a menu, a boot screen) is drawn by the bands from its per-line snapshots, off the
    // game thread - only its changed lines (sBDirty). The boot screens: the bands draw only engine B (engine A is
    // not seen). The menus: engine A as always, then engine B's changed lines (sMtWB) to go over it.
    const int mirrorBand = sMirrorOn;
    const int overlayBand = sOverlayOn;
    const int skyBand = sSkyOn;
    const int bSnap = overlayBand || skyBand;   // engine B's per-line snapshots are taken (its own bands draw it)
    Engine *se = mirrorBand ? &sEngine[1] : e;   // the engine the bands draw
    sIrisSnap = gHostIrisRadius;
    for (int b = 0; b <= s2dNB; b++) sMtBnd[b] = b * SCREEN_H / s2dNB;
    int bi = 0;
    for (int line = 0; line < SCREEN_H; line++) {
        while (bi < s2dNB && line == sMtBnd[bi]) {
            Snap2d *s = &sMtW[bi].snap;
            s->refX[0] = se->refX[0]; s->refX[1] = se->refX[1]; s->refY[0] = se->refY[0]; s->refY[1] = se->refY[1];
            s->refLine = se->refLine; s->refWritten[0] = se->refWritten[0]; s->refWritten[1] = se->refWritten[1];
            s->winActive[0] = se->winActive[0]; s->winActive[1] = se->winActive[1];
            s->objMosaicLine = se->objMosaicLine; s->objMosaicCount = se->objMosaicCount;
            if (bSnap) {
                const Engine *b1 = &sEngine[1];
                Snap2d *t = &sMtWB[bi].snap;
                t->refX[0] = b1->refX[0]; t->refX[1] = b1->refX[1]; t->refY[0] = b1->refY[0]; t->refY[1] = b1->refY[1];
                t->refLine = b1->refLine; t->refWritten[0] = b1->refWritten[0]; t->refWritten[1] = b1->refWritten[1];
                t->winActive[0] = b1->winActive[0]; t->winActive[1] = b1->winActive[1];
                t->objMosaicLine = b1->objMosaicLine; t->objMosaicCount = b1->objMosaicCount;
            }
            bi++;
        }
        memcpy(s2dRegSnap[line], se->regs, REG_SIZE);   // capture registers for this line before hblank mutates them
        memcpy(s2dPalSnap[line], se->palette, 0x400);   // capture palette for this line (hblank sky/water gradients)
        s2dPalLine[line] = s2dPalSnap[line];   // (comparing with the line above first measured slower on the Vita)
        window_vertical(e, line);
        if (line == 0) { e->objMosaicLine = 0; e->objMosaicCount = 0; }  // matches obj-render's line-0 reset
        obj_mosaic_y(e, line);
        if (bSnap) {
            memcpy(s2dRegSnapB[line], sEngine[1].regs, REG_SIZE);
            memcpy(s2dPalSnapB[line], sEngine[1].palette, 0x400);
            if (skyBand) s2dRegSnapB[line][REG_MASTER_BRIGHT] = s2dRegSnapB[line][REG_MASTER_BRIGHT + 1] = 0;
        }
        if (whole) {   // this line's inputs against last frame's: redraw it when they differ
            static u32 palh;
            const u8 *pl = bSnap ? s2dPalSnapB[line] : s2dPalSnap[line];
            const u8 *pp = line ? (bSnap ? s2dPalSnapB[line - 1] : s2dPalSnap[line - 1]) : NULL;
            if (!pp || memcmp(pl, pp, 0x400)) {
                u32 g;
                palh = hash_words(0x27d4eb2du, 0, pl, 0x400, &g);
                palh = mix32(palh, g);
            }
            u32 ls = bline_sig(sEngine[1].regs, palh, skyBand && sSkyBg3Empty ? 8u : 0u);
            if (bAll || ls != sBLineSig[line]) sBDirty[line] = 1;
            sBLineSig[line] = ls;
        }
        if (mirrorBand || bSnap) {   // engine B's own per-line state, for the band snapshots above
            Engine *e1 = &sEngine[1];
            window_vertical(e1, line);
            if (line == 0) { e1->objMosaicLine = 0; e1->objMosaicCount = 0; }
            obj_mosaic_y(e1, line);
        }
        if (sLogoOn) {   // what the hblank IRQs leave in engine B's registers, line by line
            subLines = sub_regs_hash(&sEngine[1], 0, mix32(subLines, (u32)line));
        }
        if (sLogoOn && subDraw) {   // the sub screen's BG2 line (before this line's hblank, as the sequential loop)
            Engine *e1 = &sEngine[1];
            u32 *lp = sLogo + line * SCREEN_W;
            if (reg32(e1, REG_DISPCNT) & 0x400) {
                gpu2d_render_bg_line(e1, 2, line);
                int opaque = 0;
                for (int x = 0; x < SCREEN_W; x++) {
                    u16 v = e1->bg[2][x];
                    if (!(v & PIX_OPAQUE)) { lp[x] = 0; continue; }
                    opaque++;
                    u32 r = v & 31, g = v >> 5 & 31, b = v >> 10 & 31;
                    lp[x] = (r << 3 | r >> 2) | (g << 3 | g >> 2) << 8 | (b << 3 | b >> 2) << 16 | 0xff000000u;
                }
                if (opaque == SCREEN_W) sLogoFull = 1;
            } else
                memset(lp, 0, SCREEN_W * sizeof *lp);
            __atomic_store_n(&sLogoRows[line], 1, __ATOMIC_RELEASE);   // (after the row's pixels: the present may take it now)
        }
        if (in->hblank) in->hblank(line, in->hblankContext);
        for (int n = 0; n < 2; n++) ref_advance(&sEngine[n], line);
    }
    if (sLogoOn) {   // hblank-set registers changed while the frame start did not: redraw next frame
        sSubDirty = subLines != sSubLines;
        sSubLines = subLines;
    }
    if (!sMirrorOn) sMirrorReady = 0;
    int bLines = 0;
    if (whole) {
        uint64_t t0 = host_plat_time_ns();
        int vm = vram_mark(&sEngine[1], bSnap ? (const u8 (*)[REG_SIZE])s2dRegSnapB : (const u8 (*)[REG_SIZE])s2dRegSnap,
                           sBDirty);
        if (vm)
            memset(sBDirty, 1, sizeof sBDirty);
        if (skyBand) sky_bg3_check(&sEngine[1], bAll || vm);
        if (bAll) memset(sBDirty, 1, sizeof sBDirty);
        if (skyBand) {
            // The sky lines shown (from the one at the screen's top row, a line's margin above) that are not this
            // frame's yet - all of them this frame (none ever shows late: the sky moves
            // as the camera looks up, the clouds scroll every frame on the title) - and 4 lines a frame in turn
            // whatever changed, so no line can stay wrong (a change the line check misses - the scene's setup during
            // the fade in - is redrawn within 48 frames).
            static int refresh;
            const int top = (sSkyTop < 0 ? -sSkyTop / 2 : 0) - 1;
            for (int y = 0; y < SCREEN_H; y++)
                if (sBDirty[y]) sSkyValid[y] = 0;
            if (bAll) memset(sSkyValid, 0, sizeof sSkyValid);
            for (int k = 0; k < 4; k++) sSkyValid[(refresh + k) % SCREEN_H] = 0;
            refresh = (refresh + 4) % SCREEN_H;
            for (int y = 0; y < SCREEN_H; y++) {
                sBDirty[y] = y >= top && !sSkyValid[y];
                if (sBDirty[y]) sSkyValid[y] = 1;
            }
        }
        gSubCheckNs += host_plat_time_ns() - t0;
        for (int i = 0; i < SCREEN_H; i++) bLines += sBDirty[i];
        sBValid = 1;
        gSubDraws += bLines != 0;
    }
    if ((sLogoOn && subDraw) || bLines) sLogoGen++;   // (the title logo's lines above, the bands' dirty lines)
    if (mirrorBand) sMirrorReady = 1;   // (the bands fill sLogo before this frame's present, on the render thread)
    if (!sOverlayOn) sOvlReady = 0;
    if (overlayBand) sOvlReady = 1;
    sSkyReady = skyBand || sSkyHold;   // (held: the last sky stays - sky_hold)
    sSkyEdges = skyBand;   // (the edge rows: after the bands, render_engine0_mt_bands)
    sSkyTopSnap = sSkyTop;
    sLogoReady = sLogoOn && !sLogoFull;
    sPromptSnap = gHostTitlePrompt;
    sClockSnap = gHostClockDrawn;
    sClockXSnap = gHostClockX;
    sClockYSnap = gHostClockY;
    gHostClockDrawn = 0;
    // The logo with its copyright line, centred: its finished picture spans x 35..224 (the sign 35..215 plus the
    // (R) mark; the year 72..184, centred on the DS screen's middle) and y 10..188. Fixed offsets, not the box of
    // what is drawn so far - the logo is uncovered tile by tile, and a box that grows would move it every frame.
    // Horizontally it is already centred (the year's middle is the screen's); vertically it goes up 6 lines (above
    // the middle, as asked: 4 lines free above it, 9 below).
    sLogoDx = 0;
    sLogoDy = -6;
    for (int b = 0; b < s2dNB; b++) {
        sMtW[b].in = in; sMtW[b].src = se; sMtW[b].be = &s2dBandEng[b]; sMtW[b].lo = sMtBnd[b]; sMtW[b].hi = sMtBnd[b + 1];
        sMtW[b].mirror = mirrorBand;
        sMtW[b].dirty = sBDirty;
        sMtW[b].regSnap = NULL, sMtW[b].palSnap = NULL;
        memcpy(&s2dBandEng[b], se, offsetof(Engine, out));   // inputs only, not the big output arrays
        sMtWB[b].in = in; sMtWB[b].src = &sEngine[1]; sMtWB[b].be = &s2dBandEngB[b];
        // (every band walks all of engine B's lines from line 0 and draws its share of the changed ones: a menu's
        // changed lines are often all in one half of the screen, which left one band with all the work)
        sMtWB[b].lo = 0; sMtWB[b].hi = bLines && bSnap ? SCREEN_H : 0;
        sMtWB[b].part = b; sMtWB[b].nparts = s2dNB;
        if (b) sMtWB[b].snap = sMtWB[0].snap;
        sMtWB[b].canvas = NULL; sMtWB[b].mirror = skyBand ? 3 : 2; sMtWB[b].dirty = sBDirty;
        sMtWB[b].regSnap = s2dRegSnapB, sMtWB[b].palSnap = s2dPalSnapB;
        if (bSnap && bLines) memcpy(&s2dBandEngB[b], &sEngine[1], offsetof(Engine, out));
    }
}

// Band helpers: persistent threads (created once, then woken per frame) - creating and joining threads every
// frame is expensive on the Vita. The calling thread renders band 0 itself.
static pthread_mutex_t s2dMu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t s2dGo = PTHREAD_COND_INITIALIZER, s2dDoneCv = PTHREAD_COND_INITIALIZER;
static unsigned s2dGen;
static int s2dPending, s2dHelpers;
static void *band2d_helper(void *arg) {
    int b = (int)(intptr_t)arg;
#if HOST_VITA
    extern int sceKernelChangeThreadCpuAffinityMask(int, int);
    sceKernelChangeThreadCpuAffinityMask(0, 0x60000);   // cores 1+2: core 0 belongs to the game thread
#endif
    unsigned seen = 0;
    for (;;) {
        pthread_mutex_lock(&s2dMu);
        while (s2dGen == seen) pthread_cond_wait(&s2dGo, &s2dMu);
        seen = s2dGen;
        int run = b < s2dNB;
        pthread_mutex_unlock(&s2dMu);
        if (run) band2d_worker(&sMtW[b]);
        pthread_mutex_lock(&s2dMu);
        if (--s2dPending == 0) pthread_cond_signal(&s2dDoneCv);
        pthread_mutex_unlock(&s2dMu);
    }
    return 0;
}
// (the sky) After the bands: the texture's edge rows - the plain colours of lines 0 (above the sky) and 191 (below it)
// - when they change.
static u32 sky_rgba(u32 c6) {
    u32 c = argb(c6);
    return (c >> 16 & 0xff) | (c & 0xff00) | (c & 0xff) << 16 | 0xff000000u;
}
static void sky_edges(void) {
    static u32 last[2] = {1, 1};
    for (int k = 0; k < 2; k++) {
        const u32 c = sky_rgba(rgb6(pal16(s2dPalSnapB[k ? SCREEN_H - 1 : 0], 0)));
        if (c == last[k]) continue;
        last[k] = c;
        const int row = k ? SKY_TH - 1 : 0;
        for (int x = 0; x < SKY_TW; x++) sSkyTex[row * SKY_TW + x] = c;
        __atomic_store_n(&sSkyRows[row], 1, __ATOMIC_RELEASE);
    }
}
static void render_engine0_mt_bands(u32 *canvas) {
    Engine *e = &sEngine[0];
    int nb = s2dNB;
    for (int b = 0; b < nb; b++) sMtW[b].canvas = canvas;
    while (s2dHelpers < nb - 1) {
        pthread_t t;
        pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_setstacksize(&at, 256 << 10);
        if (pthread_create(&t, &at, band2d_helper, (void *)(intptr_t)(s2dHelpers + 1)) != 0) break;
        s2dHelpers++;
    }
    pthread_mutex_lock(&s2dMu);
    s2dPending = s2dHelpers;
    s2dGen++;
    pthread_cond_broadcast(&s2dGo);
    pthread_mutex_unlock(&s2dMu);
    band2d_worker(&sMtW[0]);
    for (int b = s2dHelpers + 1; b < nb; b++) band2d_worker(&sMtW[b]);   // helpers that failed to start
    pthread_mutex_lock(&s2dMu);
    while (s2dPending) pthread_cond_wait(&s2dDoneCv, &s2dMu);
    pthread_mutex_unlock(&s2dMu);
    if (sSkyEdges) { sSkyEdges = 0; sky_edges(); }
    for (int b = 0; b < nb && !canvas && !sGcFrame; b++) {
        int lo = sMtBnd[b], n = sMtBnd[b + 1] - lo;
        memcpy(e->out + lo * SCREEN_W, s2dBandEng[b].out + lo * SCREEN_W, (size_t)n * SCREEN_W * sizeof(u32));
        if (sDefer3d) memcpy(e->defer + lo * SCREEN_W, s2dBandEng[b].defer + lo * SCREEN_W, (size_t)n * SCREEN_W * sizeof(u32));
    }
}

static int sDeferBands;   // render thread active: prepass on main, bands+canvas tail on the worker
void gpu2d_set_defer_bands(int on) {
    static int init;
    if (!init) {   // fixed for the session: the prepass and the bands must agree on the band split
        init = 1;
#if HOST_VITA
        if (on) s2dNB = 2;   // render thread + one helper: cores 1+2 (core 0 is the game thread's)
#endif
    }
    sDeferBands = on;
}

// Host test hook for the GPU compose (tools: scratch gc test): ACWW_GCTEST=<dir> dumps frames listed in
// ACWW_GCFRAMES (comma separated) - the CPU canvas, or with ACWW_GCMODE=1 the GPU-compose layers + 3D lines.
static int gc_test_want(void) {
#if HOST_VITA
    return 0;
#else
    const char *dir = getenv("ACWW_GCTEST"), *fl = getenv("ACWW_GCFRAMES");
    if (!dir || !fl) return 0;
    char key[24];
    snprintf(key, sizeof key, "%u", sFrame);
    for (const char *q = fl; (q = strstr(q, key)); q++)
        if ((q == fl || q[-1] == ',') && (q[strlen(key)] == ',' || q[strlen(key)] == 0)) return 1;
    return 0;
#endif
}
static void gc_test_write(const char *kind, const void *a, size_t na, const void *b, size_t nb) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s_%u.bin", getenv("ACWW_GCTEST"), kind, sFrame);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(a, 1, na, f);
    if (b) fwrite(b, 1, nb, f);
    fclose(f);
}
static void gc_test_dump(void) {
    if (!gc_test_want()) return;
    const u32 *l3 = sMtW[0].in ? sMtW[0].in->lines3d : NULL;
    static u32 zero[SCREEN_W * SCREEN_H];
    char path[512];
    snprintf(path, sizeof path, "%s/gc_%u.bin", getenv("ACWW_GCTEST"), sFrame);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    for (int n = 0; n < 4; n++) fwrite(sGc.bg[n], 2, SCREEN_W * SCREEN_H, f);
    fwrite(sGc.obj, 2, SCREEN_W * SCREEN_H, f);
    fwrite(sGc.aux, 2, SCREEN_W * SCREEN_H, f);
    fwrite(sGc.par, GPU2D_GC_PAR * 4, SCREEN_H, f);
    fwrite(&sGc.used, 4, 1, f);
    fwrite(l3 ? l3 : zero, 1, sizeof zero, f);
    fclose(f);
    snprintf(path, sizeof path, "%s/logo_%u.bin", getenv("ACWW_GCTEST"), sFrame);   // (the top-screen layer: the sky)
    if ((f = fopen(path, "wb"))) {
        int m = sMirrorReady ? 1 : sOvlReady ? 2 : 0, k[4] = {sSkyReady, sSkyTopSnap, sLogoReady, sLogoDy};
        fwrite(&m, 4, 1, f);
        fwrite(sLogo, 4, SCREEN_W * SCREEN_H, f);
        fwrite(k, 4, 4, f);   // (the sky: shown, its offset; the title logo: shown, its move up)
        fwrite(sSkyTex, 4, SKY_TW * SKY_TH, f);
        fclose(f);
    }
}
// Validation hook: hash the finished engine-0 canvas so a threaded run can be byte-compared with a sync run.
static void canvas_hash(void) {
    if (gc_test_want()) gc_test_write("cpu", sCanvas, SCREEN_W * SCREEN_H * 4, NULL, 0);
    static int on = -1;
    if (on < 0) on = getenv("ACWW_CANVHASH") ? 1 : 0;
    if (!on) return;
    unsigned long long h = 1469598103934665603ull;
    for (int i = 0; i < SCREEN_W * SCREEN_H; i++) { h ^= sCanvas[i]; h *= 1099511628211ull; }
    fprintf(stderr, "canvhash frame %llu = %016llx\n", (unsigned long long)sFrame, h);
}

// Worker half: run the deferred bands and finish engine-0's canvas (mirrors the mainOnly tail below). No game code.
void gpu2d_render_finish(void) {
    if (!sMtPending) return;
    sMtPending = 0;
    sGcFrame = sGcOn && !sDefer3d && gc_frame_ok();
    sGcFrameDone = 0;
    if (sGcFrame) {
        sGcCur = &sGcSets[sGcCur == &sGcSets[0]];   // the other set: the GPU may still read the last one
        gc_buffers_default(sGcCur, sGcCur == &sGcSets[1]);
        sGc.used = 0;
        render_engine0_mt_bands(NULL);
        sGcFrame = 0;
        sGcFrameDone = 1;
        sCanvasRows = SCREEN_H;
        sFrame++;
        gc_test_dump();
        return;
    }
    if (sDefer3d) {
        render_engine0_mt_bands(NULL);
        for (int i = 0; i < SCREEN_W * SCREEN_H; i++) sCanvas[i] = argb(sEngine[0].out[i]);
    } else
        render_engine0_mt_bands(sCanvas);   // (engine 0's own out[] is then not filled: only the canvas is shown)
    sCanvasRows = SCREEN_H;
    sFrame++;
    canvas_hash();
}

// Returns 1 if it rendered engine 0's lines multi-threaded; 0 to fall back to the sequential loop.
static int render_engine0_mt(const Gpu2dFrameInput *in) {
    render_engine0_mt_prepass(in);
    if (sDeferBands) { sMtPending = 1; return 1; }   // worker will run the bands + tail via gpu2d_render_finish()
    render_engine0_mt_bands(NULL);
    return 1;
}

// The top screen shows the town's sky (the game's sky on engine B, displayed normally, the screens not swapped).
unsigned gSkyStat[8];   // (render.c's log, per second) frames: sky shown; not: no sky on top, indoors, B off, swapped, B blank/mode, title/boot/menu
static int sky_shown(void) {
    extern int Sky_HostOnSub(void);
    extern int Scene_HostIsOutdoor(void);
    const u32 d = reg32(&sEngine[1], REG_DISPCNT);
    gSkyStat[1] += !Sky_HostOnSub();
    gSkyStat[3] += !(sPowCnt & 0x200);
    gSkyStat[4] += (sPowCnt & 0x8000) != 0;
    gSkyStat[5] += (d & 0x80) || (d >> 16 & 3) != 1;
    // (not POWCNT1's screen swap: the game leaves it set in town after some scene changes, with no menu open; a menu
    // that takes the top screen moves the sky off it first - Sky_SetEngine(1) - so Sky_HostOnSub says it)
    // (the game's sky task, not the scene number: on the Vita Scene_GetCurrent stays at the title's 0x2c in town after
    // a game is continued from the title, and Scene_HostIsOutdoor with it)
    // (the title's fly-over has the town's sky too, its logo over it: both shown)
    extern int MenuCtrl_IsIdle(void);
    extern int MenuCtrl_IsMenuOpen(void);
    // (a menu on its way in, or open, with the screens swapped: the menu overlay's)
    const int menu = (sPowCnt & 0x8000) && (!MenuCtrl_IsIdle() || MenuCtrl_IsMenuOpen());
    return Sky_HostOnSub() && !menu && (sPowCnt & 0x200) && !(d & 0x80) && (d >> 16 & 3) == 1;
}
// The town's horizon in the picture, in the DS's lines from the top (0 .. 192; the Vita's 3D is the DS's picture
// taller and wider, the same lines): where the ground at height 0 - the curved world's cylinder - meets the sky
// as the camera sees it. The ray from the eye that just touches the cylinder ahead of it (WorldCurve_GetHorizonAngle's
// point), in the camera's view, in the middle column; projected as the DS's perspective does (fovy).
#include <math.h>
static int sky_horizon_row(float *row) {
    extern int Camera_HostSkyView(s32 *o);
    s32 v[11];
    if (!Camera_HostSkyView(v)) return 0;
    const float k = 1.0f / 4096.0f, ex = v[0] * k, ey = v[1] * k, ez = v[2] * k, R = v[10] * k;
    float fx = v[3] * k - ex, fy = v[4] * k - ey, fz = v[5] * k - ez, ux = v[6] * k, uy = v[7] * k, uz = v[8] * k;
    float l = sqrtf(fx * fx + fy * fy + fz * fz);
    if (l < 1e-6f) return 0;
    fx /= l, fy /= l, fz /= l;
    const float ud = ux * fx + uy * fy + uz * fz;   // (up: square to the view direction)
    ux -= ud * fx, uy -= ud * fy, uz -= ud * fz;
    l = sqrtf(ux * ux + uy * uy + uz * uz);
    if (l < 1e-6f) return 0;
    ux /= l, uy /= l, uz /= l;
    const float d = sqrtf(ey * ey + ez * ez);
    if (d <= R * 1.0001f) return 0;
    const float th = atan2f(ez, ey) - acosf(R / d);
    const float py = R * cosf(th) - ey, pz = R * sinf(th) - ez;
    const float fyz = sqrtf(fy * fy + fz * fz), dyz = sqrtf(py * py + pz * pz);
    const float px = fyz > 1e-6f ? fx * dyz / fyz : 0.0f;
    const float a = px * fx + py * fy + pz * fz, b = px * ux + py * uy + pz * uz;
    const float half = (float)v[9] * 3.14159265f / 65536.0f;
    if (a <= 1e-6f || half <= 0.0f) return 0;
    *row = 96.0f - 96.0f * (b / a) / tanf(half);
    return 1;
}
// (the one-screen sky's town camera: City Folk's - Camera::onExecute) the Vita; the PC host's test of the Vita's sky
int Host_SkyCamera(void) {
#if HOST_VITA
    return 1;
#else
    static int on = -1;
    if (on < 0)
        on = getenv("ACWW_SKY") && atoi(getenv("ACWW_SKY")) && getenv("ACWW_FORCEMAIN") && atoi(getenv("ACWW_FORCEMAIN"));
    return on;
#endif
}
// The sky stays - the last one drawn, on the camera's horizon - while the top screen can't draw it: a menu (the
// inventory: the game moves its sky off the top screen, which becomes the menu's), the top screen blanked or off in a
// fade, as long as the town's sky task is up (Sky_HostOutdoorAlive: not after leaving the town).
static void sky_hold(int sky) {
    extern int Sky_HostOutdoorAlive(void);
    sSkyHold = !sky && sSkyWas && Sky_HostOutdoorAlive();
    sSkyWas = sky || sSkyHold;
    if (sSkyHold) sSkyBack = 1;
}
// (render.c, before the frame's VRAM snapshot) the sky will be drawn this frame: keep engine B's banks
int gpu2d_sky_wanted(void) { return sky_shown(); }
void gpu2d_render_frame(const Gpu2dFrameInput *in) {
    {   // (host check) both engines' finished pictures of the previous frame
        static int on = -1;
        if (on < 0) on = getenv("ACWW_OUTHASH") ? 1 : 0;
        if (on) {
            unsigned long long h = 1469598103934665603ull;
            for (int n = 0; n < 2; n++)
                for (int i = 0; i < SCREEN_W * SCREEN_H; i++) { h ^= sEngine[n].out[i]; h *= 1099511628211ull; }
            fprintf(stderr, "outhash frame %llu = %016llx\n", (unsigned long long)sFrame, h);
        }
    }
    sGpu = in->gpu;
    sDefer3d = in->defer3d || sGpu;
    const int off = 0;
#if HOST_VITA
    const int mainOnly = 1;  // the Vita shows only the main screen: never render the top screen (engine 1)
    {   // ... except its logo on the title screen (render_engine0_mt_prepass)
        extern unsigned Scene_GetCurrent(void);
        unsigned sc = Scene_GetCurrent();
        // (the town's sky task up: not the title or the boot screens, whatever the scene number - it can stay at the
        // title's after a game is continued)
        // The title: its logo is the top screen's BG2 (the town's sky never shows BG2: stars BG1, clouds BG3).
        const int bg2 = (reg32(&sEngine[1], REG_DISPCNT) & 0x400) != 0;
        sLogoOn = sc == 0x2c && bg2;
        const int sky = sky_shown();   // (with the title's logo too)
        sky_hold(sky);
        // The boot screens: their Nintendo logo is on the top screen; the Vita shows engine B's whole picture.
        sMirrorOn = sc == 0x3f && !sky;
        // The menus (the game puts engine A on the top screen, POWCNT1 bit 15, and draws the menu with engine B):
        // one screen, as the iPhone build - the scene keeps being drawn and the menu goes over it (sOverlayOn).
        sSkyOn = sky;
        sOverlayOn = !sMirrorOn && !sSkyOn && (sPowCnt & 0x8000);
        gSkyStat[0] += sSkyOn;
        gSkyStat[6] += sLogoOn || sMirrorOn || sOverlayOn;
        gSkyStat[7]++;
    }
#else
    static int forceMain = -1;
    if (forceMain < 0) forceMain = getenv("ACWW_FORCEMAIN") ? atoi(getenv("ACWW_FORCEMAIN")) : 0;
    const int mainOnly = forceMain;   // host validation of the Vita engine-0 render-thread path
    {   // (host check/profiling of the Vita's menu overlay: engine B's whole picture over the scene)
        static int ovl = -1;
        if (ovl < 0) ovl = getenv("ACWW_OVERLAY") ? atoi(getenv("ACWW_OVERLAY")) : 0;
        if (ovl) sOverlayOn = ovl == 2 || (sPowCnt & 0x8000) != 0;   // (2: always - a test of the overlay path)
        static int sky = -1;   // (host check of the Vita's sky, with ACWW_FORCEMAIN)
        if (sky < 0) sky = getenv("ACWW_SKY") ? atoi(getenv("ACWW_SKY")) : 0;
        sSkyOn = sky && forceMain && !sOverlayOn && gpu2d_sky_wanted();
        if (sky && forceMain) sky_hold(sSkyOn);
        if (sky && forceMain) {   // (and the title's logo, as the Vita: its harness check)
            extern unsigned Scene_GetCurrent(void);
            sLogoOn = Scene_GetCurrent() == 0x2c && (reg32(&sEngine[1], REG_DISPCNT) & 0x400) != 0;
        }
    }
#endif
    if (sSkyOn || sSkyHold) {   // line 0's top screen row: the sky's bottom (2 rows a line) SKY_DOWN_PX below the horizon
        float h;
        if (sky_horizon_row(&h)) {
            const int hz = (int)floorf(h * SKY_SCREEN_H / SCREEN_H + 0.5f);
            const int t = hz + SKY_DOWN_PX - 2 * SCREEN_H;
            sSkyTop = t < -4 * SCREEN_H ? -4 * SCREEN_H : t > 2 * SKY_SCREEN_H ? 2 * SKY_SCREEN_H : t;
        }
        static int log = -1;   // (host check: ACWW_SKYLOG prints the horizon every frame)
        if (log < 0) log = getenv("ACWW_SKYLOG") != 0;
        if (log) {
            s32 v[11];
            extern int Camera_HostSkyView(s32 *o);
            int got = Camera_HostSkyView(v);
            fprintf(stderr, "sky frame %u hold %d menu %d top %d horizon %.2f eye %d %d %d target %d %d %d up %d %d %d fovy %d\n", sFrame,
                    sSkyHold, (sPowCnt & 0x8000) != 0, sSkyTop, got && sky_horizon_row(&h) ? h : -999.0f, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                    v[8], v[9]);
        }
    }
    {
        static int hold, swapped;
        int now = !off && !sGpu && !sDefer3d && in->lines3d && gpu2d_tall_3d_ok() && gpu2d_main_on_bottom();
        swapped = (sPowCnt & 0x8000) ? swapped + 1 : 0;
        if (now) hold = 120;
        else if (hold && swapped > 8) hold = 0;
        else if (hold) hold--;
        sSingle = !mainOnly && (now || (hold && !off && !sGpu && !sDefer3d));
        sSingleBlank = sSingle && !now;
    }
    gHostSingleScreen = sSingle;
    {
        extern int Scene_HostIsOutdoor(void);
        sOutdoors = Scene_HostIsOutdoor();
    }
    sA3dPrev = sA3dNow;
    sA3dNow = 0;
    map_vram(in);
    if (sGpu) {
        gpu_tables(in);
        sGpuFrame.palettes = sPalettes[0];
        sGpuFrame.oams = sOams[0];
        memcpy(sPalettes[0], in->palette, 0x800);
        memcpy(sOams[0], in->oam, 0x800);
        sGpuFrame.paletteCount = sGpuFrame.oamCount = 1;
        sGpuFrame.tables[0][G2T_SCREEN_A] = 0;
    }
    for (int n = 0; n < 2; n++) {
        Engine *e = &sEngine[n];
        e->palette = in->palette + 0x400 * n;
        e->oam = in->oam + 0x400 * n;
        for (int i = 0; i < 2; i++) {
            e->refX[i] = ref28(e, 0x28 + 0x10 * (u32)i);
            e->refY[i] = ref28(e, 0x2c + 0x10 * (u32)i);
            e->refWritten[i] = 0;
        }
        e->refLine = 0;
    }
    sExtRows = 0;
    if (sSingle && !sSingleBlank && !sGpu && !sDefer3d) render_ext_sky(in);
    for (int n = 0; n < (mainOnly && !sMirrorOn && !sOverlayOn && !sSkyOn ? 1 : 2); n++) {
        Engine *e = &sEngine[n];
        for (int line = SCREEN_H + 1; line < FRAME_LINES; line++) window_vertical(e, line);
        gpu2d_render_obj_line(e, 0, &e->objNextLine0);
        e->haveNextLine0 = 1;
        if (sGpu)
            for (int x = 0; x < SCREEN_W; x++) sGpuFrame.obj0[n][x] = pack_obj(&e->objNextLine0, x);
    }
    if (mainOnly && !sGpu && !sDiagnostics && render_engine0_mt(in)) {
        if (sMtPending) {
            // Deferred: the render worker runs the bands + canvas tail via gpu2d_render_finish(). The end-of-frame
            // window update stays here on the game thread (the worker must not touch the live engines, whose
            // registers the game rewrites while the worker runs).
            for (int n = 0; n < 2; n++) window_vertical(&sEngine[n], SCREEN_H);
            return;
        }
        goto after_lines;
    }
    for (int line = 0; line < SCREEN_H; line++) {
        for (int n = 0; n < (mainOnly ? 1 : 2); n++) {
            Engine *e = &sEngine[n];
            window_vertical(e, line);
            if (sGpu) gpu_line(n, line);
            if (sGpu == 1) {}
            else if (line == 0 && e->haveNextLine0) e->obj = e->objNextLine0;
            else if (line == 0) memset(&e->obj, 0, sizeof e->obj);
            else gpu2d_render_obj_line(e, line, &e->obj);
            if (n == 1 && (sPowCnt & 0x8000) && !gHostMenuPatternShown) {
                u8 keep = e->regs[REG_DISPCNT + 1];
                e->regs[REG_DISPCNT + 1] &= (u8)~0x04;
                render_line(e, in, line);
                e->regs[REG_DISPCNT + 1] = keep;
            } else
                render_line(e, in, line);
        }
        if (in->hblank) in->hblank(line, in->hblankContext);
        if (sGpu) gpu_versions(in);
        for (int n = 0; n < 2; n++) {
            Engine *e = &sEngine[n];
            for (int i = 0; i < 2; i++) {
                u32 r = REG_BG2PA + 0x10 * (u32)i;
                e->refX[i] = e->refWritten[i] & 1 ? ref28(e, 0x28 + 0x10 * (u32)i)
                                                  : e->refX[i] + (s16)reg16(e, r + 2) * (line + 1 - e->refLine);
                e->refY[i] = e->refWritten[i] & 2 ? ref28(e, 0x2c + 0x10 * (u32)i)
                                                  : e->refY[i] + (s16)reg16(e, r + 6) * (line + 1 - e->refLine);
                e->refWritten[i] = 0;
            }
            e->refLine = line + 1;
        }
    }
after_lines:
    for (int n = 0; n < 2; n++) {
        Engine *e = &sEngine[n];
        window_vertical(e, SCREEN_H);
    }
    if (sGpu) sGpuFrame.tables[0][G2T_SCREEN_A] = (sPowCnt & 0x8000) ? 0 : 1;
    sCanvasRows = 0;
    if (mainOnly) {
        // the main screen only (engine 0, with the 3D already composited into its output): one 256x192 canvas,
        // no top screen rendered, no sky borrowed from it.
        for (int i = 0; i < SCREEN_W * SCREEN_H; i++) sCanvas[i] = argb(sEngine[0].out[i]);
        sCanvasRows = SCREEN_H;
        sFrame++;
        canvas_hash();
        return;
    }
    if (sSingle) single_compose(in, sSingleBlank);
    else if (!sGpu && !sDefer3d) classic_compose(in);
    for (int n = 0; n < 2; n++) {
        int screen = (sPowCnt & 0x8000) ? n : 1 - n;
        if (sGpu == 1) {
            memcpy(sDeferredLines[screen], sEngine[n].lineInfo, sizeof sEngine[n].lineInfo);
            continue;
        }
        if (sDefer3d) {
            const Engine *e = &sEngine[n];
            u32 *d = sDeferred[screen];
            for (int i = 0; i < SCREEN_W * SCREEN_H; i++) d[2 * i] = pack18(e->out[i]), d[2 * i + 1] = e->defer[i];
            memcpy(sDeferredLines[screen], e->lineInfo, sizeof e->lineInfo);
            continue;
        }
        u32 *s = sScreen[screen];
        const u32 *src = sEngine[n].out;
        for (int i = 0; i < SCREEN_W * SCREEN_H; i++) {
            u32 c = src[i], r = c & 0x3f, g = c >> 8 & 0x3f, b = c >> 16 & 0x3f;
            r = r << 2 | r >> 4; g = g << 2 | g >> 4; b = b << 2 | b >> 4;
            s[i] = 0xff000000u | r << 16 | g << 8 | b;
        }
    }
    sFrame++;
}

const uint32_t *gpu2d_screen(int screen) { return sScreen[screen & 1]; }
const uint32_t *gpu2d_deferred(int screen) { return sDeferred[screen & 1]; }
const uint32_t *gpu2d_deferred_lines(int screen) { return sDeferredLines[screen & 1]; }

void gpu2d_describe_pixel(int screen, int x, int y, char *buf, size_t size) {
    int n = (sPowCnt & 0x8000) ? screen : 1 - screen;
    const Engine *e = &sEngine[n];
    static const char *const kLayer[8] = {"BG0", "BG1", "BG2", "BG3", "OBJ", "backdrop", "3D", "none"};
    static const char *const kEffect[7] = {"none", "alpha", "brightness up", "brightness down",
                                           "semi-transparent OBJ", "bitmap OBJ alpha", "3D alpha"};
    int at = y * SCREEN_W + x;
    char top[16];
    if (e->dbgTop[at] == LAYER_OBJ) snprintf(top, sizeof top, "OBJ#%u", e->dbgObj[at]);
    else snprintf(top, sizeof top, "%s", kLayer[e->dbgTop[at] & 7]);
    u32 dispcnt = reg32(e, REG_DISPCNT);
    snprintf(buf, size, "engine %c (mode %u, dispcnt %08x, bldcnt %04x %04x %04x, master %04x): %s %06x (alpha %u) "
             "over %s %06x, effect %s", 'A' + n, dispcnt & 7, dispcnt, reg16(e, REG_BLDCNT), reg16(e, REG_BLDALPHA),
             reg16(e, REG_BLDY), reg16(e, REG_MASTER_BRIGHT), top, e->dbgTopColor[at], e->dbgAlpha[at],
             kLayer[e->dbgBelow[at] & 7], e->dbgBelowColor[at], kEffect[e->dbgEffect[at] % 7]);
    size_t used = strlen(buf);
    if (e->dbgTop[at] == LAYER_OBJ && e->oam) {
        const u8 *o = e->oam + e->dbgObj[at] * 8;
        used += (size_t)snprintf(buf + used, size - used, "; OBJ attributes %04x %04x %04x", o[0] | o[1] << 8,
                                 o[2] | o[3] << 8, o[4] | o[5] << 8);
        if (o[1] & 1) {
            const u8 *g = e->oam + ((o[2] | o[3] << 8) >> 9 & 31) * 32;
            used += (size_t)snprintf(buf + used, size - used, " (matrix %04x %04x %04x %04x)", g[6] | g[7] << 8,
                                     g[14] | g[15] << 8, g[22] | g[23] << 8, g[30] | g[31] << 8);
        }
    }
    used += (size_t)snprintf(buf + used, size - used, "; bgcnt %04x %04x %04x %04x", reg16(e, REG_BGCNT),
                             reg16(e, REG_BGCNT + 2), reg16(e, REG_BGCNT + 4), reg16(e, REG_BGCNT + 6));
    if (dispcnt & 0xe000)
        snprintf(buf + used, size - used, "; windows %04x %04x / %04x %04x in %04x out %04x", reg16(e, REG_WIN0H),
                 reg16(e, REG_WIN0V), reg16(e, REG_WIN1H), reg16(e, REG_WIN1V), reg16(e, REG_WININ),
                 reg16(e, REG_WINOUT));
}
