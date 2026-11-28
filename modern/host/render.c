
#include "host_features.h"
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "gpu2d/gpu2d.h"
#include "gpu3d/gpu3d.h"
#include "host_mem.h"
#include "plat.h"
#include "render.h"
#include "wgpu_display.h"

void melon_init(void);
void melon_write8(u32 addr, u8 v);
void melon_write16(u32 addr, u16 v);
void melon_write32(u32 addr, u32 v);
u32 melon_read32(u32 addr);
void melon_frame(void);
const u32 *melon_screen(int screen);
int melon_gx_fifo_irq_level(void);
void melon_debug(void);
void melon_3d_lines(u32 *out);
int melon_3d_poly_count(void);
int melon_3d_poly_get(int i, Gpu3dVertexOut *v, u32 *attr, u32 *texparam, u32 *texpal, int *flags);
void host_hblank(int line);
u8 *host_vram_bank(int i);
void host_vram_flush(void);
u32 host_vram_total(void);
u32 host_vram_bank_offset(int i);
void host_vram_snapshot(u8 *dst);
void host_vram_snapshot_live(u8 *dst);
extern unsigned char gHostPltt[0x800];
extern unsigned char gHostOam[0x800];
extern void frontend_frame(void);

// ---- render thread: run the 2D bands + 3D rasterize + present on a worker, overlapping the next frame's CPU ----
// The main thread, at the frame boundary, runs all game code (the 2D hblank-IRQ prepass and the 3D geometry
// commit) and snapshots VRAM/palette/OAM; the worker then rasterizes from that stable snapshot while the main
// thread races ahead into the next frame. This hides the ~38ms render under the ~68ms CPU (frame ~108 -> ~75ms).
static int sRenderThreadOn;
static unsigned sRenderSkipped;
static int sWorkerSpawned;
static int sLastMt;                       // the previous render_frame() dispatched to the worker (present is the worker's)
static pthread_t sWorker;
static pthread_mutex_t sWMx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t sWCvWork = PTHREAD_COND_INITIALIZER, sWCvDone = PTHREAD_COND_INITIALIZER;
static int sWReady, sWDone = 1;           // handshake: work posted / work finished
static u8 *sSnapBankMem;                  // VRAM snapshot (host_vram_total() bytes)
static u8 sSnapPal[0x800], sSnapOam[0x800];
static const u8 *sSnapBank[9];
static int sDraw3d = 1;   // the render thread rasterizes the committed geometry this frame (pipelined: only new ones)
static Gpu2dFrameInput sWorkIn;

static int sOwn;
static int sOwn3d;
static int sGpu2d;
static int sGpu3d;
static int sMelon;
static int sCompare;
static int sGpuDisplay;
static int sScale3d = 2;
static int sGpu2dMode;
static u64 sGpu2dChecked, sGpu2dDiffering;
static int sSelftest;
static long sCompareMax = 200, sCompareLines;
static u64 sCompared, sDiffering, sFirstDiffering = ~0ull;
static u64 sReads3d, sReadsDiffering, sFirstReadDiffering = ~0ull;
static long sReadLines;
static u64 sPolyFrames, sPolyDiffering, sFirstPolyDiffering = ~0ull;
static long sPolyLines;
static u64 sPix3dFrames, sPix3dDiffering, sFirstPix3dDiffering = ~0ull;
static long sPix3dLines;
static u32 s3d[256 * 192];

static int env_is(const char *name, const char *value) {
    const char *r = getenv(name);
    return r && strcmp(r, value) == 0;
}

void render_init(void) {
    sOwn = !env_is("ACWW_RENDERER", "melon");
    if (getenv("ACWW_RENDERER") && sOwn && !env_is("ACWW_RENDERER", "own"))
        fprintf(stderr, "render: ACWW_RENDERER=%s unknown (own, melon): own\n", getenv("ACWW_RENDERER"));
    sOwn3d = getenv("ACWW_RENDERER3D") ? !env_is("ACWW_RENDERER3D", "melon") : sOwn;
    if (getenv("ACWW_RENDERER3D") && sOwn3d && !env_is("ACWW_RENDERER3D", "own") && !env_is("ACWW_RENDERER3D", "wgpu"))
        fprintf(stderr, "render: ACWW_RENDERER3D=%s unknown (own, wgpu, melon): own\n", getenv("ACWW_RENDERER3D"));
    if (sOwn3d) sOwn = 1;
    sCompare = getenv("ACWW_RENDER_COMPARE") ? atoi(getenv("ACWW_RENDER_COMPARE")) : 0;
    int asked = env_is("ACWW_RENDERER3D", "wgpu");
    int wgpu = asked || (!getenv("ACWW_RENDERER3D") && sOwn3d && !sCompare && !getenv("ACWW_RENDER_SELFTEST") &&
                         !getenv("ACWW_RENDER_SELFTEST3D"));
    if (wgpu) {
        int readback = sCompare || getenv("ACWW_RENDER_SELFTEST") || getenv("ACWW_RENDER_SELFTEST3D") ||
                       env_is("ACWW_WGPU_READBACK", "1");
        if (getenv("ACWW_SCALE3D")) sScale3d = atoi(getenv("ACWW_SCALE3D"));
        sScale3d = readback || sScale3d < 1 ? 1 : sScale3d > 8 ? 8 : sScale3d;
        if (wdisp_init(asked) && gpu3d_use_wgpu(wdisp_device(), wdisp_queue(), sScale3d, readback)) sGpuDisplay = !readback;
        else if (asked) fprintf(stderr, "render: ACWW_RENDERER3D=wgpu: no WebGPU in this build or on this machine: own\n");
        if (sGpuDisplay && !env_is("ACWW_GPU2D", "0")) sGpu2dMode = env_is("ACWW_GPU2D_CHECK", "1") ? 2 : 1;
        if (sGpuDisplay && getenv("ACWW_GPU2D_SELFTEST")) sGpu2dMode = 2;
    }
#if HOST_VITA
    // Vita GPU 3D rasterizer: when the vitaGL GPU path is on and the shader compiler is reachable, route the DS
    // 3D through the g3w_* seam (vita_gl3d.c) in readback mode - the GPU rasterizes, the software 2D compositor
    // still composites. Falls back to the software rasterizer otherwise (no crash).
    extern int vita_gl3d_available(void);
    if (sOwn3d && !sCompare && !sGpuDisplay && vita_gl3d_available()) {
        if (gpu3d_use_wgpu(0, 0, 1, 1)) fprintf(stderr, "render: Vita GPU 3D rasterizer active\n");
        else fprintf(stderr, "render: Vita GPU 3D unavailable; software rasterizer\n");
    }
#endif
    if (getenv("ACWW_RENDER_COMPARE_MAX")) sCompareMax = strtol(getenv("ACWW_RENDER_COMPARE_MAX"), 0, 10);
    if (getenv("ACWW_RENDER_SELFTEST") || getenv("ACWW_RENDER_SELFTEST3D")) sCompare = sCompare ? sCompare : 1;
    sGpu2d = sOwn || sCompare;
    sGpu3d = sOwn3d || sCompare;
    sMelon = !sOwn || !sOwn3d || sCompare;
    if (sMelon) melon_init();
    gpu2d_reset();
    gpu2d_set_diagnostics(sCompare || sGpu2dMode == 2);
    gpu3d_reset();
    gpu3d_set_diagnostics(sCompare);
    {
        // GX command processing on its own core. Off for the comparison/self-test harnesses, which single-step
        // the pipeline.
        int on = sOwn3d && !sCompare && !sMelon;
        // Synchronous commit everywhere by default: the geometry committed at a vblank is the frame the game just
        // finished, paired with the texture/palette VRAM snapshotted at the same vblank. The pipelined commit
        // (ACWW_GX_ASYNC=1) can draw one frame's polygons with the next frame's palettes (wrong colours).
        // The Vita defaults to the pipelined commit: the game thread no longer waits at the vblank for the worker
        // (~3 ms a frame). Each geometry frame is drawn with the VRAM snapshot taken at its own vblank (render.c
        // keeps several), so late geometry still gets its own textures/palettes.
#if HOST_VITA
        int async = 1;
#else
        int async = 0;
#endif
        gpu3d_mt_enable(on, async);
    }
    if (getenv("ACWW_GPU2D_SELFTEST")) {
        if (sGpu2dMode != 2) {
            fprintf(stderr, "render: ACWW_GPU2D_SELFTEST needs ACWW_RENDERER3D=wgpu (and WebGPU)\n");
            exit(1);
        }
        sSelftest = 1;
        render_selftest_gpu2d(strtol(getenv("ACWW_GPU2D_SELFTEST"), 0, 10));
    }
    if (getenv("ACWW_RENDER_SELFTEST")) {
        sGpu2d = sGpu3d = sCompare = sSelftest = sMelon = 1;
        render_selftest(strtol(getenv("ACWW_RENDER_SELFTEST"), 0, 10));
    }
    if (getenv("ACWW_RENDER_SELFTEST3D")) {
        void render_selftest3d(long scenes) __attribute__((noreturn));
        sGpu2d = sGpu3d = sSelftest = sMelon = 1;
        render_selftest3d(strtol(getenv("ACWW_RENDER_SELFTEST3D"), 0, 10));
    }
#if HOST_VITA
    sRenderThreadOn = 1;
#else
    sRenderThreadOn = getenv("ACWW_RENDERTHREAD") && atoi(getenv("ACWW_RENDERTHREAD"));  // host validation only
#endif
    if (sRenderThreadOn) fprintf(stderr, "render: render thread active (overlaps 2D bands + 3D with the next frame's CPU)\n");
}

static int is_3d(u32 addr) {
    return addr == 0x04000060 || addr == 0x04000304 || (addr >= 0x04000240 && addr < 0x04000247) ||
           (addr >= 0x04000320 && addr < 0x040006a4);
}
void render_write8(u32 addr, u8 v) {
    if (sMelon) melon_write8(addr, v);
    if (sGpu2d) gpu2d_write8(addr, v);
    if (sGpu3d && is_3d(addr)) gpu3d_write8(addr, v);
}
void render_write16(u32 addr, u16 v) {
    if (sMelon) melon_write16(addr, v);
    if (sGpu2d) gpu2d_write16(addr, v);
    if (sGpu3d && is_3d(addr)) gpu3d_write16(addr, v);
}
void melon_3d_state(Gpu3dState *s);
static long sStateLines;
static void compare_state(u32 addr, u32 v) {
    Gpu3dState a, b;
    memset(&a, 0, sizeof a);
    memset(&b, 0, sizeof b);
    gpu3d_state(&a);
    melon_3d_state(&b);
    a.pipe |= gpu3d_read32(0x04000600) & 0xff00ff00;
    b.pipe |= melon_read32(0x04000600) & 0xff00ff00;
    {
        int nout, clipped, type;
        unsigned serial;
        static unsigned last;
        int outcome = gpu3d_debug_last_polygon(&nout, &clipped, &serial, &type);
        if (serial != last) {
            last = serial;
            static struct { int key[4]; int pp, slots, ctr, pv; long n; } tab[512];
            static int ntab;
            int key[4] = {outcome, nout, clipped, type};
            int i;
            for (i = 0; i < ntab; i++)
                if (!memcmp(tab[i].key, key, sizeof key) && tab[i].pp == b.pp && tab[i].slots == b.slots &&
                    tab[i].ctr == b.slotctr)
                    break;
            if (i == ntab && ntab < 512) {
                memcpy(tab[i].key, key, sizeof key);
                tab[i].pp = b.pp, tab[i].slots = b.slots, tab[i].ctr = b.slotctr;
                ntab++;
                fprintf(stderr, "polysetup: outcome %d nout %d clipped %d type %d -> melon pp %d slots %x ctr %d (vp %d)\n",
                        outcome, nout, clipped, type, b.pp, b.slots, b.slotctr, b.vp);
            }
        }
    }
    if (!memcmp(&a, &b, sizeof a) || sStateLines >= sCompareMax) return;
    sStateLines++;
    fprintf(stderr, "render-compare: frame %llu after write %08x = %08x the geometry state differs:\n",
            (unsigned long long)host_frame(), addr, v);
    fprintf(stderr, "  own   pipe %x fifo %u flush %u polys %u verts %u mode %u possp %u projsp %u\n", a.pipe, a.fifo,
            a.flush, a.npolys, a.nverts, a.mtxmode, a.possp, a.projsp);
    fprintf(stderr, "  melon pipe %x fifo %u flush %u polys %u verts %u mode %u possp %u projsp %u\n", b.pipe, b.fifo,
            b.flush, b.npolys, b.nverts, b.mtxmode, b.possp, b.projsp);
    fprintf(stderr, "  pipelines own vp %d np %d pp %d slots %x ctr %d, melon vp %d np %d pp %d slots %x ctr %d\n", a.vp, a.np,
            a.pp, a.slots, a.slotctr, b.vp, b.np, b.pp, b.slots, b.slotctr);
    const int32_t *ma[4] = {a.proj, a.pos, a.vec, a.tex}, *mb[4] = {b.proj, b.pos, b.vec, b.tex};
    for (int m = 0; m < 4; m++)
        if (memcmp(ma[m], mb[m], 64)) {
            fprintf(stderr, "  %s matrix:", (const char *[]){"projection", "position", "vector", "texture"}[m]);
            for (int i = 0; i < 16; i++) fprintf(stderr, " %x/%x", ma[m][i], mb[m][i]);
            fprintf(stderr, "\n");
        }
    if (memcmp(a.light, b.light, sizeof a.light)) {
        fprintf(stderr, "  lights:");
        for (int l = 0; l < 4; l++)
            fprintf(stderr, " %d,%d,%d/%d,%d,%d", a.light[l][0], a.light[l][1], a.light[l][2], b.light[l][0], b.light[l][1],
                    b.light[l][2]);
        fprintf(stderr, "\n");
    }
    if (memcmp(a.lightColor, b.lightColor, sizeof a.lightColor) || memcmp(a.material, b.material, sizeof a.material)) {
        fprintf(stderr, "  light colours / material (diffuse, ambient, specular, emission), own/melon:");
        for (int l = 0; l < 4; l++)
            fprintf(stderr, " %d,%d,%d/%d,%d,%d", a.lightColor[l][0], a.lightColor[l][1], a.lightColor[l][2],
                    b.lightColor[l][0], b.lightColor[l][1], b.lightColor[l][2]);
        fprintf(stderr, " |");
        for (int m = 0; m < 4; m++)
            fprintf(stderr, " %d,%d,%d/%d,%d,%d", a.material[m][0], a.material[m][1], a.material[m][2], b.material[m][0],
                    b.material[m][1], b.material[m][2]);
        fprintf(stderr, "\n");
    }
    if (memcmp(a.normal, b.normal, sizeof a.normal) || memcmp(a.vtxColor, b.vtxColor, 3))
        fprintf(stderr, "  normal %d,%d,%d / %d,%d,%d  vertex colour %d,%d,%d / %d,%d,%d\n", a.normal[0], a.normal[1],
                a.normal[2], b.normal[0], b.normal[1], b.normal[2], a.vtxColor[0], a.vtxColor[1], a.vtxColor[2],
                b.vtxColor[0], b.vtxColor[1], b.vtxColor[2]);
    if (memcmp(a.vtxColor, b.vtxColor, 3)) {
        fprintf(stderr, "  vector matrix:");
        for (int i = 0; i < 16; i++) fprintf(stderr, " %d", a.vec[i]);
        fprintf(stderr, "\n  lights:");
        for (int l = 0; l < 4; l++)
            fprintf(stderr, " %d,%d,%d c%d,%d,%d", a.light[l][0], a.light[l][1], a.light[l][2], a.lightColor[l][0],
                    a.lightColor[l][1], a.lightColor[l][2]);
        fprintf(stderr, "\n  material:");
        for (int m = 0; m < 4; m++) fprintf(stderr, " %d,%d,%d", a.material[m][0], a.material[m][1], a.material[m][2]);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "  own commands:");
    gpu3d_debug_history();
}
void render_write32(u32 addr, u32 v) {
    if (sMelon) melon_write32(addr, v);
    if (sGpu2d && (addr < 0x04000400 || addr > 0x04000600)) gpu2d_write32(addr, v);
    if (sGpu3d && is_3d(addr)) gpu3d_write32(addr, v);
    if (sCompare > 1 && !sSelftest && addr >= 0x04000400 && addr <= 0x04000600) compare_state(addr, v);
}

u32 render_read32(u32 addr) {
    if (!sOwn3d) {
        u32 m = melon_read32(addr);
        if (sCompare) {
            u32 o = gpu3d_read32(addr);
            sReads3d++;
            if (o != m) {
                if (sFirstReadDiffering == ~0ull) sFirstReadDiffering = host_frame();
                sReadsDiffering++;
                { void gpu3d_debug_history(void); if (sReadLines < 3) gpu3d_debug_history(); }
                if (sReadLines++ < sCompareMax)
                    fprintf(stderr, "render-compare: frame %llu read %08x: own %08x melon %08x\n",
                            (unsigned long long)host_frame(), addr, o, m);
            }
        }
        return m;
    }
    u32 o = gpu3d_read32(addr);
    if (sCompare) {
        u32 m = melon_read32(addr);
        sReads3d++;
        if (o != m) {
            if (sFirstReadDiffering == ~0ull) sFirstReadDiffering = host_frame();
            sReadsDiffering++;
            if (sReadLines++ < sCompareMax)
                fprintf(stderr, "render-compare: frame %llu read %08x: own %08x melon %08x\n",
                        (unsigned long long)host_frame(), addr, o, m);
        }
    }
    return o;
}
int render_gx_fifo_irq_level(void) {
    if (!sOwn3d) return melon_gx_fifo_irq_level();
    // Asked on every IRQ check (after every I/O access): only GXSTAT's FIFO-IRQ mode bits, read directly instead
    // of a full GXSTAT read (store-queue flush, diagnostics, the whole status word) - same answer.
    extern int gpu3d_fifo_irq_level(void);
    return gpu3d_fifo_irq_level();
}
void render_debug(void) { if (sMelon) melon_debug(); }

int render_compare(const char *kind, u64 label) {
    sCompared++;
    int differs = 0;
    for (int s = 0; s < 2; s++) {
        const u32 *m = melon_screen(s), *o = gpu2d_screen(s);
        int first = -1, count = 0;
        for (int i = 0; i < 256 * 192; i++)
            if ((m[i] ^ o[i]) & 0xffffff) { if (first < 0) first = i; count++; }
        if (first < 0) continue;
        differs = 1;
        if (sCompareLines < sCompareMax) {
            char what[256];
            gpu2d_describe_pixel(s, first % 256, first / 256, what, sizeof what);
            fprintf(stderr, "render-compare: %s %llu %s screen: %d pixels differ, first (%d, %d) own %06x melon %06x: %s\n",
                    kind, (unsigned long long)label, s ? "bottom" : "top", count, first % 256, first / 256,
                    o[first] & 0xffffff, m[first] & 0xffffff, what);
            if (++sCompareLines == sCompareMax) fprintf(stderr, "render-compare: (no more lines: ACWW_RENDER_COMPARE_MAX)\n");
        }
    }
    if (differs) {
        if (sFirstDiffering == ~0ull) sFirstDiffering = label;
        sDiffering++;
    }
    return differs;
}

static void print_poly(const char *who, const Gpu3dVertexOut *v, int n, u32 attr, u32 tp, u32 pal, int flags) {
    fprintf(stderr, "    %s: attr %08x tex %08x pal %04x flags %x, %d vertices:", who, attr, tp, pal, flags, n);
    for (int k = 0; k < n; k++)
        fprintf(stderr, " (%d,%d z%06x w%x c%d,%d,%d t%d,%d)", v[k].x, v[k].y, v[k].z, v[k].w, v[k].r, v[k].g, v[k].b,
                v[k].s, v[k].t);
    fprintf(stderr, "\n");
}
enum { PK_COUNT, PK_ATTR, PK_FLAGS, PK_POS, PK_DEPTH, PK_COLOR, PK_TEX, PK_N };
static const char *const kPolyKinds[PK_N] = {"vertex count", "attributes", "flags", "position", "depth", "colour", "texcoord"};
static u64 sPolyKind[PK_N][2], sPolysCompared;
static void census_polygons(u64 label) {
    int nm = melon_3d_poly_count(), no = gpu3d_poly_count();
    int n = nm < no ? nm : no;
    for (int i = 0; i < n; i++) {
        Gpu3dVertexOut vm[10], vo[10];
        u32 am, ao, tm, to, pm, po;
        int fm, fo;
        memset(vm, 0, sizeof vm);
        memset(vo, 0, sizeof vo);
        int km = melon_3d_poly_get(i, vm, &am, &tm, &pm, &fm);
        int ko = gpu3d_poly_get(i, vo, &ao, &to, &po, &fo);
        sPolysCompared++;
        int clipped = 0;
        for (int k = 0; k < km; k++) clipped |= vm[k].clipped;
        int kinds[PK_N] = {0};
        kinds[PK_COUNT] = km != ko;
        kinds[PK_ATTR] = am != ao || tm != to || pm != po;
        kinds[PK_FLAGS] = fm != fo;
        for (int k = 0; k < km && k < ko; k++) {
            kinds[PK_POS] |= vm[k].x != vo[k].x || vm[k].y != vo[k].y;
            kinds[PK_DEPTH] |= vm[k].z != vo[k].z || vm[k].w != vo[k].w;
            kinds[PK_COLOR] |= vm[k].r != vo[k].r || vm[k].g != vo[k].g || vm[k].b != vo[k].b;
            kinds[PK_TEX] |= vm[k].s != vo[k].s || vm[k].t != vo[k].t;
        }
        static FILE *dump;
        static long dumped, dumpMax = -1;
        if (dumpMax < 0) dumpMax = getenv("ACWW_RENDER_FITDUMP_MAX") ? atol(getenv("ACWW_RENDER_FITDUMP_MAX")) : 200000;
        if (!dump && getenv("ACWW_RENDER_FITDUMP")) dump = fopen(getenv("ACWW_RENDER_FITDUMP"), "w");
        if (dump && km == ko && dumped < dumpMax) {
            for (int k = 0; k < km; k++, dumped++) {
                fprintf(dump, "V %llu %d %d %d ", (unsigned long long)label, i, k, vm[k].clipped);
                fprintf(dump, "own %d %d %d %d %d %d %d %d %d melon %d %d %d %d %d %d %d %d %d ", vo[k].x, vo[k].y,
                        vo[k].z, vo[k].w, vo[k].r, vo[k].g, vo[k].b, vo[k].s, vo[k].t, vm[k].x, vm[k].y, vm[k].z, vm[k].w,
                        vm[k].r, vm[k].g, vm[k].b, vm[k].s, vm[k].t);
                fprintf(dump, "mclip %d %d %d %d ", vm[k].clip[0], vm[k].clip[1], vm[k].clip[2], vm[k].clip[3]);
                void gpu3d_debug_vertex(int i, int k, FILE *f);
                gpu3d_debug_vertex(i, k, dump);
                fprintf(dump, "\n");
            }
        }
        for (int k = 0; k < PK_N; k++) {
            if (!kinds[k]) continue;
            if (!sPolyKind[k][clipped]++) {
                fprintf(stderr, "render-compare: frame %llu polygon %d: %s differs (%s), the first such:\n",
                        (unsigned long long)label, i, kPolyKinds[k], clipped ? "clipped" : "not clipped");
                print_poly("own  ", vo, ko, ao, to, po, fo);
                print_poly("melon", vm, km, am, tm, pm, fm);
                gpu3d_debug_poly(i);
            }
            break;
        }
    }
}
static int compare_polygons(const char *kind, u64 label) {
    sPolyFrames++;
    census_polygons(label);
    if (getenv("ACWW_RENDER_POLYDUMP") && label == strtoull(getenv("ACWW_RENDER_POLYDUMP"), 0, 10)) {
        int nm = melon_3d_poly_count(), no = gpu3d_poly_count();
        for (int i = 0; i < (nm > no ? nm : no); i++) {
            Gpu3dVertexOut v[10];
            u32 a, t, p;
            int f;
            for (int w = 0; w < 2; w++) {
                memset(v, 0, sizeof v);
                int k = w ? melon_3d_poly_get(i, v, &a, &t, &p, &f) : gpu3d_poly_get(i, v, &a, &t, &p, &f);
                fprintf(stderr, "%s %4d %08x %08x f%d:", w ? " | melon" : "polydump own", i, a, t, f);
                for (int j = 0; j < k; j++)
                    fprintf(stderr, " %d,%d,%x c%d,%d,%d t%d,%d", v[j].x, v[j].y, v[j].z, v[j].r >> 4, v[j].g >> 4, v[j].b >> 4,
                            v[j].s, v[j].t);
            }
            fprintf(stderr, "\n");
        }
    }
    int nm = melon_3d_poly_count(), no = gpu3d_poly_count();
    int n = nm < no ? nm : no, bad = -1;
    Gpu3dVertexOut vm[10], vo[10];
    u32 am = 0, ao = 0, tm = 0, to = 0, pm = 0, po = 0;
    int fm = 0, fo = 0, km = 0, ko = 0;
    for (int i = 0; i < n && bad < 0; i++) {
        memset(vm, 0, sizeof vm);
        memset(vo, 0, sizeof vo);
        km = melon_3d_poly_get(i, vm, &am, &tm, &pm, &fm);
        ko = gpu3d_poly_get(i, vo, &ao, &to, &po, &fo);
        if (km != ko || am != ao || tm != to || pm != po || fm != fo || memcmp(vm, vo, sizeof vm)) bad = i;
    }
    if (bad < 0 && nm == no) return 0;
    if (bad < 0) {
        bad = n;
        memset(vm, 0, sizeof vm);
        memset(vo, 0, sizeof vo);
        km = melon_3d_poly_get(n, vm, &am, &tm, &pm, &fm);
        ko = gpu3d_poly_get(n, vo, &ao, &to, &po, &fo);
    }
    if (sFirstPolyDiffering == ~0ull) sFirstPolyDiffering = label;
    sPolyDiffering++;
    if (sPolyLines++ < sCompareMax) {
        fprintf(stderr, "render-compare: %s %llu polygons: own %d melon %d", kind, (unsigned long long)label, no, nm);
        if (bad >= 0) {
            fprintf(stderr, ", the first differing: %d\n", bad);
            print_poly("own  ", vo, ko, ao, to, po, fo);
            print_poly("melon", vm, km, am, tm, pm, fm);
            gpu3d_debug_poly(bad);
        } else fprintf(stderr, "\n");
    }
    return 1;
}
static int compare_3d(const char *kind, u64 label) {
    sPix3dFrames++;
    melon_3d_lines(s3d);
    const u32 *o = gpu3d_lines();
    int first = -1, count = 0;
    for (int i = 0; i < 256 * 192; i++)
        if ((o[i] ^ s3d[i]) & 0x1f3f3f3f) { if (first < 0) first = i; count++; }
    {
        const char *d = getenv("ACWW_RENDER_DUMP3D");
        char *colon;
        if (d && strtoull(d, &colon, 10) == label && *colon == ':') {
            char name[512];
            snprintf(name, sizeof name, "%s.bin", colon + 1);
            FILE *f = fopen(name, "wb");
            if (f) {
                fwrite(o, 4, 256 * 192, f);
                fwrite(s3d, 4, 256 * 192, f);
                int g3_raster_pixel_polygon(int x, int y);
                for (int i = 0; i < 256 * 192; i++) {
                    int16_t v = (int16_t)g3_raster_pixel_polygon(i % 256, i / 256);
                    fwrite(&v, 2, 1, f);
                }
                u32 g3_raster_depth(int x, int y);
                int melon_3d_buffers(u32 *depth, u32 *attr);
                static u32 dep[256 * 192];
                for (int i = 0; i < 256 * 192; i++) dep[i] = g3_raster_depth(i % 256, i / 256);
                fwrite(dep, 4, 256 * 192, f);
                if (!melon_3d_buffers(dep, 0)) memset(dep, 0, sizeof dep);
                fwrite(dep, 4, 256 * 192, f);
                fclose(f);
            }
            snprintf(name, sizeof name, "%s.txt", colon + 1);
            f = fopen(name, "w");
            if (f) {
                for (int i = 0; i < gpu3d_poly_count(); i++) {
                    Gpu3dVertexOut v[10];
                    u32 a, t, p;
                    int fl, k = gpu3d_poly_get(i, v, &a, &t, &p, &fl);
                    fprintf(f, "poly %08x %08x %04x %d %d", a, t, p, fl, k);
                    for (int j = 0; j < k; j++)
                        fprintf(f, " %d %d %d %d %d %d %d %d %d", v[j].x, v[j].y, v[j].z, v[j].w, v[j].r, v[j].g, v[j].b,
                                v[j].s, v[j].t);
                    fprintf(f, "\n");
                }
                fclose(f);
            }
        }
    }
    if (first < 0) return 0;
    if (sFirstPix3dDiffering == ~0ull) sFirstPix3dDiffering = label;
    sPix3dDiffering++;
    if (sPix3dLines++ < sCompareMax) {
        Gpu3dPolyInfo pi;
        char what[160] = "the clear colour";
        if (gpu3d_pixel_polygon(first % 256, first / 256, &pi))
            snprintf(what, sizeof what, "polygon %d (attr %08x tex %08x pal %04x, %d vertices%s)", pi.index, pi.attr,
                     pi.texparam, pi.texpal, pi.nverts, pi.translucent ? ", translucent" : "");
        fprintf(stderr, "render-compare: %s %llu 3D: %d pixels differ, first (%d, %d) own %08x melon %08x: %s\n", kind,
                (unsigned long long)label, count, first % 256, first / 256, o[first], s3d[first], what);
    }
    return 1;
}
int render_compare_3d(const char *kind, u64 label) {
    int a = compare_polygons(kind, label);
    int b = compare_3d(kind, label);
    return a || b;
}

static void gpu3d_run_frame(void) {
    const u8 *banks[9];
    for (int i = 0; i < 9; i++) banks[i] = host_vram_bank(i);
    gpu3d_frame(banks);
}

int render_check_gpu2d(void) {
    static u32 gpu[256 * 384 * 2];
    static long reported;
    if (!wdisp_read_2d(gpu)) return 1;
    sGpu2dChecked++;
    int count = 0, first = -1;
    for (int s = 0; s < 2; s++) {
        const u32 *cpu = gpu2d_deferred(s);
        for (int i = 0; i < 256 * 192 * 2; i++)
            if (cpu[i] != gpu[s * 256 * 192 * 2 + i]) {
                if (first < 0) first = s * 256 * 192 * 2 + i;
                count++;
            }
    }
    if (!count) return 0;
    sGpu2dDiffering++;
    if (reported++ < sCompareMax) {
        int px = first / 2, s = px / (256 * 192), y = px / 256 % 192, x = px % 256, w = first & 1;
        const u32 *cpu = gpu2d_deferred(s) + (y * 256 + x) * 2;
        char what[256];
        gpu2d_describe_pixel(s, x, y, what, sizeof what);
        fprintf(stderr, "gpu2d-check: frame %llu: %d words differ, first %s screen (%d, %d) word %d: cpu %08x %08x gpu "
                "%08x %08x (%s)\n", (unsigned long long)host_frame(), count, s ? "bottom" : "top", x, y, w, cpu[0], cpu[1],
                gpu[(px) * 2], gpu[px * 2 + 1], what);
    }
    return 1;
}

static uint64_t sHblankNs;   // of the 2D prepass: the hblank IRQ replay (game code), vs. the per-line snapshots
static void render_hblank(int line, void *context) {
    (void)context;
    uint64_t t = host_plat_time_ns();
    host_hblank(line);
    sHblankNs += host_plat_time_ns() - t;
}
static uint64_t sRender2dNs, sRender3dNs;
static int sRenderFrames;

// The worker's half of a frame: pure compute over the snapshot (no game code). Produces the 2D canvas, rasterizes
// the committed 3D, then paces + presents. All GPU/GL work lives here so the GL context stays on one thread.
static uint64_t sWorkerNs;
static uint64_t sWBandsNs, sWPresentNs;
static uint64_t sHoWaitRtNs, sHoSnapNs, sHoCommitNs;   // game-thread hand-off split
// --- FPS overlay (Vita, while testing; ACWW_FPS=0 hides it) ---------------------------------------------------------
// Top-left: frames shown per second, and game speed (DS frames run per second / 60, so 100% = full speed).
static unsigned sVblankCount;   // DS frames run (render_frame calls, game thread)
static const unsigned short kGlyph[128] = {   // 3x5, bit 14 = top-left
    ['0'] = 0x7b6f, ['1'] = 0x2c97, ['2'] = 0x73e7, ['3'] = 0x73cf, ['4'] = 0x5bc9, ['5'] = 0x79cf, ['6'] = 0x79ef,
    ['7'] = 0x7249, ['8'] = 0x7bef, ['9'] = 0x7bcf, ['F'] = 0x79a4, ['P'] = 0x7be4, ['S'] = 0x79cf, ['%'] = 0x52a5,
    ['G'] = 0x796f, ['A'] = 0x2bed, ['M'] = 0x5fed, ['E'] = 0x79e7,
};
static void overlay_text(uint32_t *c, int x0, int y0, const char *s) {
    int n = (int)strlen(s), w = n * 8 + 2, h = 14;
    for (int y = y0; y < y0 + h && y < 192; y++)
        for (int x = x0; x < x0 + w && x < 256; x++) c[y * 256 + x] = 0x000000;
    for (int i = 0; i < n; i++) {
        unsigned g = kGlyph[(unsigned char)s[i] & 127];
        for (int r = 0; r < 5; r++)
            for (int col = 0; col < 3; col++)
                if (g >> (14 - (r * 3 + col)) & 1)
                    for (int dy = 0; dy < 2; dy++)
                        for (int dx = 0; dx < 2; dx++) {
                            int x = x0 + 2 + i * 8 + col * 2 + dx, y = y0 + 2 + r * 2 + dy;
                            if (x < 256 && y < 192) c[y * 256 + x] = 0xffffff;
                        }
    }
}
static void overlay_fps(uint32_t *canvas) {
    static int on = -1;
    if (on < 0) on = !(getenv("ACWW_FPS") && atoi(getenv("ACWW_FPS")) == 0);
    if (!on || !canvas) return;
    static uint64_t t0;
    static unsigned frames, vb0, fps, speed;
    uint64_t now = host_plat_time_ns();
    frames++;
    if (!t0) { t0 = now; vb0 = __atomic_load_n(&sVblankCount, __ATOMIC_RELAXED); }
    if (now - t0 >= 500000000u) {
        unsigned vb = __atomic_load_n(&sVblankCount, __ATOMIC_RELAXED);
        double sec = (now - t0) / 1e9;
        fps = (unsigned)(frames / sec + 0.5);
        speed = (unsigned)((vb - vb0) / sec / 60.0 * 100.0 + 0.5);
        frames = 0; t0 = now; vb0 = vb;
    }
    char buf[32];
    snprintf(buf, sizeof buf, "%u FPS %u%%", fps, speed);
    overlay_text(canvas, 0, 0, buf);
}   // render-thread split: 2D bands / present (3D is sRender3dNs)
// 2D job thread: this frame's 2D bands composite with the PREVIOUS frame's 3D image (the 3D output is double-
// buffered: the 2D input pointer was taken in the prepass, and this frame's 3D renders into the other buffer), so
// the 2D work runs here while the render thread does the 3D, instead of one after the other.
static pthread_mutex_t s2dJobMx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t s2dJobGo = PTHREAD_COND_INITIALIZER, s2dJobDoneCv = PTHREAD_COND_INITIALIZER;
static int s2dJobPending, s2dJobDone = 1, s2dJobStarted;
static uint64_t s2dJobNs;
static void *job2d_main(void *a) {
    (void)a;
#if HOST_VITA
    extern int sceKernelChangeThreadCpuAffinityMask(int, int);
    sceKernelChangeThreadCpuAffinityMask(0, 0x60000);   // cores 1+2: core 0 belongs to the game thread
#endif
    for (;;) {
        pthread_mutex_lock(&s2dJobMx);
        while (!s2dJobPending) pthread_cond_wait(&s2dJobGo, &s2dJobMx);
        s2dJobPending = 0;
        pthread_mutex_unlock(&s2dJobMx);
        uint64_t t = host_plat_time_ns();
        gpu2d_render_finish();                 // 2D bands + canvas from the per-line/VRAM snapshots
        s2dJobNs = host_plat_time_ns() - t;
        pthread_mutex_lock(&s2dJobMx);
        s2dJobDone = 1;
        pthread_cond_signal(&s2dJobDoneCv);
        pthread_mutex_unlock(&s2dJobMx);
    }
    return 0;
}
static int job2d_start(void) {
    static int ok = -1;
    if (ok < 0) {
        ok = 1;
        {
            pthread_t t;
            pthread_attr_t at;
            pthread_attr_init(&at);
            pthread_attr_setstacksize(&at, 1 << 20);
            ok = pthread_create(&t, &at, job2d_main, 0) == 0;
            s2dJobStarted = ok;
        }
    }
    if (!ok) return 0;
    pthread_mutex_lock(&s2dJobMx);
    s2dJobDone = 0;
    s2dJobPending = 1;
    pthread_cond_signal(&s2dJobGo);
    pthread_mutex_unlock(&s2dJobMx);
    return 1;
}
static void job2d_wait(void) {
    pthread_mutex_lock(&s2dJobMx);
    while (!s2dJobDone) pthread_cond_wait(&s2dJobDoneCv, &s2dJobMx);
    pthread_mutex_unlock(&s2dJobMx);
}
// The longest render-thread frame since the last report and its parts (us), for the "render thread split" line.
static unsigned long long sWWorst, sWWorst3d, sWWorstWait2d, sWWorstPresent, sWWorstTex, sWWorstUploads, sWWorstSync,
    sWWorstSwap, sWWorstBands;
// Whether the main engine's screen is fully black for the frame being handed off (its brightness-down effect at 16 on
// every layer with no window or second target, or master brightness at -16): a late frame there repeats a black
// picture, which cannot be seen. Set at hand-off, read when the render thread starts that frame (log only).
static int sHoBlack, sWWorstBlack;
static void render_worker_phase(void) {
    uint64_t tw = host_plat_time_ns();
    const int black = sHoBlack;
    uint64_t f3d = 0, fwait = 0, fpres = 0, fbands = 0;
    int par = job2d_start();
    if (!par) {
        gpu2d_render_finish();             // 2D bands + canvas from the per-line/VRAM snapshots
        __atomic_fetch_add(&sWBandsNs, host_plat_time_ns() - tw, __ATOMIC_RELAXED);
    }
    if (!(sGpu3d && sDraw3d)) { extern void gpu3d_release_committed(void); gpu3d_release_committed(); }
    if (sGpu3d && sDraw3d) {
        uint64_t t3d = host_plat_time_ns();
        gpu3d_render_committed();
        f3d = host_plat_time_ns() - t3d;
        sRender3dNs += f3d;
    }
    if (par) {
        uint64_t tj = host_plat_time_ns();
        job2d_wait();
        fwait = host_plat_time_ns() - tj;
        fbands = s2dJobNs;
        __atomic_fetch_add(&sWBandsNs, s2dJobNs, __ATOMIC_RELAXED);
    }
    // Present the canvas this worker just produced. Only the GL present happens here: the frontend body (input
    // polling, UI) is a coroutine on the main thread's stack and keeps running there every DS frame.
#if HOST_VITA
    {
        uint64_t tp = host_plat_time_ns();
        extern const uint32_t *gpu2d_canvas(int *rows);
        extern void vita_gl_present_canvas(const void *, int, int, int, int, int, int);
        extern int vita_gl_gc_init(void);
        extern void vita_gl_present_gc(const Gpu2dGc *, unsigned, const uint32_t *);
        extern unsigned vita_gl3d_frame_tex(void);
        static int gcTried;
        if (!gcTried) { gcTried = 1; if (sGpu3d && vita_gl_gc_init()) gpu2d_set_gpu_compose(1); }
        const Gpu2dGc *gc = gpu2d_gc_frame();
        if (gc) {   // composed on the GPU: the 2D layers + this frame's 3D colour buffer
            static uint32_t ovl[256 * 16];
            for (int i = 0; i < 256 * 16; i++) ovl[i] = 0x01000000u;   // (alpha byte 0 = an overlay pixel)
            // the FPS counter only when the options ask for it (its rows otherwise cost the compose for nothing)
            const int fpsOn = feature_on(FEAT_FPS);
            if (fpsOn) overlay_fps(ovl);
            vita_gl_present_gc(gc, vita_gl3d_frame_tex(), fpsOn ? ovl : NULL);
        } else {
            int rows = 0;
            const uint32_t *c = gpu2d_canvas(&rows);
            if (c && rows >= 192 && rows <= 640) overlay_fps((uint32_t *)c);   // the canvas is rebuilt every frame
            if (c && rows >= 192 && rows <= 640) vita_gl_present_canvas(c, 256, rows, 0, 0, 256, rows);
        }
        fpres = host_plat_time_ns() - tp;
        __atomic_fetch_add(&sWPresentNs, fpres, __ATOMIC_RELAXED);
    }
#endif
    uint64_t ftot = host_plat_time_ns() - tw;
    if (ftot / 1000 > sWWorst) {
#if HOST_VITA
        extern unsigned long long gG3FrameTexNs, gG3FrameUploads, gG3FrameSyncNs, gVitaFrameSwapNs;
        sWWorstTex = gG3FrameTexNs / 1000; sWWorstUploads = gG3FrameUploads; sWWorstSync = gG3FrameSyncNs / 1000;
        sWWorstSwap = gVitaFrameSwapNs / 1000;
#endif
        sWWorstBlack = black;
        sWWorst = ftot / 1000; sWWorst3d = f3d / 1000; sWWorstWait2d = fwait / 1000; sWWorstPresent = fpres / 1000;
        sWWorstBands = fbands / 1000;
    }
    __atomic_fetch_add(&sWorkerNs, ftot, __ATOMIC_RELAXED);
}

static void *render_worker_main(void *a) {
    (void)a;
#if HOST_VITA
    {
        extern int gVitaDeferSwap;
        gVitaDeferSwap = 1;   // this thread now does all GL: its swaps come after the hand-off (below)
    }
    extern int sceKernelChangeThreadCpuAffinityMask(int, int);
    sceKernelChangeThreadCpuAffinityMask(0, 0x60000);   // cores 1+2: core 0 belongs to the game thread
#endif
    for (;;) {
        pthread_mutex_lock(&sWMx);
        while (!sWReady) pthread_cond_wait(&sWCvWork, &sWMx);
        sWReady = 0;
        pthread_mutex_unlock(&sWMx);
        render_worker_phase();
        pthread_mutex_lock(&sWMx);
        sWDone = 1;
        pthread_cond_signal(&sWCvDone);
        pthread_mutex_unlock(&sWMx);
#if HOST_VITA
        {   // the frame's data is read and its drawing submitted: the game goes on while the swap waits for the display
            extern void vita_gl_swap_pending(void);
            vita_gl_swap_pending();
        }
#endif
    }
    return 0;
}

// Barrier: block until the worker has finished the in-flight frame (it reads the snapshot buffers we are about to
// overwrite, and the 2D/3D engine state we are about to mutate).
void render_thread_sync(void) {
    if (!sWorkerSpawned) return;
    extern volatile unsigned char gHostPhase;   // phase sampler (host.h): 6 = PH_WAIT
    unsigned char ph = gHostPhase;
    gHostPhase = 6;
    pthread_mutex_lock(&sWMx);
    while (!sWDone) pthread_cond_wait(&sWCvDone, &sWMx);
    pthread_mutex_unlock(&sWMx);
    gHostPhase = ph;
}

static void render_worker_kick(void) {
    if (!sWorkerSpawned) {
        pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_setstacksize(&at, 2 << 20);
        pthread_create(&sWorker, &at, render_worker_main, 0);
        sWorkerSpawned = 1;
    }
    pthread_mutex_lock(&sWMx);
    sWDone = 0; sWReady = 1;
    pthread_cond_signal(&sWCvWork);
    pthread_mutex_unlock(&sWMx);
}

// hw.c calls this after render_frame(); when the worker owns this frame's present/pace, hw.c must not do its own.
int render_owns_present(void) { return sLastMt; }

static int render_thread_applicable(void) {
    if (!(sRenderThreadOn && sOwn && sOwn3d && sGpu3d && !sGpuDisplay && !sGpu2dMode && !sMelon && !sCompare &&
          !sSelftest))
        return 0;
#if HOST_VITA
    {
        extern int vita_gl_enabled(void);
        if (!vita_gl_enabled()) return 0;   // the worker presents through vitaGL only
    }
#endif
    return 1;
}

static void render_frame_sync(void);

// Main-thread half: wait for the previous frame's worker, snapshot inputs, run the game-code prepass + geometry
// commit, then hand the frame to the worker and return so the main thread can start the next frame's CPU.
static void render_frame_mt(void) {
#if HOST_VITA
    {
        // The game thread gets core 0 to itself; the GX worker, render thread and 2D band helper share cores 1+2.
        // Done here (after start-up) so library threads created during init keep their default masks.
        static int pinned;
        extern int sceKernelChangeThreadCpuAffinityMask(int, int);
        // (Priority above the render-side threads' 159.)
        extern int sceKernelChangeThreadPriority(int, int);
        if (!pinned) { pinned = 1; sceKernelChangeThreadCpuAffinityMask(0, 0x10000); sceKernelChangeThreadPriority(0, 128); }
    }
#endif
    uint64_t h0 = host_plat_time_ns();
    render_thread_sync();
    uint64_t h1 = host_plat_time_ns();
    // VRAM snapshot buffers. Synchronous commit: one. Pipelined: a few, each tagged with the geometry frame
    // (SWAP_BUFFERS count) issued in the DS frame it was taken at; a buffer is reused once its geometry frame has
    // been drawn (or it has none).
    enum { SNAPS = 4 };
    static u8 *snapMem[SNAPS];
    static unsigned snapTag[SNAPS], lastIssued, lastSwapped;
    static u8 snapCnt[SNAPS][9];
    static u32 snapGens[SNAPS][9];
    static const u8 *snapBanks[SNAPS][9];
    extern unsigned gpu3d_commit_swapped(void), gpu3d_swaps_issued(void);
    extern int gpu3d_pipelined(void);
    extern void gpu3d_bind_banks(const uint8_t *const bank[9], const uint8_t cnt[9]);
    extern u8 gG3VramCnt[9];
    const int piped = gpu3d_pipelined();
    int slot = 0;
    if (piped) {
        int best = -1;
        for (int k = 0; k < SNAPS; k++)
            if (snapTag[k] <= lastSwapped && (best < 0 || snapTag[k] < snapTag[best])) best = k;
        if (best < 0) for (int k = 0; k < SNAPS; k++) if (best < 0 || snapTag[k] < snapTag[best]) best = k;
        slot = best;
    }
    {
        // Main-screen-only (the Vita, and the host with ACWW_FORCEMAIN): per-bank copies of just the banks the
        // engine-A 2D and the 3D read this frame, shared between slots while unchanged. Otherwise the whole block.
        static int mainOnly = -1;
        if (mainOnly < 0) {
#if HOST_VITA
            mainOnly = 1;
#else
            mainOnly = getenv("ACWW_FORCEMAIN") && atoi(getenv("ACWW_FORCEMAIN"));
#endif
        }
        extern void host_vram_snapshot_banks(const u8 *out[9], int slot, int keepLcdc);
        extern int gpu2d_main_vram_display(void);
#if HOST_VITA
        {   // the title screen: the sub screen's logo is drawn onto the main screen (gpu2d), so keep its banks too
            extern unsigned Scene_GetCurrent(void);
            extern int gHostSnapSub;
            unsigned sc = Scene_GetCurrent();
            extern int gpu2d_sub_shown(void);
            gHostSnapSub = sc == 0x2c || sc == 0x3f || gpu2d_sub_shown();   // the title's logo; the boot screens'
                                                                            // Nintendo logo; the menus (engine B)
        }
#endif
        if (mainOnly) host_vram_snapshot_banks(sSnapBank, slot, gpu2d_main_vram_display());
        else {
            if (!snapMem[slot]) snapMem[slot] = malloc(host_vram_total());
            sSnapBankMem = snapMem[slot];
            host_vram_snapshot_live(sSnapBankMem);   // one pass: equivalent to host_vram_flush() + host_vram_snapshot()
            for (int i = 0; i < 9; i++) sSnapBank[i] = sSnapBankMem + host_vram_bank_offset(i);
        }
        // Bank generations as of this snapshot, for the 3D texture sync (gpu3d_tex.h).
        extern u32 gG3BankGenFrame[9];
        extern int gG3BankGenValid;
        extern const u32 *host_vram_bank_gens(void);
        memcpy(gG3BankGenFrame, host_vram_bank_gens(), sizeof gG3BankGenFrame);
        gG3BankGenValid = 1;
        memcpy(snapGens[slot], gG3BankGenFrame, sizeof snapGens[slot]);
    }
    {
        unsigned issued = gpu3d_swaps_issued();
        snapTag[slot] = issued != lastIssued ? issued : 0;   // the geometry frame finished in this DS frame
        lastIssued = issued;
        memcpy(snapCnt[slot], gG3VramCnt, sizeof snapCnt[slot]);
        memcpy(snapBanks[slot], sSnapBank, sizeof snapBanks[slot]);
    }
    memcpy(sSnapPal, gHostPltt, sizeof sSnapPal);
    memcpy(sSnapOam, gHostOam, sizeof sSnapOam);
    {
        extern u8 *host_io(u32 addr);
        const u16 bc = *(const u16 *)host_io(0x04000050), by = *(const u16 *)host_io(0x04000054) & 31;
        const u16 mb = *(const u16 *)host_io(0x0400006c);
        const u32 dc = *(const u32 *)host_io(0x04000000);
        sHoBlack = ((mb >> 14) == 2 && (mb & 31) >= 16) ||
                   ((bc >> 6 & 3) == 3 && (bc & 0x3f) == 0x3f && !(bc >> 8 & 0x3f) && by >= 16 && !(dc >> 13 & 7));
    }
    const u32 *lines3d = sOwn3d ? gpu3d_lines() : s3d;
    Gpu2dFrameInput *in = &sWorkIn;
    for (int i = 0; i < 9; i++) in->bank[i] = sSnapBank[i];
    in->palette = gHostPltt;   // live: the hblank prepass rewrites it per line; the bands only read the per-line copies
    in->oam = sSnapOam;
    in->lines3d = lines3d;
    in->lines3dTop = sOwn3d && lines3d ? gpu3d_lines_top() : NULL;
    in->topRows = sOwn3d ? gpu3d_top_rows() : 192;
    in->hblank = sOwn ? render_hblank : 0;
    in->hblankContext = 0;
    in->defer3d = 0;
    in->gpu = 0;
    uint64_t t2d = host_plat_time_ns();
    sHoSnapNs += t2d - h1;
    sHoWaitRtNs += h1 - h0;
    gpu2d_set_defer_bands(1);
    gpu2d_render_frame(in);                 // main-thread prepass only (bands deferred to the worker)
    sRender2dNs += host_plat_time_ns() - t2d;
    {
        extern int gHostCanvasRows;
        int aBottom = gpu2d_main_on_bottom();
        int above = gHostCanvasRows - (aBottom ? 192 : 384);
        gpu3d_set_tall(gpu2d_tall_3d_ok() && above > 0, above);
    }
    {
        extern const u8 *host_vram_snapshot_dirty(void);
        static const u8 *dirty[9];
        const u8 *d = host_vram_snapshot_dirty();
        for (int i = 0; i < 9; i++) dirty[i] = d + host_vram_bank_offset(i) / 1024;
        (void)dirty;   // the snapshot is a plain copy (no dirty flags): the texture setup compares on the render thread
    }
    uint64_t c0 = host_plat_time_ns();
    gpu3d_commit(sSnapBank);                // swap geometry + bind snapshot texture banks (no rasterization)
    if (piped) {
        // Draw 3D only when a new geometry frame came in, with the snapshot taken when that frame was finished.
        unsigned sw = gpu3d_commit_swapped();
        sDraw3d = sw != 0;
        if (sw) {
            lastSwapped = sw;
            for (int k = 0; k < SNAPS; k++)
                if (snapTag[k] == sw && snapMem[k]) {
                    extern u32 gG3BankGenFrame[9];
                    gpu3d_bind_banks(snapBanks[k], snapCnt[k]);
                    memcpy(gG3BankGenFrame, snapGens[k], sizeof gG3BankGenFrame);
                    break;
                }
        }
    } else sDraw3d = 1;
#if HOST_VITA
    {   // engine B is the picture shown (a menu, the boot screens): the 3D is not seen, skip drawing it - the render
        // thread's cores are then free for engine B's bands
        extern int gpu2d_mirror_shown(void);
        if (gpu2d_mirror_shown()) sDraw3d = 0;
    }
#endif
    sHoCommitNs += host_plat_time_ns() - c0;
    {   // this frame's hand-off, for the pacing report's longest frame (hw.c)
        extern unsigned long long gHoFrameWaitNs, gHoFrameNs;
        gHoFrameWaitNs = h1 - h0;
        gHoFrameNs = host_plat_time_ns() - h0;
    }
    render_worker_kick();
    if (++sRenderFrames >= 60) {
        extern unsigned long long gpu3d_mt_skipped(void);
        // 60 drawn frames: times are per drawn frame; "skipped" = DS frames that ran without being drawn
        fprintf(stderr, "render: 2d prepass %llu us, 3d %llu us, render worker total %llu us (avg/drawn frame);"
                " %u frames not drawn, %llu geometry frames not built\n",
                (unsigned long long)(sRender2dNs / 60000), (unsigned long long)(sRender3dNs / 60000),
                (unsigned long long)(__atomic_exchange_n(&sWorkerNs, 0, __ATOMIC_RELAXED) / 60000),
                sRenderSkipped, gpu3d_mt_skipped());
        {   // the Vita's CPU-drawn sub-screen picture (title logo, boot screens, menus): redrawn only on change
            extern unsigned gSubFrames, gSubDraws;
            extern unsigned long long gSubCheckNs;
            if (gSubFrames)
                fprintf(stderr, "render: sub-screen picture redrawn %u of %u frames (change check %llu us/frame)\n",
                        gSubDraws, gSubFrames, gSubCheckNs / 1000 / gSubFrames);
            gSubFrames = gSubDraws = 0;
            gSubCheckNs = 0;
            {   // the menus over the scene: engine B's lines redrawn on the CPU (the bands) and what they cost
                extern unsigned long long gBProfNs, gBProfLines, gBProfRuns, gBgKindN[2][5][2];
                if (gBProfLines)
                    fprintf(stderr, "render: menu layer %llu lines redrawn, %llu us/line (its BG lines: text %llu, %llu us; "
                            "affine %llu, %llu us; extended %llu, %llu us; bitmap %llu, %llu us)\n",
                            gBProfLines, gBProfNs / 1000 / gBProfLines, gBgKindN[1][1][0], gBgKindN[1][1][1] / 1000,
                            gBgKindN[1][2][0], gBgKindN[1][2][1] / 1000, gBgKindN[1][3][0], gBgKindN[1][3][1] / 1000,
                            gBgKindN[1][4][0], gBgKindN[1][4][1] / 1000);
                gBProfNs = gBProfLines = gBProfRuns = 0;
                memset(gBgKindN, 0, sizeof gBgKindN);
            }
        }
#if HOST_VITA
        {   // what is seen as stutter: refreshes of the screen that showed the same picture again
            extern int gVitaGpuProbed;
            extern uint64_t gVitaGpu3dNs, gVitaGpuComposeNs, gVitaGpuPresentNs;
            if (__atomic_exchange_n(&gVitaGpuProbed, 0, __ATOMIC_ACQUIRE)) {
                fprintf(stderr, "gpu (one frame, the GPU waited for after each pass): 3d %llu us, compose %llu us, "
                        "present %llu us\n", (unsigned long long)(gVitaGpu3dNs / 1000),
                        (unsigned long long)(gVitaGpuComposeNs / 1000), (unsigned long long)(gVitaGpuPresentNs / 1000));
                    extern void vita_gl_gc_times(uint64_t *one, uint64_t *tiles, int *fastTiles, int *usingTiles);
                    uint64_t one, tiles; int ft, u;
                    vita_gl_gc_times(&one, &tiles, &ft, &u);
                    if (one) fprintf(stderr, "gpu compose: one quad %llu us, tiles %llu us (%d of 432 tiles the fast program) - "
                                     "using %s\n", (unsigned long long)(one / 1000), (unsigned long long)(tiles / 1000), ft,
                                     u ? "tiles" : "one quad");
                    extern void vita_gl_gc_why(int w[8], float f[4]);
                    int w[8];
                    float f[4];
                    vita_gl_gc_why(w, f);
                    if (one) fprintf(stderr, "gpu compose frame: 3D %.0f, logo %.0f, boot mirror %.0f, prompt shift %.0f\n",
                                     f[0], f[1], f[2], f[3]);
                    if (one) fprintf(stderr, "gpu compose tiles, lines (of 192): whole frame %d, effects %d, iris %d, "
                                     "overlay/clock %d, 2D across %d, both sides %d, no 3D %d, 3D shifted %d\n",
                                     w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
            }
            {
                extern void vita_gl_gc_stats(unsigned *frames, unsigned *tiled, unsigned *fast);
                unsigned fr, ti, fa;
                vita_gl_gc_stats(&fr, &ti, &fa);
                if (fr) fprintf(stderr, "compose: tiles in %u of %u frames (on average %u of 432 tiles the fast program)\n",
                                ti, fr, fa / fr);
            }
            extern unsigned gVitaRepeats;
            fprintf(stderr, "display: %u refreshes repeated the previous picture (last 60 frames)\n",
                    __atomic_exchange_n(&gVitaRepeats, 0, __ATOMIC_RELAXED));
        }
#endif
        {
            unsigned long long bands = __atomic_exchange_n(&sWBandsNs, 0, __ATOMIC_RELAXED) / 60000;
            unsigned long long pres = __atomic_exchange_n(&sWPresentNs, 0, __ATOMIC_RELAXED) / 60000;
#if HOST_VITA
            extern unsigned long long vita_gl3d_stats(char *buf, int n);
            char g3[320];
            vita_gl3d_stats(g3, sizeof g3);
#else
            const char *g3 = "";
#endif
            fprintf(stderr, "render thread split (avg/drawn frame): 2d bands %llu us, present %llu us%s\n", bands, pres, g3);
            fprintf(stderr, "render thread longest frame %llu us%s: 3d %llu us (texture sync %llu us, texture decode+upload %llu us "
                    "for %llu textures), 2d bands %llu us (waited %llu us after the 3d), present %llu us (swap %llu us)\n",
                    sWWorst, sWWorstBlack ? " (screen faded to black)" : "", sWWorst3d, sWWorstSync, sWWorstTex,
                    sWWorstUploads, sWWorstBands, sWWorstWait2d, sWWorstPresent, sWWorstSwap);
            sWWorst = 0;
            fprintf(stderr, "game thread hand-off split (avg/frame): wait for render thread %llu us, VRAM snapshot %llu us,"
                    " 2d prepass %llu us (of which hblank IRQs %llu us), geometry commit (incl. wait for GX worker) %llu us\n",
                    (unsigned long long)(sHoWaitRtNs / 60000), (unsigned long long)(sHoSnapNs / 60000),
                    (unsigned long long)(sRender2dNs / 60000), (unsigned long long)(sHblankNs / 60000),
                    (unsigned long long)(sHoCommitNs / 60000));
            sHoWaitRtNs = sHoSnapNs = sHoCommitNs = sHblankNs = 0;
        }
        sRenderSkipped = 0;
        sRender2dNs = sRender3dNs = 0;
        sRenderFrames = 0;
    }
}

void render_frame(void) {
    __atomic_fetch_add(&sVblankCount, 1, __ATOMIC_RELAXED);
    if (render_thread_applicable()) {
        sLastMt = 1;
        render_frame_mt();
        return;
    }
    render_thread_sync();                   // a worker frame may still be in flight; finish it before a sync frame
    gpu2d_set_defer_bands(0);
    sLastMt = 0;
    render_frame_sync();
}

static void render_frame_sync(void) {
    const u32 *lines3d = s3d;
    if (sGpu2d) {
        if (sGpuDisplay) lines3d = NULL;
        else if (sOwn3d) lines3d = gpu3d_lines();
        else melon_3d_lines(s3d);
    }
    if (sMelon) melon_frame();
    if (sGpu2d || sGpu3d) host_vram_flush();
    if (sGpu2d) {
        Gpu2dFrameInput in;
        for (int i = 0; i < 9; i++) in.bank[i] = host_vram_bank(i);
        in.palette = gHostPltt;
        in.oam = gHostOam;
        in.lines3d = lines3d;
        in.lines3dTop = sOwn3d && !sGpuDisplay && lines3d ? gpu3d_lines_top() : NULL;
        in.topRows = sOwn3d && !sGpuDisplay ? gpu3d_top_rows() : 192;
        in.hblank = sOwn ? render_hblank : 0;
        in.hblankContext = 0;
        in.defer3d = sGpuDisplay;
        in.gpu = sGpu2dMode;
        uint64_t t2d = host_plat_time_ns();
        gpu2d_render_frame(&in);
        sRender2dNs += host_plat_time_ns() - t2d;
        if (sGpuDisplay) {
            const u32 *deferred[2] = {gpu2d_deferred(0), gpu2d_deferred(1)};
            const u32 *lines[2] = {gpu2d_deferred_lines(0), gpu2d_deferred_lines(1)};
            if (sGpu2dMode) wdisp_frame_2d(gpu2d_gpu_frame(), in.bank, lines, gpu3d_wgpu_picture(), sScale3d);
            else wdisp_frame(deferred, lines, gpu3d_wgpu_picture(), sScale3d);
            if (sGpu2dMode == 2 && !sSelftest) render_check_gpu2d();
        }
    }
    if (sGpu2d && sOwn3d) {
        extern int gHostCanvasRows;
        int aBottom = gpu2d_main_on_bottom();
        int above = gHostCanvasRows - (aBottom ? 192 : 384);
        gpu3d_set_tall(gpu2d_tall_3d_ok() && above > 0, above);
    }
    if (sGpu3d) {
        uint64_t t3d = host_plat_time_ns();
        gpu3d_run_frame();
        sRender3dNs += host_plat_time_ns() - t3d;
    }
    if (++sRenderFrames >= 60) {
        fprintf(stderr, "render: 2d %llu us, 3d %llu us (avg/frame)\n",
                (unsigned long long)(sRender2dNs / 60000), (unsigned long long)(sRender3dNs / 60000));
        sRender2dNs = sRender3dNs = 0;
        sRenderFrames = 0;
    }
    if (sCompare && !sSelftest) {
        render_compare("frame", host_frame());
        render_compare_3d("frame", host_frame());
    }
}

const u32 *render_screen(int screen) {
    if (sGpuDisplay) return wdisp_screens() + 256 * 192 * (screen & 1);
    return sOwn ? gpu2d_screen(screen) : melon_screen(screen);
}
int render_on_gpu(void) { return sGpuDisplay; }

void render_exit_report(void) {
    if (sCompare) {
        fprintf(stderr, "render-compare: %llu frames compared, %llu differ", (unsigned long long)sCompared,
                (unsigned long long)sDiffering);
        if (sDiffering) fprintf(stderr, " (the first: %llu)", (unsigned long long)sFirstDiffering);
        fprintf(stderr, "\nrender-compare: 3D: %llu reads compared, %llu differ", (unsigned long long)sReads3d,
                (unsigned long long)sReadsDiffering);
        if (sReadsDiffering) fprintf(stderr, " (the first in frame %llu)", (unsigned long long)sFirstReadDiffering);
        fprintf(stderr, "; %llu polygon lists compared, %llu differ", (unsigned long long)sPolyFrames,
                (unsigned long long)sPolyDiffering);
        if (sPolyDiffering) fprintf(stderr, " (the first: %llu)", (unsigned long long)sFirstPolyDiffering);
        fprintf(stderr, "; %llu pictures compared, %llu differ", (unsigned long long)sPix3dFrames,
                (unsigned long long)sPix3dDiffering);
        if (sPix3dDiffering) fprintf(stderr, " (the first: %llu)", (unsigned long long)sFirstPix3dDiffering);
        fprintf(stderr, "\nrender-compare: 3D: %llu polygons compared;", (unsigned long long)sPolysCompared);
        for (int k = 0; k < PK_N; k++)
            if (sPolyKind[k][0] || sPolyKind[k][1])
                fprintf(stderr, " %s differs: %llu (+%llu clipped)", kPolyKinds[k], (unsigned long long)sPolyKind[k][0],
                        (unsigned long long)sPolyKind[k][1]);
        fprintf(stderr, "\n");
    }
    if (sGpu2dMode == 2)
        fprintf(stderr, "gpu2d-check: %llu frames checked, %llu differ\n", (unsigned long long)sGpu2dChecked,
                (unsigned long long)sGpu2dDiffering);
    if (sGpu2d && getenv("ACWW_GPU2D_FEATURES")) gpu2d_print_features();
    if (sGpu3d && getenv("ACWW_GPU3D_FEATURES")) gpu3d_print_features();
}
