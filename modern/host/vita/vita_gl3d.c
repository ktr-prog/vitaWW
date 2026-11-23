// GPU rasterizer for the DS 3D engine, via vitaGL - the Vita implementation of the g3w_* renderer seam that the
// software rasterizer (gpu3d_raster.c) and the desktop/iOS WebGPU backend (gpu3d_wgpu.c) share. When active,
// g3_render() hands us the transformed/lit polygons (gG3Render) plus the decoded VRAM, we draw them as GPU
// triangles into an offscreen target, and read the result back into the DS-format colour buffer the 2D
// compositor already consumes. This moves the ~100ms/frame software 3D rasterization onto the GPU.
//
// Shaders: vitaGL compiles its pipeline shaders at runtime with the on-device Cg compiler (libshacccg.suprx),
// exactly as the PC build's OpenGL driver compiles shaders at startup. The compiler is loaded from inside the
// VPK (app0:) when bundled, else from ur0:/data - so a VPK that ships the compiler is fully self-contained.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vitaGL.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#include "../gpu3d/gpu3d_internal.h"
#include "../gpu3d/gpu3d.h"
#include "vita.h"

#define VW 256
#define VH 192
// The colour buffers are the Vita's screen size: with the GPU compose the 3D is drawn at 960x544, the whole screen
// (the DS's view widened to the screen's 30:17, the canvas's GC_W columns), one 3D pixel per screen pixel; the compose takes its
// DS-resolution sample from it and the present puts the full-resolution picture wherever the 3D shows unchanged
// (vita_gl.c). The readback path (no GPU compose) draws 256x192 in the corner.
#define FBO_W 960
#define FBO_H 544

static int sEnabled;     // g3w_init called (gpu3d_use_wgpu succeeded)
static int sReady;       // GL objects created (lazy, after the vitaGL context is up)
static GLuint sFbo[2], sColorTex[2], sDepthRb[2];
static int sPrevDraw = -1;   // FBO index drawn into last call, read back (non-stalling) this call; -1 = none yet

// Timing split of g3w_render, reported on the "render thread split" line.
extern uint64_t host_plat_time_ns(void);
unsigned long long gG3FrameTexNs, gG3FrameUploads, gG3FrameSyncNs;
enum { TEX_REFRESH_BUDGET_NS = 2500000 };
static unsigned long long sStRefreshes, sStDeferred;   // the last frame's (render.c: longest frame)
static unsigned long long sStReadNs, sStSubmitNs, sStTexNs, sStTexUploads, sStCalls, sStDraws, sStGlNs, sStResolveNs, sStSyncNs;
unsigned long long vita_gl3d_stats(char *buf, int n) {
    unsigned long long c = sStCalls ? sStCalls : 1;
    extern unsigned long long gVitaComposeNs, gVitaSwapNs, gVitaPresents;
    unsigned long long pc = gVitaPresents ? gVitaPresents : 1;
    snprintf(buf, (size_t)n, "; 3d: texture sync %llu us, draw submit %llu us (of which textures %llu us [decode+upload %llu us, %llu uploads/frame, %llu of them refreshes, %llu refreshes deferred in 60 frames], glDrawElements %llu us, %llu draws/frame); "
             "present: compose %llu us, swap %llu us",
             sStSyncNs / c / 1000, sStSubmitNs / c / 1000, sStResolveNs / c / 1000, sStTexNs / c / 1000, sStTexUploads / c, sStRefreshes / c, sStDeferred, sStGlNs / c / 1000, sStDraws / c, gVitaComposeNs / pc / 1000, gVitaSwapNs / pc / 1000);
    sStDraws = sStGlNs = sStResolveNs = sStSyncNs = sStRefreshes = sStDeferred = 0;
    gVitaComposeNs = gVitaSwapNs = gVitaPresents = 0;
    sStReadNs = sStSubmitNs = sStTexNs = sStTexUploads = sStCalls = 0;
    return c;
} 
// --- texture cache --------------------------------------------------------------------------------------------
// Each DS texture (texparam/texpal) is decoded to an RGBA GL texture once and reused while its VRAM is unchanged.
// DS games rewrite texture VRAM in place (animated textures every few frames, new data on scene loads), so every
// frame the flat texture/palette VRAM is compared with last frame's in 1 KB blocks, and entries whose texel,
// 4x4-index or palette bytes touch a changed block are re-decoded on their next use. Entries unused for a while
// are freed; an entry used in the current frame is never evicted (its GL texture may still be referenced by
// draws queued this frame).
enum { TC_SLOTS = 2048, TC_MAX = 1024, TC_BLK = 1024, TC_IMG = 0x80000, TC_PAL = 0x20000 };
typedef struct { uint32_t tp, pal, lastUsed; GLuint tex; uint8_t used, stale; } TexEnt;
static TexEnt sTex[TC_SLOTS];
static int sTexCount;
static int sLastTexSlot = -1;   // consecutive polygons (sorted by texture) usually hit the same entry
static uint32_t sTexFrame = 1;
static u8 sDirtyImg[TC_IMG / TC_BLK], sDirtyPal[TC_PAL / TC_BLK];
static uint32_t sDecodeBuf[1024 * 1024];   // max DS texture 1024x1024

static GLint ds_wrap(int repeat, int flip) {
    return flip ? GL_MIRRORED_REPEAT : (repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
}

static int range_dirty(const u8 *dirty, u32 nblk, u32 base, u32 size) {
    if (!size) return 0;
    u32 first = base / TC_BLK, n = (base % TC_BLK + size + TC_BLK - 1) / TC_BLK;
    if (n > nblk) n = nblk;
    for (u32 i = 0; i < n; i++)
        if (dirty[(first + i) % nblk]) return 1;
    return 0;
}
// VRAM bytes a texture reads: texels, the 4x4-compressed palette indices, and its palette.
static int tex_touches_dirty(uint32_t tp, uint32_t pp) {
    static const u8 bpp[8] = {0, 8, 2, 4, 8, 2, 8, 16};
    static const u32 palBytes[8] = {0, 64, 8, 32, 512, 0x10000, 16, 0};
    int fmt = tp >> 26 & 7;
    u32 w = 8u << (tp >> 20 & 7), h = 8u << (tp >> 23 & 7);
    u32 a = ((tp & 0xffff) << 3) & (TC_IMG - 1);
    if (range_dirty(sDirtyImg, TC_IMG / TC_BLK, a, w * h * bpp[fmt] / 8)) return 1;
    if (fmt == 5) {
        u32 x = (0x20000 + (a & 0x1ffff) / 2 + (a >= 0x40000 ? 0x10000 : 0)) & (TC_IMG - 1);
        if (range_dirty(sDirtyImg, TC_IMG / TC_BLK, x, w * h / 8)) return 1;
    }
    u32 pb = (fmt == 2 ? pp << 3 : pp << 4) & (TC_PAL - 1);
    return range_dirty(sDirtyPal, TC_PAL / TC_BLK, pb, palBytes[fmt]);
}
static void tex_upload(TexEnt *e) {
    uint64_t t0 = host_plat_time_ns();
    sStTexUploads++;
    gG3FrameUploads++;
    int w, h;
    gpu3d_texture_size(e->tp, &w, &h);
    if (w <= 0 || h <= 0 || (long)w * h > (long)(sizeof sDecodeBuf / 4)) return;
    gpu3d_decode_texture(e->tp, e->pal, sDecodeBuf);
    if (!e->tex) glGenTextures(1, &e->tex);
    glBindTexture(GL_TEXTURE_2D, e->tex);
    // vitaGL/GXM can still be sampling this texture from the previous queued frame. Re-specifying the image
    // gives the driver a fresh backing store; glTexSubImage2D on the reused storage caused alternating
    // stale/new texture contents on hardware.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, sDecodeBuf);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, ds_wrap(e->tp >> 16 & 1, e->tp >> 18 & 1));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, ds_wrap(e->tp >> 17 & 1, e->tp >> 19 & 1));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    sStTexNs += host_plat_time_ns() - t0;
    gG3FrameTexNs += host_plat_time_ns() - t0;
}
static int tex_slot(uint32_t tp, uint32_t pp) {
    int h = (int)((tp * 2654435761u ^ pp * 40503u) & (TC_SLOTS - 1));
    while (sTex[h].used && !(sTex[h].tp == tp && sTex[h].pal == pp)) h = (h + 1) & (TC_SLOTS - 1);
    return h;
}
// Free entries not used since frame `keepFrom` and rebuild the probe table without them.
static void tex_sweep(uint32_t keepFrom) {
    static TexEnt keep[TC_SLOTS];
    int n = 0;
    for (int i = 0; i < TC_SLOTS; i++) {
        if (!sTex[i].used) continue;
        if (sTex[i].lastUsed >= keepFrom) keep[n++] = sTex[i];
        else if (sTex[i].tex) glDeleteTextures(1, &sTex[i].tex);
    }
    memset(sTex, 0, sizeof sTex);
    for (int i = 0; i < n; i++) sTex[tex_slot(keep[i].tp, keep[i].pal)] = keep[i];
    sTexCount = n;
    sLastTexSlot = -1;
}
// Once per frame, before any draw: mark entries whose VRAM changed (g3_tex_setup flags the changed 1 KB blocks
// while it updates the flat texture/palette image) and sweep old entries.
extern u8 gG3TexDirtyImg[TC_IMG / TC_BLK], gG3TexDirtyPal[TC_PAL / TC_BLK];
static void tex_frame_begin(const u8 *img, const u8 *pal) {
    (void)img; (void)pal;
    sTexFrame++;
    int any = 0;
    for (u32 b = 0; b < TC_IMG / TC_BLK && !any; b++) any = gG3TexDirtyImg[b];
    for (u32 b = 0; b < TC_PAL / TC_BLK && !any; b++) any = gG3TexDirtyPal[b];
    if (any) {
        memcpy(sDirtyImg, gG3TexDirtyImg, sizeof sDirtyImg);
        memcpy(sDirtyPal, gG3TexDirtyPal, sizeof sDirtyPal);
        for (int i = 0; i < TC_SLOTS; i++)
            if (sTex[i].used && !sTex[i].stale && tex_touches_dirty(sTex[i].tp, sTex[i].pal)) sTex[i].stale = 1;
    }
    if (sTexCount > TC_MAX * 3 / 4) tex_sweep(sTexFrame > 120 ? sTexFrame - 120 : 0);
}
static GLuint tex_for(uint32_t texparam, uint32_t texpal) {
    if ((texparam >> 26 & 7) == 0) return 0;   // format 0 == no texture
    if (sLastTexSlot >= 0) {
        TexEnt *l = &sTex[sLastTexSlot];
        if (l->used && !l->stale && l->tp == texparam && l->pal == texpal) { l->lastUsed = sTexFrame; return l->tex; }
    }
    int s = tex_slot(texparam, texpal);
    TexEnt *e = &sTex[s];
    if (e->used) {
        // A texture whose VRAM changed is decoded again - unless this frame already spent its upload budget: then
        // it is drawn as it was and refreshed on a following frame (a burst of changed textures no longer stalls
        // one frame; textures never seen before are always uploaded).
        if (e->stale && (gG3FrameTexNs < TEX_REFRESH_BUDGET_NS || !e->tex)) { tex_upload(e); e->stale = 0; sStRefreshes++; }
        else if (e->stale) sStDeferred++;
        e->lastUsed = sTexFrame;
        sLastTexSlot = s;
        return e->tex;
    }
    if (sTexCount >= TC_MAX) {                 // full: drop everything not used this frame, then re-probe
        tex_sweep(sTexFrame);
        if (sTexCount >= TC_MAX) return 0;
        s = tex_slot(texparam, texpal);
        e = &sTex[s];
    }
    e->tp = texparam; e->pal = texpal; e->used = 1; e->stale = 0; e->lastUsed = sTexFrame; e->tex = 0;
    sTexCount++;
    tex_upload(e);
    sLastTexSlot = s;
    return e->tex;
}

static void load_shader_compiler(void) {
    // Pre-load libshacccg from the bundled/game-folder/system paths. vitaShaRK caches the module on first success,
    // so vitaGL's own lazy shark_init(NULL) then becomes a no-op and the GPU pipeline shaders compile. This is
    // what lets a VPK that carries libshacccg (app0:) compile shaders itself, with no separate user install.
    extern int shark_init(const char *path);
    extern const char *const kShaderCompilerPaths[];
    for (int i = 0; kShaderCompilerPaths[i]; i++)
        if (shark_init(kShaderCompilerPaths[i]) >= 0) {
            fprintf(stderr, "vita_gl3d: shader compiler loaded from %s\n", kShaderCompilerPaths[i]);
            return;
        }
    fprintf(stderr, "vita_gl3d: no shader compiler found (GPU 3D will not compile shaders)\n");
}

static void lazy_init(void) {
    if (sReady) return;
    load_shader_compiler();
    // Two FBOs: draws are submitted to one while the previous one is read back. Deferring the readback by a frame
    // lets the GPU finish this frame's draws during the CPU phase, so glReadPixels no longer stalls.
    glGenFramebuffers(2, sFbo);
    glGenTextures(2, sColorTex);
    glGenRenderbuffers(2, sDepthRb);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, sColorTex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, FBO_W, FBO_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindRenderbuffer(GL_RENDERBUFFER, sDepthRb[i]);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, FBO_W, FBO_H);
        glBindFramebuffer(GL_FRAMEBUFFER, sFbo[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sColorTex[i], 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, sDepthRb[i]);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    sReady = 1;
}

// g3w_* interface (replaces the WebGPU backend on Vita) -------------------------------------------------------
int g3w_init(void *device, void *queue, int scale) {
    (void)device; (void)queue; (void)scale;
    sEnabled = 1;      // GL objects are created lazily on first render, once the vitaGL context exists
    extern int gG3ClipSpaceOut, gG3GpuXform;
    gG3ClipSpaceOut = 1;   // consume clip-space pos[] directly; skip the software screen/z divides in poly_screen
    gG3GpuXform = 0;        // CPU transforms to clip space (cheap render: one identity projection, all draws
                           // batch together with no per-object matrix loads). The GPU still clips: the CPU
                           // Sutherland-Hodgman clip is skipped (gG3ClipSpaceOut) and GL clips the raw polys.
                           // GPU vertex transform was a net loss - the per-object glLoadMatrixf flushes cost more
                           // GXM render time than the ~13ms CPU transform they removed.
    return 1;
}
void *g3w_picture(void) { return 0; }
// The colour texture of the last frame drawn (GPU compose samples it), 0 before the first.
unsigned vita_gl3d_frame_tex(void) { return sReady && sPrevDraw >= 0 ? sColorTex[sPrevDraw] : 0; }   // readback path only (out buffer), no shared picture

#define BATCH_V 0xC000  // vertices per draw: vitaGL's glDrawArrays index limit (MAX_IDX_NUMBER)

void g3w_render(const G3RenderState *r, const u8 *img, const u8 *pal, u32 *out) {
    if (!sEnabled || !out) return;
    lazy_init();
    uint64_t tSync = host_plat_time_ns();
    gG3FrameTexNs = gG3FrameUploads = 0;
    // Group the polygons by texture once (a hash of texparam/texpal, O(n)): each distinct texture is synced,
    // resolved and bound once per frame, and the opaque polygons are drawn group by group (in the order each
    // texture first appears) - no qsort with a pointer-chasing comparator, no per-polygon cache lookup.
    enum { GH = 16384 };   // > MAX_POLYGONS: the table never fills
    static uint64_t gKey[GH];
    static uint32_t gStamp[GH], sGStamp;
    static int gId[GH];
    static uint64_t grpKey[MAX_POLYGONS];
    static int grpCount[MAX_POLYGONS], grpPos[MAX_POLYGONS];
    static GLuint grpTex[MAX_POLYGONS];
    static int sPolyGrp[MAX_POLYGONS];
    int ngrp = 0;
    sGStamp++;
    if (!sGStamp) { memset(gStamp, 0, sizeof gStamp); sGStamp = 1; }
    for (int k = 0; k < r->npoly; k++) {
        const G3Polygon *p = r->poly[k];
        if (p->nv < 3) { sPolyGrp[k] = -1; continue; }
        uint64_t key = (p->texparam >> 26 & 7) ? ((uint64_t)p->texparam << 32 | p->texpal) : 0;
        uint32_t h = (uint32_t)(key ^ key >> 29 ^ key >> 47) * 0x9e3779b1u >> 18;
        while (gStamp[h] == sGStamp && gKey[h] != key) h = (h + 1) & (GH - 1);
        if (gStamp[h] != sGStamp) {
            gStamp[h] = sGStamp; gKey[h] = key; gId[h] = ngrp;
            grpKey[ngrp] = key; grpCount[ngrp] = 0; ngrp++;
        }
        int g = gId[h];
        sPolyGrp[k] = g;
        if (!p->translucent && !(p->attr & 0x4000)) grpCount[g]++;   // (depth-equal decals go after: below)
    }
    // lazy texture setup: sync the VRAM ranges of every texture drawn this frame (sets the change flags)
    for (int g = 0; g < ngrp; g++)
        if (grpKey[g]) gpu3d_tex_sync_for((uint32_t)(grpKey[g] >> 32), (uint32_t)grpKey[g]);
    tex_frame_begin(img, pal);   // re-decode only textures whose VRAM changed since last frame
    uint64_t tStart = host_plat_time_ns();
    sStSyncNs += tStart - tSync;
    gG3FrameSyncNs = tStart - tSync;
    sStCalls++;

    // Deferred readback: read back the PREVIOUS frame's FBO into `out` (DS 3D line format) before submitting this
    // frame. By now the GPU has long finished those draws (a whole frame has passed), so glReadPixels does not
    // stall - the ~18ms the CPU used to spend waiting for the GPU is gone. Costs one frame of extra 3D latency.
    extern int gVitaGc;   // GPU compose: the 2D/3D blend samples the colour buffer on the GPU, nothing to read
    if (gVitaGc) {
    } else if (sPrevDraw >= 0) {
        glBindFramebuffer(GL_FRAMEBUFFER, sFbo[sPrevDraw]);
        static u8 rb[VW * VH * 4];
        // (glReadPixels: a GPU transfer into main RAM measured slower here, ~9 ms vs ~4 ms)
        const u8 *rbs = rb;
        glReadPixels(0, 0, VW, VH, GL_RGBA, GL_UNSIGNED_BYTE, rb);
        for (int y = 0; y < VH; y++) {
            const u32 *src = (const u32 *)(rbs + (VH - 1 - y) * VW * 4);
            u32 *dst = out + y * VW;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            // Convert four RGBA pixels at a time. The readback buffer is tightly packed RGBA8;
            // the DS compositor wants 6-bit RGB plus the opaque 5-bit alpha marker.
            const uint32x4_t rgbMask = vdupq_n_u32(0x3f);
            const uint32x4_t alphaMask = vdupq_n_u32(0x1f);
            const uint32x4_t alphaOpaque = vdupq_n_u32(0x80);
            int x = 0;
            for (; x + 4 <= VW; x += 4) {
                uint32x4_t p = vld1q_u32(src + x);
                uint32x4_t red = vandq_u32(vshrq_n_u32(p, 2), rgbMask);
                uint32x4_t green = vandq_u32(vshrq_n_u32(p, 10), rgbMask);
                uint32x4_t blue = vandq_u32(vshrq_n_u32(p, 18), rgbMask);
                uint32x4_t alpha = vorrq_u32(vandq_u32(vshrq_n_u32(p, 27), alphaMask), alphaOpaque);
                uint32x4_t packed = vorrq_u32(red, vorrq_u32(vshlq_n_u32(green, 8),
                    vorrq_u32(vshlq_n_u32(blue, 16), vshlq_n_u32(alpha, 24))));
                vst1q_u32(dst + x, packed);
            }
            for (; x < VW; x++) {
                u32 p = src[x];
                dst[x] = ((p >> 2) & 0x3f) | (((p >> 10) & 0x3f) << 8) | (((p >> 18) & 0x3f) << 16) |
                         ((((p >> 27) & 0x1f) | 0x80u) << 24);
            }
#else
            for (int x = 0; x < VW; x++) {
                u32 p = src[x];
                dst[x] = ((p >> 2) & 0x3f) | (((p >> 10) & 0x3f) << 8) | (((p >> 18) & 0x3f) << 16) |
                         ((((p >> 27) & 0x1f) | 0x80u) << 24);
            }
#endif
        }
    } else {
        for (int i = 0; i < VW * VH; i++) out[i] = 0;   // first frame: no 3D rendered yet
    }

    uint64_t tRead = host_plat_time_ns();
    sStReadNs += tRead - tStart;
    int draw = (sPrevDraw + 1) & 1;   // submit this frame to the other FBO
    glBindFramebuffer(GL_FRAMEBUFFER, sFbo[draw]);
    { extern int gVitaGc; if (gVitaGc) glViewport(0, 0, FBO_W, FBO_H); else glViewport(0, 0, VW, VH); }
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glDisable(GL_CULL_FACE);   // polys arrive pre-culled and pre-clipped from the CPU geometry pipeline

    u32 cc = r->clearColor;
    float cr = (cc & 31) / 31.0f, cg = (cc >> 5 & 31) / 31.0f, cb = (cc >> 10 & 31) / 31.0f;
    glClearColor(cr, cg, cb, (r->clearAttrHi & 31) / 31.0f);
    glClearDepthf(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    // Discard transparent texels so the background shows through, matching the software rasterizer (which does
    // "if (!al) return"): alpha 0 is dropped always, and with DS alpha-test (dispcnt bit 2) anything <= alphaRef.
    // Without this, transparent parts of textured quads (billboards, foliage) fill with the opaque clear colour.
    float aref = (r->dispcnt & 4) ? (float)r->alphaRef / 31.0f : 0.0f;
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, aref);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    // Batched clip-space drawing. The CPU transforms verts to DS clip space (pos[]); the GPU does the perspective
    // divide/viewport/depth and the frustum clip (the CPU Sutherland-Hodgman clip is skipped). Because every
    // polygon uses the same identity projection, there is no per-object matrix load, so all polygons batch into
    // one glDrawArrays per run sharing texture+blend - ~2400 draw calls collapse to a handful. Polys arrive
    // pre-culled from the CPU facing test, so GL culling stays off.
    // Indexed batches: each polygon's 3-4 vertices once, colours as bytes, plus a 16-bit fan index list.
    static float bPos[BATCH_V][4], bUv[BATCH_V][2];
    static u8 bCol8[BATCH_V][4];
    static u16 bIdx[BATCH_V * 3];
    static u8 sCol8[64];
    static int col8Init;
    if (!col8Init) { col8Init = 1; for (int c = 0; c < 64; c++) sCol8[c] = (u8)((c * 255 + 31) / 63); }
    int bn = 0, bi = 0, haveState = 0, curBlend = -1;
    GLuint curTex = 0;
#define GL3D_FLUSH() do { \
        if (bi) { glVertexPointer(4, GL_FLOAT, 0, bPos); glColorPointer(4, GL_UNSIGNED_BYTE, 0, bCol8); \
            glTexCoordPointer(2, GL_FLOAT, 0, bUv); \
            uint64_t tg_ = host_plat_time_ns(); glDrawElements(GL_TRIANGLES, bi, GL_UNSIGNED_SHORT, bIdx); \
            sStGlNs += host_plat_time_ns() - tg_; sStDraws++; } \
        bn = bi = 0; } while (0)
    // Render opaque polygons sorted by texture (same texture batches into one draw, cutting GXM texture-bind
    // flushes - the dominant 3D render cost when the scene interleaves many textures). Depth test resolves the
    // opaque image regardless of draw order. Translucent polygons follow in submission order (order matters).
    static int sOrder[MAX_POLYGONS];
    int nOrd = 0;
    for (int g = 0; g < ngrp; g++) { grpPos[g] = nOrd; nOrd += grpCount[g]; }
    // Polygons with the DS "depth equal" test (attr bit 14: decals - ground patterns, shadows) are drawn on top of
    // the surface under them: after every other opaque polygon, in submission order, and pulled towards the camera
    // a little (the DS passes them within +-0x200 of the stored depth; below). Sorted in with the rest and with a plain
    // LEQUAL test, whether a decal won against the coplanar ground changed from frame to frame (flicker).
    int nPlain = 0;
    for (int k = 0; k < r->npoly; k++)
        if (sPolyGrp[k] >= 0 && !r->poly[k]->translucent && !(r->poly[k]->attr & 0x4000)) {
            sOrder[grpPos[sPolyGrp[k]]++] = k;
            nPlain++;
        }
    nOrd = nPlain;
    for (int k = 0; k < r->npoly; k++)
        if (sPolyGrp[k] >= 0 && !r->poly[k]->translucent && (r->poly[k]->attr & 0x4000)) sOrder[nOrd++] = k;
    for (int k = 0; k < r->npoly; k++)
        if (sPolyGrp[k] >= 0 && r->poly[k]->translucent) sOrder[nOrd++] = k;
    // Resolve every texture before drawing anything. A cache miss or refresh uploads a texture, which rebinds
    // GL_TEXTURE_2D; doing that inside the draw loop made the batch still pending for the previous texture draw
    // with the newly uploaded one (wrong textures for a frame whenever a texture was uploaded).
    uint64_t tRes = host_plat_time_ns();
    for (int g = 0; g < ngrp; g++) grpTex[g] = tex_for((uint32_t)(grpKey[g] >> 32), (uint32_t)grpKey[g]);
    sStResolveNs += host_plat_time_ns() - tRes;
    for (int oi = 0; oi < nOrd; oi++) {
        int k = sOrder[oi];
        const G3Polygon *p = r->poly[k];
        GLuint tex = grpTex[sPolyGrp[k]];
        int blend = p->translucent ? 1 : 0;
        int full = bn + p->nv > BATCH_V || bi + (p->nv - 2) * 3 > BATCH_V * 3;
        if (!haveState || tex != curTex || blend != curBlend || full) {
            GL3D_FLUSH();
            if (tex) { glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, tex); } else glDisable(GL_TEXTURE_2D);
            if (blend) { glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE); }
            else { glDisable(GL_BLEND); glDepthMask(GL_TRUE); }
            curTex = tex; curBlend = blend; haveState = 1;
        }
        int tw = 8 << (p->texparam >> 20 & 7), th = 8 << (p->texparam >> 23 & 7);
        // No divisions per vertex (the A9's VFP divide is slow and unpipelined), same floats as before:
        // colours come from a table built with the old expression; 1/(16*tw) and 1/(16*th) are exact powers of
        // two, so multiplying is bit-identical to dividing by 16 and then by tw/th; alpha is per polygon.
        const float su = 1.0f / (16.0f * tw), sv = 1.0f / (16.0f * th);
        const u8 a8 = p->translucent ? (u8)(((p->attr >> 16 & 31) * 255 + 15) / 31) : 255;
        // depth-equal: pulled forward by 2e-5 in NDC depth - far above the float error of two coplanar polygons'
        // interpolated depths, and (W-buffered, as the town is) a few hundredths of a unit at the usual distances
        const float zEq = (p->attr & 0x4000) ? 2e-5f : 0.0f;
        for (int i = 0; i < p->nv; i++) {
            const G3VHot *v = p->hv[0] ? p->hv[i] : (const G3VHot *)0;
            G3VHot tmp;
            if (!v) {   // (no lean-vertex pool memory): same fields from the full vertex
                const G3Vertex *f = p->v[i];
                memcpy(tmp.pos, f->pos, sizeof tmp.pos); memcpy(tmp.color, f->color, sizeof tmp.color);
                memcpy(tmp.tex, f->tex, sizeof tmp.tex);
                v = &tmp;
            }
            // DS clip-space position (pos[]=NDC*w); GL does the divide/viewport/depth and clips the frustum.
            // v->color is the 9-bit interpolation value, so >>3 to the displayed 6-bit channel.
            const int o = bn + i;
            bPos[o][0] = (float)v->pos[0]; bPos[o][1] = (float)v->pos[1]; bPos[o][2] = (float)v->pos[2] - zEq * (float)v->pos[3];
            bPos[o][3] = (float)v->pos[3];
            bCol8[o][0] = sCol8[(v->color[0] >> 3) & 63]; bCol8[o][1] = sCol8[(v->color[1] >> 3) & 63];
            bCol8[o][2] = sCol8[(v->color[2] >> 3) & 63]; bCol8[o][3] = a8;
            bUv[o][0] = (float)v->tex[0] * su;
            bUv[o][1] = (float)v->tex[1] * sv;
        }
        for (int i = 1; i + 1 < p->nv; i++) { bIdx[bi++] = (u16)bn; bIdx[bi++] = (u16)(bn + i); bIdx[bi++] = (u16)(bn + i + 1); }
        bn += p->nv;
    }
    GL3D_FLUSH();
#undef GL3D_FLUSH

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDepthMask(GL_TRUE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // The draws are now in flight on the GPU; this FBO is read back next frame (no stall), not now.
    sPrevDraw = draw;
    sStSubmitNs += host_plat_time_ns() - tRead;
}

// Enable predicate used by render_init: GPU path on, and the shader compiler reachable (else fall back to the
// software rasterizer rather than crash in vitaGL's runtime shader compile).
int vita_gl3d_available(void) {
    return vita_gl_enabled() && vita_gl_have_shader_compiler();
}
