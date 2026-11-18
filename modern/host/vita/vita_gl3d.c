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

static int sEnabled;     // g3w_init called (gpu3d_use_wgpu succeeded)
static int sReady;       // GL objects created (lazy, after the vitaGL context is up)
static GLuint sFbo[2], sColorTex[2], sDepthRb[2];
static int sPrevDraw = -1;   // FBO index drawn into last call, read back (non-stalling) this call; -1 = none yet

// --- texture cache: decode each unique DS texture (by texparam/texpal) to an RGBA GL texture once ------------
#define TEXCACHE 256
static struct { uint32_t tp, pal; GLuint tex; int used, w, h; } sTexCache[TEXCACHE];
static uint32_t sDecodeBuf[1024 * 1024];   // max DS texture 1024x1024
// The flattened DS texture/palette windows are rebuilt from VRAM every frame. Keep a
// snapshot so edits or bank remaps invalidate decoded GL textures instead of showing stale data.
#define IMG_BYTES (4 * 0x20000)
#define PAL_BYTES (8 * 0x4000)
static u8 sImgSnapshot[IMG_BYTES], sPalSnapshot[PAL_BYTES];
static int sVramSnapshotValid;
static int sLastTexSlot = -1;

static void sync_texture_cache(const u8 *img, const u8 *pal) {
    if (!img || !pal) return;
    if (!sVramSnapshotValid || memcmp(img, sImgSnapshot, IMG_BYTES) ||
        memcmp(pal, sPalSnapshot, PAL_BYTES)) {
        memcpy(sImgSnapshot, img, IMG_BYTES);
        memcpy(sPalSnapshot, pal, PAL_BYTES);
        for (int i = 0; i < TEXCACHE; i++) sTexCache[i].used = 0;
        sLastTexSlot = -1;
        sVramSnapshotValid = 1;
    }
}

static GLint ds_wrap(int repeat, int flip) {
    return flip ? GL_MIRRORED_REPEAT : (repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
}

static GLuint tex_for(uint32_t texparam, uint32_t texpal) {
    if ((texparam >> 26 & 7) == 0) return 0;   // format 0 == no texture
    if (sLastTexSlot >= 0 && sTexCache[sLastTexSlot].used &&
        sTexCache[sLastTexSlot].tp == texparam && sTexCache[sLastTexSlot].pal == texpal)
        return sTexCache[sLastTexSlot].tex;
    int slot = (int)((texparam * 2654435761u ^ texpal * 40503u) & (TEXCACHE - 1));
    int empty = -1;
    for (int i = 0; i < TEXCACHE; i++) {       // probe for a match or the first free slot
        int s = (slot + i) & (TEXCACHE - 1);
        if (!sTexCache[s].used) { if (empty < 0) empty = s; continue; }
        if (sTexCache[s].tp == texparam && sTexCache[s].pal == texpal) {
            sLastTexSlot = s;
            return sTexCache[s].tex;
        }
    }
    int w, h;
    gpu3d_texture_size(texparam, &w, &h);
    if (w <= 0 || h <= 0 || (long)w * h > (long)(sizeof sDecodeBuf / 4)) return 0;
    gpu3d_decode_texture(texparam, texpal, sDecodeBuf);
    int s = empty >= 0 ? empty : slot;         // insert at a free slot; evict only if the cache is full
    if (!sTexCache[s].tex) glGenTextures(1, &sTexCache[s].tex);
    glBindTexture(GL_TEXTURE_2D, sTexCache[s].tex);
    if (sTexCache[s].w == w && sTexCache[s].h == h)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, sDecodeBuf);
    else
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, sDecodeBuf);
    sTexCache[s].w = w; sTexCache[s].h = h;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, ds_wrap(texparam >> 16 & 1, texparam >> 18 & 1));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, ds_wrap(texparam >> 17 & 1, texparam >> 19 & 1));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    sTexCache[s].tp = texparam; sTexCache[s].pal = texpal; sTexCache[s].used = 1;
    sLastTexSlot = s;
    return sTexCache[s].tex;
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, VW, VH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindRenderbuffer(GL_RENDERBUFFER, sDepthRb[i]);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, VW, VH);
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
void *g3w_picture(void) { return 0; }   // readback path only (out buffer), no shared picture

#define BATCH_V 65536   // capacity of the batched 3D vertex buffers (~2400 polys expanded to triangles/frame)

static const G3RenderState *sSortR;
static int cmp_tex(const void *a, const void *b) {
    const G3Polygon *pa = sSortR->poly[*(const int *)a], *pb = sSortR->poly[*(const int *)b];
    uint64_t ka = (uint64_t)pa->texparam << 32 | pa->texpal;
    uint64_t kb = (uint64_t)pb->texparam << 32 | pb->texpal;
    if (ka != kb) return ka < kb ? -1 : 1;
    return 0;
}

void g3w_render(const G3RenderState *r, const u8 *img, const u8 *pal, u32 *out) {
    if (!sEnabled || !out) return;
    lazy_init();
    sync_texture_cache(img, pal);

    // Deferred readback: read back the PREVIOUS frame's FBO into `out` (DS 3D line format) before submitting this
    // frame. By now the GPU has long finished those draws (a whole frame has passed), so glReadPixels does not
    // stall - the ~18ms the CPU used to spend waiting for the GPU is gone. Costs one frame of extra 3D latency.
    if (sPrevDraw >= 0) {
        glBindFramebuffer(GL_FRAMEBUFFER, sFbo[sPrevDraw]);
        static u8 rb[VW * VH * 4];
        glReadPixels(0, 0, VW, VH, GL_RGBA, GL_UNSIGNED_BYTE, rb);
        for (int y = 0; y < VH; y++) {
            const u32 *src = (const u32 *)(rb + (VH - 1 - y) * VW * 4);
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

    int draw = (sPrevDraw + 1) & 1;   // submit this frame to the other FBO
    glBindFramebuffer(GL_FRAMEBUFFER, sFbo[draw]);
    glViewport(0, 0, VW, VH);
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
    static float bPos[BATCH_V][4], bCol[BATCH_V][4], bUv[BATCH_V][2];
    int bn = 0, haveState = 0, curBlend = -1;
    GLuint curTex = 0;
#define GL3D_FLUSH() do { if (bn) { glVertexPointer(4, GL_FLOAT, 0, bPos); glColorPointer(4, GL_FLOAT, 0, bCol); \
        glTexCoordPointer(2, GL_FLOAT, 0, bUv); glDrawArrays(GL_TRIANGLES, 0, bn); bn = 0; } } while (0)
    // Render opaque polygons sorted by texture (same texture batches into one draw, cutting GXM texture-bind
    // flushes - the dominant 3D render cost when the scene interleaves many textures). Depth test resolves the
    // opaque image regardless of draw order. Translucent polygons follow in submission order (order matters).
    static int sOrder[8192];
    int nOrd = 0;
    for (int k = 0; k < r->npoly; k++) if (r->poly[k]->nv >= 3 && !r->poly[k]->translucent && nOrd < 8192) sOrder[nOrd++] = k;
    sSortR = r;
    if (nOrd > 1) qsort(sOrder, nOrd, sizeof sOrder[0], cmp_tex);
    for (int k = 0; k < r->npoly; k++) if (r->poly[k]->nv >= 3 && r->poly[k]->translucent && nOrd < 8192) sOrder[nOrd++] = k;
    for (int oi = 0; oi < nOrd; oi++) {
        int k = sOrder[oi];
        const G3Polygon *p = r->poly[k];
        if (p->nv < 3) continue;
        GLuint tex = tex_for(p->texparam, p->texpal);
        int blend = p->translucent ? 1 : 0;
        if (!haveState || tex != curTex || blend != curBlend || bn + (p->nv - 2) * 3 > BATCH_V) {
            GL3D_FLUSH();
            if (tex) { glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, tex); } else glDisable(GL_TEXTURE_2D);
            if (blend) { glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE); }
            else { glDisable(GL_BLEND); glDepthMask(GL_TRUE); }
            curTex = tex; curBlend = blend; haveState = 1;
        }
        int tw = 8 << (p->texparam >> 20 & 7), th = 8 << (p->texparam >> 23 & 7);
        float vp[16][4], vc[16][4], vu[16][2];
        for (int i = 0; i < p->nv; i++) {
            const G3Vertex *v = p->v[i];
            // DS clip-space position (pos[]=NDC*w); GL does the divide/viewport/depth and clips the frustum.
            // v->color is the 9-bit interpolation value, so >>3 to the displayed 6-bit channel.
            vp[i][0] = (float)v->pos[0]; vp[i][1] = (float)v->pos[1]; vp[i][2] = (float)v->pos[2]; vp[i][3] = (float)v->pos[3];
            vc[i][0] = (v->color[0] >> 3) / 63.0f; vc[i][1] = (v->color[1] >> 3) / 63.0f; vc[i][2] = (v->color[2] >> 3) / 63.0f;
            vc[i][3] = p->translucent ? (p->attr >> 16 & 31) / 31.0f : 1.0f;
            vu[i][0] = tw ? (float)v->tex[0] / 16.0f / tw : 0.0f;
            vu[i][1] = th ? (float)v->tex[1] / 16.0f / th : 0.0f;
        }
        for (int i = 1; i + 1 < p->nv; i++) {   // fan -> triangles: (0, i, i+1)
            const int idx[3] = {0, i, i + 1};
            for (int j = 0; j < 3; j++, bn++) {
                int s = idx[j];
                bPos[bn][0] = vp[s][0]; bPos[bn][1] = vp[s][1]; bPos[bn][2] = vp[s][2]; bPos[bn][3] = vp[s][3];
                bCol[bn][0] = vc[s][0]; bCol[bn][1] = vc[s][1]; bCol[bn][2] = vc[s][2]; bCol[bn][3] = vc[s][3];
                bUv[bn][0] = vu[s][0]; bUv[bn][1] = vu[s][1];
            }
        }
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
}

// Enable predicate used by render_init: GPU path on, and the shader compiler reachable (else fall back to the
// software rasterizer rather than crash in vitaGL's runtime shader compile).
int vita_gl3d_available(void) {
    return vita_gl_enabled() && vita_gl_have_shader_compiler();
}
