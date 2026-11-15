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

#include "../gpu3d/gpu3d_internal.h"
#include "../gpu3d/gpu3d.h"
#include "vita.h"

#define VW 256
#define VH 192

static int sEnabled;     // g3w_init called (gpu3d_use_wgpu succeeded)
static int sReady;       // GL objects created (lazy, after the vitaGL context is up)
static GLuint sFbo, sColorTex, sDepthRb;

// --- texture cache: decode each unique DS texture (by texparam/texpal) to an RGBA GL texture once ------------
#define TEXCACHE 256
static struct { uint32_t tp, pal; GLuint tex; int used; } sTexCache[TEXCACHE];
static uint32_t sDecodeBuf[1024 * 1024];   // max DS texture 1024x1024

static GLint ds_wrap(int repeat, int flip) {
    return flip ? GL_MIRRORED_REPEAT : (repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
}

static GLuint tex_for(uint32_t texparam, uint32_t texpal) {
    if ((texparam >> 26 & 7) == 0) return 0;   // format 0 == no texture
    int slot = (int)((texparam * 2654435761u ^ texpal * 40503u) % TEXCACHE);
    int empty = -1;
    for (int i = 0; i < TEXCACHE; i++) {       // probe for a match or the first free slot
        int s = (slot + i) % TEXCACHE;
        if (!sTexCache[s].used) { if (empty < 0) empty = s; continue; }
        if (sTexCache[s].tp == texparam && sTexCache[s].pal == texpal) return sTexCache[s].tex;
    }
    int w, h;
    gpu3d_texture_size(texparam, &w, &h);
    if (w <= 0 || h <= 0 || (long)w * h > (long)(sizeof sDecodeBuf / 4)) return 0;
    gpu3d_decode_texture(texparam, texpal, sDecodeBuf);
    int s = empty >= 0 ? empty : slot;         // insert at a free slot; evict only if the cache is full
    if (!sTexCache[s].tex) glGenTextures(1, &sTexCache[s].tex);
    glBindTexture(GL_TEXTURE_2D, sTexCache[s].tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, sDecodeBuf);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, ds_wrap(texparam >> 16 & 1, texparam >> 18 & 1));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, ds_wrap(texparam >> 17 & 1, texparam >> 19 & 1));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    sTexCache[s].tp = texparam; sTexCache[s].pal = texpal; sTexCache[s].used = 1;
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
    glGenFramebuffers(1, &sFbo);
    glGenTextures(1, &sColorTex);
    glBindTexture(GL_TEXTURE_2D, sColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, VW, VH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenRenderbuffers(1, &sDepthRb);
    glBindRenderbuffer(GL_RENDERBUFFER, sDepthRb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, VW, VH);
    glBindFramebuffer(GL_FRAMEBUFFER, sFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sColorTex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, sDepthRb);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    sReady = 1;
}

// g3w_* interface (replaces the WebGPU backend on Vita) -------------------------------------------------------
int g3w_init(void *device, void *queue, int scale) {
    (void)device; (void)queue; (void)scale;
    sEnabled = 1;      // GL objects are created lazily on first render, once the vitaGL context exists
    extern int gG3ClipSpaceOut, gG3GpuXform;
    gG3ClipSpaceOut = 1;   // consume clip-space pos[] directly; skip the software screen/z divides in poly_screen
    gG3GpuXform = 0;        // CPU transforms + clips; GPU rasterizes batched clip-space triangles. The per-poly
                           // GPU matrix transform was a net loss (per-poly glLoadMatrixf + 2400 draw calls cost
                           // ~21ms of GXM render for a ~1ms CPU saving); batched clip-space draws are far cheaper.
    return 1;
}
void *g3w_picture(void) { return 0; }   // readback path only (out buffer), no shared picture

#define BATCH_V 65536   // capacity of the batched 3D vertex buffers (~2400 polys expanded to triangles/frame)

void g3w_render(const G3RenderState *r, const u8 *img, const u8 *pal, u32 *out) {
    (void)img; (void)pal;   // textures decoded on demand via gpu3d_decode_texture (reads the same flat VRAM)
    if (!sEnabled || !out) return;
    lazy_init();

    glBindFramebuffer(GL_FRAMEBUFFER, sFbo);
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

    // Batched drawing. The DS re-emits ~2400 polygons/frame; one glDrawArrays per polygon is ~2400 GXM draw-call
    // flushes - the dominant 3D render cost. Instead, expand each polygon's triangle fan and append it to big
    // arrays, issuing one glDrawArrays(GL_TRIANGLES) per run of polygons that share GL state (texture + blend).
    // Vertices are DS clip space fed under an identity projection, so there is no per-poly matrix and runs batch
    // freely. State is changed only at a flush, between runs.
    static float bPos[BATCH_V][4], bCol[BATCH_V][4], bUv[BATCH_V][2];
    int bn = 0, haveState = 0, curBlend = -1;
    GLuint curTex = 0;
#define GL3D_FLUSH() do { if (bn) { glVertexPointer(4, GL_FLOAT, 0, bPos); glColorPointer(4, GL_FLOAT, 0, bCol); \
        glTexCoordPointer(2, GL_FLOAT, 0, bUv); glDrawArrays(GL_TRIANGLES, 0, bn); bn = 0; } } while (0)
    for (int k = 0; k < r->npoly; k++) {
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
            // DS clip-space position (pos[]=NDC*w); GL does the perspective divide, viewport and depth. v->color
            // is the 9-bit interpolation value, so >>3 to the displayed 6-bit channel (dividing by 63 directly
            // was ~8x too bright).
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

    // Read back the FBO and pack into the DS 3D line format the 2D compositor expects:
    //   byte0=R6, byte1=G6, byte2=B6, byte3=(alpha5 | 0x80)   (0x80000000 marks a 3D pixel present).
    // glReadPixels is bottom-up, so flip vertically (dst row y <- src row VH-1-y).
    static u8 rb[VW * VH * 4];
    glReadPixels(0, 0, VW, VH, GL_RGBA, GL_UNSIGNED_BYTE, rb);
    for (int y = 0; y < VH; y++) {
        const u8 *src = rb + (VH - 1 - y) * VW * 4;
        u32 *dst = out + y * VW;
        for (int x = 0; x < VW; x++) {
            u32 R = src[x * 4 + 0] >> 2, G = src[x * 4 + 1] >> 2, B = src[x * 4 + 2] >> 2, A = src[x * 4 + 3] >> 3;
            dst[x] = R | G << 8 | B << 16 | ((A | 0x80u) << 24);
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// Enable predicate used by render_init: GPU path on, and the shader compiler reachable (else fall back to the
// software rasterizer rather than crash in vitaGL's runtime shader compile).
int vita_gl3d_available(void) {
    return vita_gl_enabled() && vita_gl_have_shader_compiler();
}
