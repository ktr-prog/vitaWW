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

static GLuint tex_for(uint32_t texparam, uint32_t texpal) {
    if ((texparam >> 26 & 7) == 0) return 0;   // no texture
    int slot = (int)((texparam * 2654435761u ^ texpal * 40503u) % TEXCACHE);
    for (int i = 0; i < 8; i++) {              // small linear probe
        int s = (slot + i) % TEXCACHE;
        if (sTexCache[s].used && sTexCache[s].tp == texparam && sTexCache[s].pal == texpal) return sTexCache[s].tex;
    }
    int w, h;
    gpu3d_texture_size(texparam, &w, &h);
    if (w <= 0 || h <= 0 || (long)w * h > (long)(sizeof sDecodeBuf / 4)) return 0;
    gpu3d_decode_texture(texparam, texpal, sDecodeBuf);
    int s = slot;                              // evict the hash slot (textures are long-lived; simple cache)
    if (!sTexCache[s].tex) glGenTextures(1, &sTexCache[s].tex);
    glBindTexture(GL_TEXTURE_2D, sTexCache[s].tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, sDecodeBuf);
    int rptS = texparam >> 16 & 1, rptT = texparam >> 17 & 1;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, rptS ? GL_REPEAT : GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, rptT ? GL_REPEAT : GL_CLAMP_TO_EDGE);
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
    return 1;
}
void *g3w_picture(void) { return 0; }   // readback path only (out buffer), no shared picture

// one polygon's vertices at a time (DS polygons have at most 10 verts), re-filled and drawn per polygon
static float sVPos[16][4];
static float sVCol[16][4];
static float sVUv[16][2];

void g3w_render(const G3RenderState *r, const u8 *img, const u8 *pal, u32 *out) {
    (void)img; (void)pal;   // textures decoded on demand via gpu3d_decode_texture (reads the same flat VRAM)
    if (!sEnabled || !out) return;
    lazy_init();

    glBindFramebuffer(GL_FRAMEBUFFER, sFbo);
    glViewport(0, 0, VW, VH);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();

    u32 cc = r->clearColor;
    float cr = (cc & 31) / 31.0f, cg = (cc >> 5 & 31) / 31.0f, cb = (cc >> 10 & 31) / 31.0f;
    glClearColor(cr, cg, cb, (r->clearAttrHi & 31) / 31.0f);
    glClearDepthf(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, sVPos);
    glColorPointer(4, GL_FLOAT, 0, sVCol);
    glTexCoordPointer(2, GL_FLOAT, 0, sVUv);

    for (int k = 0; k < r->npoly; k++) {
        const G3Polygon *p = r->poly[k];
        if (p->nv < 3) continue;
        int tw = 8 << (p->texparam >> 20 & 7), th = 8 << (p->texparam >> 23 & 7);
        for (int i = 0; i < p->nv; i++) {
            const G3Vertex *v = p->v[i];
            float w = (float)(p->w[i] > 0 ? p->w[i] : 1);
            float x = (float)v->hx / 16.0f / VW * 2.0f - 1.0f;
            float y = 1.0f - (float)v->hy / 16.0f / VH * 2.0f;
            float z = (float)p->z[i] / 16777215.0f;
            if (z < 0) z = 0; else if (z > 1) z = 1;
            sVPos[i][0] = x * w; sVPos[i][1] = y * w; sVPos[i][2] = (2.0f * z - 1.0f) * w; sVPos[i][3] = w;
            sVCol[i][0] = v->color[0] / 63.0f; sVCol[i][1] = v->color[1] / 63.0f;
            sVCol[i][2] = v->color[2] / 63.0f; sVCol[i][3] = p->translucent ? (p->attr >> 16 & 31) / 31.0f : 1.0f;
            sVUv[i][0] = tw ? (float)v->tex[0] / 16.0f / tw : 0.0f;
            sVUv[i][1] = th ? (float)v->tex[1] / 16.0f / th : 0.0f;
        }
        GLuint tex = tex_for(p->texparam, p->texpal);
        if (tex) { glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, tex); }
        else glDisable(GL_TEXTURE_2D);
        if (p->translucent) { glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE); }
        else { glDisable(GL_BLEND); glDepthMask(GL_TRUE); }
        glDrawArrays(GL_TRIANGLE_FAN, 0, p->nv);
    }

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
