// GPU rendering path for the Vita, via vitaGL (OpenGL ES over GXM).
//
// Goal: move the DS image onto the Vita GPU, the way the iOS/desktop builds use WebGPU/Metal. This file owns the
// vitaGL context and the GPU present path, which the Vita always uses.
//
// Stage 1 (now): vitaGL owns the display and presents the composited DS canvas as a GPU-textured, letterboxed
// image. It does this with glBlitFramebuffer, which in vitaGL uses PRECOMPILED blit/clear shaders - so the
// runtime Cg compiler (libshacccg.suprx) is NOT needed and nothing is compiled on device. We upload the canvas
// into a texture, attach it to an FBO, clear the screen black, and blit the FBO to the screen scaled to FIT
// (4:3, black bars). Stage 2 moves the DS 3D geometry onto the GPU (which will need real shader compilation).
//
// The frontend uses SDL for input and audio only when this path is active (both vitaGL and SDL's renderer would
// init GXM, and only one may).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vitaGL.h>

#include "vita.h"
#include "../gpu2d/gpu2d.h"
#include "../gpu2d/gpu2d_compose.h"
#include "../host_features.h"

#define SCREEN_W 960
#define SCREEN_H 544

static int sOn = -1;   // -1 unknown, 0 off, 1 on
static int sReady;
static GLuint sTex;    // holds the uploaded canvas
static GLuint sFbo;    // read source for the blit (canvas texture attached)
static int sTexW, sTexH;

// The GPU path is always on on the Vita.
int vita_gl_enabled(void) {
    if (sOn < 0) {
        sOn = 1;
    }
    return sOn;
}

// The on-device Cg compiler libshacccg.suprx is needed to compile shaders at runtime (for the upcoming GPU 3D
// rasterizer). The current present path (glBlitFramebuffer/glClear) uses precompiled shaders and needs none, so
// this is informational for now; the 3D path will require it. vitaGL/vitaShaRK probe these two paths.
// Where the runtime shader compiler may live, in priority order: bundled inside the VPK (app0:), next to the
// game (ux0:data/vitaWW/ - just a file in the game folder, no system install), then a system install (ur0:).
const char *const kShaderCompilerPaths[] = {
    "app0:libshacccg.suprx",
    "ux0:data/vitaWW/libshacccg.suprx",
    "ur0:/data/libshacccg.suprx",
    "ur0:data/external/libshacccg.suprx",
    0,
};
// The Cg compiler module (~0.4 s to load) is needed only to compile a shader that is not cached yet. It is loaded
// once at start-up, behind the boot screen (vita_gl_init), not in the first 3D frame (a ~380 ms stop at every
// start). Every place that may compile makes sure of it first (vita_shader_compiler_ensure).
extern uint64_t host_plat_time_ns(void);
static int sShTried, sShOk;
int vita_shader_compiler_ensure(void) {
    if (sShTried) return sShOk;
    sShTried = 1;
    extern int shark_init(const char *path);
    uint64_t t = host_plat_time_ns();
    for (int i = 0; kShaderCompilerPaths[i] && !sShOk; i++)
        if (shark_init(kShaderCompilerPaths[i]) >= 0) {
            sShOk = 1;
            fprintf(stderr, "vita_gl: shader compiler loaded from %s (%llu ms)\n", kShaderCompilerPaths[i],
                    (unsigned long long)((host_plat_time_ns() - t) / 1000000));
        }
    if (!sShOk) fprintf(stderr, "vita_gl: the shader compiler could not be loaded (shaders not cached cannot compile)\n");
    return sShOk;
}
int vita_gl_have_shader_compiler(void) {
    for (int i = 0; kShaderCompilerPaths[i]; i++) {
        FILE *f = fopen(kShaderCompilerPaths[i], "rb");
        if (f) { fclose(f); return 1; }
    }
    return 0;
}

// Bring up the GPU context. Returns 1 on success (the frontend falls back to the SDL renderer on 0).
int vita_gl_init(void) {
    if (!vita_gl_enabled()) return 0;
    // No immediate-mode/legacy pool needed: we only use glClear + glBlitFramebuffer (precompiled shaders).
    vglInitExtended(0, SCREEN_W, SCREEN_H, 8 * 1024 * 1024, SCE_GXM_MULTISAMPLE_NONE);
    vglWaitVblankStart(GL_TRUE);   // V-Sync: each picture is flipped at a vblank of the display (59.94 Hz), one per refresh
    fprintf(stderr, "vita_gl: shader compiler (libshacccg.suprx) %s - needed for the GPU 3D path\n",
            vita_gl_have_shader_compiler() ? "found" : "NOT found; install it to ur0:/data/");
    vita_shader_compiler_ensure();
    glViewport(0, 0, SCREEN_W, SCREEN_H);
    glGenTextures(1, &sTex);
    glGenFramebuffers(1, &sFbo);
    sReady = 1;
    return 1;
}

// (Re)allocate the canvas texture and attach it to the read FBO when the canvas size changes.
static void ensure_tex(int w, int h, const void *pixels) {
    glBindTexture(GL_TEXTURE_2D, sTex);
    if (w != sTexW || h != sTexH) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_BGRA, w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, pixels);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, sFbo);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sTex, 0);
        sTexW = w;
        sTexH = h;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_BGRA, GL_UNSIGNED_BYTE, pixels);
    }
}

// Blit the canvas texture (sTex, via sFbo) to the whole screen, and flip.
static void blit_canvas(GLuint fbo, int sx, int sy, int sw, int sh);
static void present_swap(void);

// ---- GPU compose -------------------------------------------------------------------------------------------------
// The 2D bands hand over the frame's layers (gpu2d.h Gpu2dGc) and the GPU composes them with the 3D colour buffer
// straight into the canvas texture (gpu2d_compose.h): no 3D readback, no CPU compose, no canvas upload.
extern uint64_t host_plat_time_ns(void);
unsigned gVitaRepeats;   // display refreshes that repeated a picture (render.c reports it)
unsigned gVitaRepeatsTotal;   // (never reset: vita_gl3d.c's GPU probe)
// GPU time probe (vita_gl3d.c starts it): a frame drawn with the GPU waited for after each pass, logged.
uint64_t gVitaGpu3dNs, gVitaGpuComposeNs, gVitaGpuPresentNs;
int gVitaGpuProbe, gVitaGpuProbed;
unsigned long long gVitaComposeNs, gVitaSwapNs, gVitaPresents, gVitaFrameSwapNs;   // present split (reported with the 3D stats)
int gVitaGc;   // GPU compose active (also tells the 3D renderer to skip its readback)
static GLuint sGcProg, sGcTex[2][8], sGcCanvas, sGcFbo;
// The compose in tiles (gpu2d_compose.h gc_tile_quads): where a tile can only show the 3D or the backdrop, a small
// program draws it; the rest gets the full compose. Measured against the one-quad compose on the probe frames.
static GLuint sGcTileProg;
static GLint sGcTileAttr = -1;
static float sGcFlip;      // the GPU's pixel rows against the quad's y (measured at start: gc_flip_measure)
static int sGcTiled = 1, sGcTileFast;
static uint64_t sGcOneNs, sGcTileNs;
static const unsigned *sGcLogoPx;   // this frame's title logo (RGBA), for the tiles
static int sGcWhy[8];
static float sGcWhole[4];
void vita_gl_gc_why(int w[8], float f[4]) { memcpy(w, sGcWhy, sizeof sGcWhy); memcpy(f, sGcWhole, sizeof sGcWhole); }
static GLint sGcAttr = -1, sGcHave3d = -1, sGcOvlOn = -1;
static GLuint sGcLp[2];   // the per-line parameters, decoded (gpu2d_compose.h gc_line_texels): one per layer set
// The composed canvas is the Vita screen's 30:17 at the DS's 192 lines (339 / 192 = 1.7656, 960 / 544 = 1.7647: the
// canvas pixels come out square on the screen): GC_W columns (the 3D view across all of them, the 2D layers
// 1:1 in the middle 256), in a texture GC_TEX_W wide (a multiple of 32 for the GPU surface).
enum { GC_W = 339, GC_TEX_W = 352 };
enum { GC_L = (GC_W - 256) / 2, GC_R = GC_W - 256 - GC_L };   // the side bars beside the DS's 256 columns: 41 left, 42 right
static GLint sGcLogoOn = -1, sGcLogoOfs = -1, sGcBg3Shift = -1, sGcMirror = -1, sGcObjMove = -1, sGcObjShift = -1;
// The menu over the scene (gpu2d: engine B's picture, alpha 0 where it has nothing): a small pass of its own over
// the composed canvas, in the 256 DS columns. (Kept out of the big compose shader, which is near the limits of the
// Vita's shader compiler.)
static GLuint sOvlProg;
static GLint sOvlPos = -1, sOvlTex = -1;
static int sOvlNow;
static const char kMenuVertexShaderCg[] =
    "void main(float2 aPos, float2 aTex, out float4 oPos : POSITION, out float2 vTex : TEXCOORD0) {\n"
    "  oPos = float4(aPos, 0.0, 1.0); vTex = aTex; }\n";
static const char kMenuFragmentShaderCg[] =
    "float4 main(float2 vTex : TEXCOORD0, uniform sampler2D uTex) : COLOR {\n"
    "  float4 c = tex2D(uTex, vTex);\n  if (c.a < 0.25) discard;\n  return float4(c.rgb, 1.0);\n}\n";   // the canvas to the screen, sharp bilinear (gpu2d_compose.h kBlit*)
static GLuint sGcLogoTex;
static int sGcState;   // 0 untried, 1 ready, -1 unavailable
static GLuint gc_shader(GLenum type, const char *src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024] = "";
        glGetShaderInfoLog(sh, sizeof log, NULL, log);
        fprintf(stderr, "vita_gl: a shader failed to compile: %s\n", log);
        return 0;
    }
    return sh;
}
// A Cg program, compiled once: the linked binary is kept in ux0:data/vitaWW (keyed by a hash of the sources) and
// loaded from there on later starts (compiling takes seconds on the Vita). 0 on failure.
static GLuint cached_program(const char *vsrc, const char *fsrc, const char *name) {
    GLuint prog;
    unsigned long long h = 1469598103934665603ull;
    for (const char *q = vsrc; *q; q++) { h ^= (unsigned char)*q; h *= 1099511628211ull; }
    for (const char *q = fsrc; *q; q++) { h ^= (unsigned char)*q; h *= 1099511628211ull; }
    char cachePath[128];
    snprintf(cachePath, sizeof cachePath, "ux0:data/vitaWW/%s-%016llx.bin", name, h);
    prog = glCreateProgram();
    int fromCache = 0;
    FILE *cf = fopen(cachePath, "rb");
    if (cf) {
        fseek(cf, 0, SEEK_END);
        long len = ftell(cf);
        fseek(cf, 0, SEEK_SET);
        void *bin = len > 0 ? malloc((size_t)len) : NULL;
        if (bin && fread(bin, 1, (size_t)len, cf) == (size_t)len) {
            glProgramBinary(prog, 0, bin, (GLsizei)len);
            GLint ok = 0;
            glGetProgramiv(prog, GL_LINK_STATUS, &ok);
            fromCache = ok != 0;
        }
        free(bin);
        fclose(cf);
        fprintf(stderr, "vita_gl: %s shaders %s the cache\n", name, fromCache ? "loaded from" : "could not be loaded from");
        if (!fromCache) { glDeleteProgram(prog); prog = glCreateProgram(); }
    }
    if (!fromCache) {
        fprintf(stderr, "vita_gl: %s shaders: compiling (first start only)\n", name);
        vita_shader_compiler_ensure();
        GLuint vs = gc_shader(GL_CG_VERTEX_SHADER_EXT, vsrc);
        GLuint fs = vs ? gc_shader(GL_CG_FRAGMENT_SHADER_EXT, fsrc) : 0;
        if (!vs || !fs) return 0;
        fprintf(stderr, "vita_gl: %s shaders: linking\n", name);
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glLinkProgram(prog);
        GLint ok = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) { fprintf(stderr, "vita_gl: %s program failed to link\n", name); return 0; }
        GLint len = 0;
        glGetProgramiv(prog, GL_PROGRAM_BINARY_LENGTH, &len);
        void *bin = len > 0 ? malloc((size_t)len) : NULL;
        if (bin) {
            GLsizei got = 0;
            GLenum fmt = 0;
            glGetProgramBinary(prog, len, &got, &fmt, bin);
            FILE *wf = got > 0 ? fopen(cachePath, "wb") : NULL;
            if (wf) { fwrite(bin, 1, (size_t)got, wf); fclose(wf); }
            free(bin);
        }
    }
    return prog;
}
// The compose's fixed uniforms (any of the compose programs).
static void gc_static_uniforms(GLuint prog) {
    glUseProgram(prog);
    glUniform1f(glGetUniformLocation(prog, "uW"), (float)GC_W);
    glUniform1f(glGetUniformLocation(prog, "u3dCols"), (float)GC_W);   // the 3D is drawn GC_W wide (vita_gl3d.c)
    // The 3D colour buffer is the screen's 960 columns for the canvas's GC_W (vita_gl3d.c): the compose's sample of
    // canvas column x lands in screen column (x + 0.5) * 960 / GC_W, the 3D pixel under the middle of that column.
    glUniform1f(glGetUniformLocation(prog, "u3dTexW"), (float)GC_W);
    static const char *const kNames[11] = {"uBg0", "uBg1", "uBg2", "uBg3", "uObj", "uAux", "uPar", "uOvl", "u3d", "uLogo", "uLp"};
    for (int i = 0; i < 11; i++) glUniform1i(glGetUniformLocation(prog, kNames[i]), i);
    // (the vertex shader's copies: it works out the coordinates the fragment program reads at)
    glUniform1f(glGetUniformLocation(prog, "uVW"), (float)GC_W);
    glUniform1f(glGetUniformLocation(prog, "uV3dCols"), (float)GC_W);
    glUniform1f(glGetUniformLocation(prog, "uV3dTexW"), (float)GC_W);
    glUseProgram(0);
}
static int sSkyFrames;   // frames the compose drew with the sky (render.c's log)
int vita_gl_sky_frames(void) { return __atomic_exchange_n(&sSkyFrames, 0, __ATOMIC_RELAXED); }
// This frame's compose uniforms (vita_gl_present_gc), set on whichever compose program draws.
static struct { float have3d, ovlOn, logoOn, mirror, ofs[2], bg3, mv[4], objShift, sky; } sGcU;
static void gc_frame_uniforms(GLuint prog) {
    // (the locations looked up once per program: vitaGL finds a uniform by comparing names)
    static GLuint progs[4];
    static GLint loc[4][12];
    int k = 0;
    while (k < 4 && progs[k] && progs[k] != prog) k++;
    if (k == 4) k = 0;
    if (progs[k] != prog) {
        static const char *const kN[12] = {"uHave3d", "uOvlOn", "uLogoOn", "uMirror", "uLogoOfs", "uBg3Shift",
                                           "uObjMove", "uVLogoOfs", "uVBg3Shift", "uVFlip", "uObjShift", "uSkyOn"};
        progs[k] = prog;
        for (int i = 0; i < 12; i++) loc[k][i] = glGetUniformLocation(prog, kN[i]);
    }
    const GLint *l = loc[k];
    glUseProgram(prog);
    glUniform1f(l[0], sGcU.have3d);
    glUniform1f(l[1], sGcU.ovlOn);
    glUniform1f(l[2], sGcU.logoOn);
    glUniform1f(l[3], sGcU.mirror);
    glUniform2f(l[4], sGcU.ofs[0], sGcU.ofs[1]);
    glUniform1f(l[5], sGcU.bg3);
    glUniform4f(l[6], sGcU.mv[0], sGcU.mv[1], sGcU.mv[2], sGcU.mv[3]);
    glUniform2f(l[7], sGcU.ofs[0], sGcU.ofs[1]);
    glUniform1f(l[8], sGcU.bg3);
    glUniform1f(l[9], sGcFlip);
    glUniform1f(l[10], sGcU.objShift);
    glUniform1f(l[11], sGcU.sky);
}
// The present, built from plain texture copies and the GPU's blending (per-pixel arithmetic at the screen's
// 960x544 is what costs on the Vita: a shader of a few dozen operations took 10 ms, a copy 1.2 ms):
//  1. the canvas, nearest, 3x into an offscreen picture (1017x576): each canvas pixel a 3x3 block
//  2. the 3D colour buffer (960x544, full resolution) copied to the screen
//  3. on DS lines with a master brightness (the fades): that brightness over the 3D, by blending white or black
//  4. the 3x picture to the screen, linear (= "sharp bilinear": crisp blocks, only the block edges blended), blended
//     by the canvas alpha - 0 where the compose marked the pixel as exactly the 3D (the full-resolution 3D shows),
//     1 elsewhere (the 2D, anything blended, the menus)
enum { PRE_W = GC_W * 3, PRE_H = 192 * 3 };
static const char kCopyVertexCg[] =
    "void main(float2 aPos, float2 aTex, out float4 oPos : POSITION, out float2 vTex : TEXCOORD0) {\n"
    "  oPos = float4(aPos, 0.0, 1.0); vTex = aTex; }\n";
static const char kCopyFragmentCg[] =
    "float4 main(float2 vTex : TEXCOORD0, uniform sampler2D uTex) : COLOR { return tex2D(uTex, vTex); }\n";
// The 3x copy of the canvas: the texel under each pixel by its index (exact blocks whatever the texture's filter).
static const char kNearFragmentCg[] =
    "float4 main(float2 vTex : TEXCOORD0, uniform sampler2D uTex, uniform float4 uSz) : COLOR {\n"
    "  return tex2D(uTex, (floor(vTex * uSz.xy) + 0.5) * uSz.zw); }\n";
static const char kFillVertexCg[] =
    "void main(float2 aPos, float4 aCol, out float4 oPos : POSITION, out float4 vCol : COLOR) {\n"
    "  oPos = float4(aPos, 0.0, 1.0); vCol = aCol; }\n";
static const char kFillFragmentCg[] =
    "float4 main(float4 vCol : COLOR) : COLOR { return vCol; }\n";
static GLuint sCopyProg, sFillProg, sNearProg, sPreTex, sPreFbo;
static GLint sNearPos = -1, sNearTex = -1;
static GLint sCopyPos = -1, sCopyTex = -1, sFillPos = -1, sFillCol = -1;
static void present_init(void) {
    sCopyProg = cached_program(kCopyVertexCg, kCopyFragmentCg, "copy");
    sFillProg = cached_program(kFillVertexCg, kFillFragmentCg, "fill");
    sNearProg = cached_program(kCopyVertexCg, kNearFragmentCg, "near");
    if (sNearProg) {
        sNearPos = glGetAttribLocation(sNearProg, "aPos");
        sNearTex = glGetAttribLocation(sNearProg, "aTex");
        glUseProgram(sNearProg);
        glUniform1i(glGetUniformLocation(sNearProg, "uTex"), 0);
        glUniform4f(glGetUniformLocation(sNearProg, "uSz"), (float)GC_TEX_W, 192.0f, 1.0f / GC_TEX_W, 1.0f / 192.0f);
        glUseProgram(0);
    }
    if (sCopyProg && sFillProg) {
        sCopyPos = glGetAttribLocation(sCopyProg, "aPos");
        sCopyTex = glGetAttribLocation(sCopyProg, "aTex");
        sFillPos = glGetAttribLocation(sFillProg, "aPos");
        sFillCol = glGetAttribLocation(sFillProg, "aCol");
        glUseProgram(sCopyProg);
        glUniform1i(glGetUniformLocation(sCopyProg, "uTex"), 0);
        glUseProgram(0);
    }
    if (!sCopyProg || !sFillProg || !sNearProg || sCopyPos < 0 || sCopyTex < 0 || sFillPos < 0 || sFillCol < 0 ||
        sNearPos < 0 || sNearTex < 0) {
        fprintf(stderr, "vita_gl: present shaders unavailable (linear blit)\n");
        sCopyProg = (GLuint)-1;
        return;
    }
    glGenTextures(1, &sPreTex);
    glBindTexture(GL_TEXTURE_2D, sPreTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, PRE_W, PRE_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &sPreFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, sPreFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sPreTex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    fprintf(stderr, "vita_gl: present ready (3x canvas + full-resolution 3D, blended)\n");
}
static const float kFullPos[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
static void copy_quad(GLuint tex, const float *uv) {
    glUseProgram(sCopyProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glEnableVertexAttribArray((GLuint)sCopyPos);
    glEnableVertexAttribArray((GLuint)sCopyTex);
    glVertexAttribPointer((GLuint)sCopyPos, 2, GL_FLOAT, GL_FALSE, 0, kFullPos);
    glVertexAttribPointer((GLuint)sCopyTex, 2, GL_FLOAT, GL_FALSE, 0, uv);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableVertexAttribArray((GLuint)sCopyPos);
    glDisableVertexAttribArray((GLuint)sCopyTex);
}
// canvas (row 0 = DS line 0 = the top of the picture) into the 3x picture, nearest (the canvas texture's filter)
static const float kCanvasUv[8] = {0, 1, (float)GC_W / GC_TEX_W, 1, 0, 0, (float)GC_W / GC_TEX_W, 0};
static const float kFlatUv[8] = {0, 0, 1, 0, 0, 1, 1, 1};   // the 3D buffer / the 3x picture: row 0 at the bottom
static void near_quad(void) {   // the canvas over the whole target, each pixel the canvas texel under it
    glUseProgram(sNearProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sGcCanvas);
    glEnableVertexAttribArray((GLuint)sNearPos);
    glEnableVertexAttribArray((GLuint)sNearTex);
    glVertexAttribPointer((GLuint)sNearPos, 2, GL_FLOAT, GL_FALSE, 0, kFullPos);
    glVertexAttribPointer((GLuint)sNearTex, 2, GL_FLOAT, GL_FALSE, 0, kCanvasUv);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableVertexAttribArray((GLuint)sNearPos);
    glDisableVertexAttribArray((GLuint)sNearTex);
}
static void present_pre(void) {
    glBindFramebuffer(GL_FRAMEBUFFER, sPreFbo);
    glViewport(0, 0, PRE_W, PRE_H);
    glDisable(GL_BLEND);
    near_quad();
}
static void present_3d(unsigned tex3d) {
    glDisable(GL_BLEND);
    copy_quad(tex3d, kFlatUv);
}
// The fades over the 3D: per DS line with a master brightness, white (up) or black (down) at factor / 16.
static int present_fades(const Gpu2dGc *gc) {
    static float pos[GC_VERTS][2], col[GC_VERTS][4];
    int n = 0;
    for (int y = 0; y < 192; y++) {
        const unsigned char *p = gc->par + y * GPU2D_GC_PAR * 4;
        const int mode = p[10], f = p[11] > 16 ? 16 : p[11];
        if (!(mode == 1 || mode == 2) || !f) continue;
        const float ya = 1.0f - 2.0f * (float)y / 192.0f, yb = 1.0f - 2.0f * (float)(y + 1) / 192.0f;
        const float q[6][2] = {{-1, ya}, {1, ya}, {-1, yb}, {-1, yb}, {1, ya}, {1, yb}};
        const float c = mode == 1 ? 1.0f : 0.0f, a = (float)f / 16.0f;
        for (int k = 0; k < 6; k++, n++) {
            pos[n][0] = q[k][0]; pos[n][1] = q[k][1];
            col[n][0] = col[n][1] = col[n][2] = c; col[n][3] = a;
        }
    }
    if (!n) return 0;
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(sFillProg);
    glEnableVertexAttribArray((GLuint)sFillPos);
    glEnableVertexAttribArray((GLuint)sFillCol);
    glVertexAttribPointer((GLuint)sFillPos, 2, GL_FLOAT, GL_FALSE, 0, pos);
    glVertexAttribPointer((GLuint)sFillCol, 4, GL_FLOAT, GL_FALSE, 0, col);
    glDrawArrays(GL_TRIANGLES, 0, n);
    glDisableVertexAttribArray((GLuint)sFillPos);
    glDisableVertexAttribArray((GLuint)sFillCol);
    glDisable(GL_BLEND);
    return 1;
}
// smooth: the 3x picture, linear (block edges blended); sharp (the option off): the canvas straight to the screen,
// each screen pixel the canvas pixel under it (plain enlarged pixels, 2 or 3 screen pixels wide)
static void present_final(int has3d, int smooth) {
    if (has3d) {
        glEnable(GL_BLEND);
        glBlendEquation(GL_FUNC_ADD);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    } else
        glDisable(GL_BLEND);
    if (smooth) copy_quad(sPreTex, kFlatUv);
    else near_quad();
    glDisable(GL_BLEND);
}
static void present_draw(const Gpu2dGc *gc, unsigned tex3d, int has3d) {
    const int smooth = feature_on(FEAT_SMOOTH);   // (the options file / settings screen: smooth_pixels)
    if (smooth) present_pre();   // (first: the offscreen picture, then the screen in one go)
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, SCREEN_W, SCREEN_H);
    if (has3d) {
        present_3d(tex3d);
        present_fades(gc);
    }
    present_final(has3d, smooth);
    glUseProgram(0);
}
// Which way the GPU numbers the pixel rows of the canvas against the quad's y: draw the measuring program into the
// canvas and read a pixel back (red: they agree). The compose's interpolated coordinates follow it.
static void gc_flip_measure(void) {
    GLuint p = cached_program(kGcqVertexShaderCg, kFlipCalFragmentCg, "flipcal");
    if (!p) { fprintf(stderr, "vita_gl: GPU compose: no row-direction check (assuming the GL direction)\n"); return; }
    GLint a = glGetAttribLocation(p, "aPos");
    if (a < 0) return;
    gc_static_uniforms(p);
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "uVFlip"), 0.0f);
    glUniform2f(glGetUniformLocation(p, "uVLogoOfs"), 0.0f, 0.0f);
    glUniform1f(glGetUniformLocation(p, "uVBg3Shift"), 0.0f);
    static const float kQ[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, sGcFbo);
    glViewport(0, 0, GC_W, 192);
    glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND); glDisable(GL_SCISSOR_TEST); glDisable(GL_CULL_FACE);
    glEnableVertexAttribArray((GLuint)a);
    glVertexAttribPointer((GLuint)a, 2, GL_FLOAT, GL_FALSE, 0, kQ);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableVertexAttribArray((GLuint)a);
    unsigned char px[2][4] = {{0}};
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px[0]);
    glReadPixels(GC_W / 2, 100, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px[1]);
    glUseProgram(0);
    sGcFlip = px[0][0] < 128 ? 1.0f : 0.0f;
    fprintf(stderr, "vita_gl: GPU compose: pixel rows %s the quad's y (check %u %u)\n",
            sGcFlip > 0.5f ? "run against" : "follow", px[0][0], px[1][0]);
}
int vita_gl_gc_init(void) {
    if (sGcState) return sGcState > 0;
    sGcState = -1;
    if (!sReady) return 0;
    // Cg sources compiled as is (GL_CG_*: not through vitaGL's GLSL translator). Compiling takes seconds on the
    // Vita, so the linked program is kept in ux0:data/vitaWW (keyed by a hash of the sources) and loaded from there
    // on later starts. Each step is logged, so a failure on a device shows where it happened.
    sGcProg = cached_program(kGcqVertexShaderCg, kGcqFragmentShaderCg, "composeq");
    if (!sGcProg) return 0;
    sGcAttr = glGetAttribLocation(sGcProg, "aPos");
    fprintf(stderr, "vita_gl: GPU compose: linked (aPos at %d), creating textures\n", (int)sGcAttr);
    if (sGcAttr < 0) return 0;
    sGcTileProg = cached_program(kGcqVertexShaderCg, kGcTileFragmentShaderCg, "composet");
    if (sGcTileProg) {
        sGcTileAttr = glGetAttribLocation(sGcTileProg, "aPos");
        if (sGcTileAttr < 0) sGcTileProg = 0;
        else gc_static_uniforms(sGcTileProg);
    }
    fprintf(stderr, "vita_gl: GPU compose: tiles %s\n", sGcTileProg ? "ready" : "unavailable (one quad)");
    sGcHave3d = glGetUniformLocation(sGcProg, "uHave3d");
    sGcOvlOn = glGetUniformLocation(sGcProg, "uOvlOn");
    gc_static_uniforms(sGcProg);
    glUseProgram(sGcProg);
    sGcLogoOn = glGetUniformLocation(sGcProg, "uLogoOn");
    sGcLogoOfs = glGetUniformLocation(sGcProg, "uLogoOfs");
    sGcBg3Shift = glGetUniformLocation(sGcProg, "uBg3Shift");
    sGcMirror = glGetUniformLocation(sGcProg, "uMirror");
    sGcObjMove = glGetUniformLocation(sGcProg, "uObjMove");
    sGcObjShift = glGetUniformLocation(sGcProg, "uObjShift");
    glUniform1f(sGcMirror, 0.0f);
    glUniform1f(sGcLogoOn, 0.0f);
    glUseProgram(0);
    for (int set = 0; set < 2; set++) {
        glGenTextures(8, sGcTex[set]);
        for (int i = 0; i < 8; i++) {
            glBindTexture(GL_TEXTURE_2D, sGcTex[set][i]);
            if (i < 6) glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE_ALPHA, 256, 192, 0, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, NULL);
            else if (i == 6) glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, GPU2D_GC_PAR, 192, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            else glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
    }
    // The bands write straight into the layer textures (both sets): nothing is uploaded per frame.
    for (int set = 0; set < 2; set++) {
        void *ptr[7];
        for (int i = 0; i < 7; i++) {
            glBindTexture(GL_TEXTURE_2D, sGcTex[set][i]);
            ptr[i] = vglGetTexDataPointer(GL_TEXTURE_2D);
            if (!ptr[i]) { fprintf(stderr, "vita_gl: GPU compose: no texture memory pointer\n"); return 0; }
        }
        uint16_t *bg[4] = {ptr[0], ptr[1], ptr[2], ptr[3]};
        gpu2d_set_gc_buffers(set, bg, ptr[4], ptr[5], ptr[6]);
    }
    glGenTextures(2, sGcLp);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, sGcLp[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 8, 192, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glGenTextures(1, &sGcCanvas);
    glBindTexture(GL_TEXTURE_2D, sGcCanvas);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, GC_TEX_W, 192, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);   // (copied 3x, nearest: present_pre)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &sGcFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, sGcFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sGcCanvas, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    gc_flip_measure();
    sGcState = 1;
    gVitaGc = 1;
    fprintf(stderr, "vita_gl: GPU compose on (2D layers + 3D blended on the GPU)\n");
    return 1;
}

// The compose's draw into the canvas: one quad, or the tiles (fast ones by the small program, the rest in full).
static unsigned sGcStatFrames, sGcStatTiled, sGcStatFast;
void vita_gl_gc_stats(unsigned *frames, unsigned *tiled, unsigned *fast) {
    *frames = __atomic_exchange_n(&sGcStatFrames, 0, __ATOMIC_RELAXED);
    *tiled = __atomic_exchange_n(&sGcStatTiled, 0, __ATOMIC_RELAXED);
    *fast = __atomic_exchange_n(&sGcStatFast, 0, __ATOMIC_RELAXED);
}
// tiled: 0 one quad, 1 the tiles, 2 the tiles if enough of them are fast (else one quad)
static void gc_draw(const Gpu2dGc *gc, const float *quad, int tiled) {
    static float fv[GC_TILE_VERTS][2], sv[GC_TILE_VERTS][2];
    int nf = 0, ns = 0;
    if (tiled) {
        // (the title's logo and moved BG3: their tiles get the full compose, the rest of the picture need not)
        const int whole = sGcU.have3d < 0.5f || sGcU.mirror > 0.5f;
        static unsigned extra[192], logoCells[192];
        static unsigned logoGen = ~0u;
        static int logoKey[3] = {-1, 0, 0};
        if (sGcU.logoOn > 0.5f && sGcLogoPx) {   // (the logo's cells: worked out again only when it changes)
            extern unsigned gpu2d_gc_logo_gen(void);
            unsigned gen = gpu2d_gc_logo_gen();
            int key[3] = {1, (int)sGcU.ofs[0], (int)sGcU.ofs[1]};
            if (gen != logoGen || memcmp(key, logoKey, sizeof key)) {
                gc_tile_extra(gc->par, GPU2D_GC_PAR * 4, GC_W, sGcLogoPx, key[1], key[2], 0, logoCells);
                logoGen = gen;
                memcpy(logoKey, key, sizeof key);
            }
        } else {
            logoKey[0] = -1;
        }
        gc_tile_extra(gc->par, GPU2D_GC_PAR * 4, GC_W, NULL, 0, 0, (int)sGcU.bg3, extra);
        if (logoKey[0] == 1)
            for (int y = 0; y < 192; y++) extra[y] |= logoCells[y];
        const int oy0 = sGcU.objShift > 0.5f ? (int)sGcU.mv[1] : 999, oy1 = sGcU.objShift > 0.5f ? (int)sGcU.mv[3] : 999;
        sGcTileFast = gc_tile_quads(gc->par, GPU2D_GC_PAR * 4, GC_W, (float)GC_W, sGcFlip > 0.5f, whole,
                                    sGcU.ovlOn > 0.5f, oy0, oy1, extra, fv, &nf, sv, &ns);
        // (auto: the tiles only when at least half of them take the small program; else the one quad)
        if (tiled == 2 && sGcTileFast * 2 < 432) tiled = 0;
        __atomic_fetch_add(&sGcStatFrames, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&sGcStatFast, (unsigned)sGcTileFast, __ATOMIC_RELAXED);
    }
    if (tiled) {
        __atomic_fetch_add(&sGcStatTiled, 1, __ATOMIC_RELAXED);
        if (nf) {
            gc_frame_uniforms(sGcTileProg);
            glUseProgram(sGcTileProg);
            glEnableVertexAttribArray((GLuint)sGcTileAttr);
            glVertexAttribPointer((GLuint)sGcTileAttr, 2, GL_FLOAT, GL_FALSE, 0, fv);
            glDrawArrays(GL_TRIANGLES, 0, nf);
            glDisableVertexAttribArray((GLuint)sGcTileAttr);
        }
        if (ns) {
            gc_frame_uniforms(sGcProg);
            glUseProgram(sGcProg);
            glEnableVertexAttribArray((GLuint)sGcAttr);
            glVertexAttribPointer((GLuint)sGcAttr, 2, GL_FLOAT, GL_FALSE, 0, sv);
            glDrawArrays(GL_TRIANGLES, 0, ns);
            glDisableVertexAttribArray((GLuint)sGcAttr);
        }
        return;
    }
    gc_frame_uniforms(sGcProg);
    glUseProgram(sGcProg);
    glEnableVertexAttribArray((GLuint)sGcAttr);
    glVertexAttribPointer((GLuint)sGcAttr, 2, GL_FLOAT, GL_FALSE, 0, quad);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableVertexAttribArray((GLuint)sGcAttr);
}
void vita_gl_gc_times(uint64_t *one, uint64_t *tiles, int *fastTiles, int *usingTiles) {
    *one = sGcOneNs; *tiles = sGcTileNs; *fastTiles = sGcTileFast; *usingTiles = sGcTiled;
}
// Compose gc + the 3D colour buffer tex3d (0: none yet) into the canvas texture and present it. ovl: the FPS
// overlay rows (256x16, alpha byte 0 = overlay pixel) or NULL.
void vita_gl_present_gc(const Gpu2dGc *gc, unsigned tex3d, const uint32_t *ovl) {
    if (!sReady || sGcState <= 0) return;
    static int first = 1;
    if (first) fprintf(stderr, "vita_gl: GPU compose: first frame\n");
    const int probe = gVitaGpuProbe;   // (vita_gl3d.c waited for the GPU before this frame's 3D)
    uint64_t tp = 0;
    if (probe) { tp = host_plat_time_ns(); glFinish(); gVitaGpu3dNs = host_plat_time_ns() - tp; }
    uint64_t t0 = host_plat_time_ns();
    GLuint *t = sGcTex[gc->set & 1];   // the bands wrote this set's texture memory directly
    for (int i = 0; i < 7; i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, t[i]);
    }
    glActiveTexture(GL_TEXTURE0 + 7);
    glBindTexture(GL_TEXTURE_2D, t[7]);
    if (ovl) {
        static uint8_t o[256 * 16 * 4];
        for (int i = 0; i < 256 * 16; i++) {
            uint32_t v = ovl[i];
            o[i * 4] = (uint8_t)(v >> 16); o[i * 4 + 1] = (uint8_t)(v >> 8); o[i * 4 + 2] = (uint8_t)v;
            o[i * 4 + 3] = (v >> 24) ? 0 : 255;
        }
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 16, GL_RGBA, GL_UNSIGNED_BYTE, o);
    }
    glActiveTexture(GL_TEXTURE0 + 8);
    glBindTexture(GL_TEXTURE_2D, tex3d ? tex3d : t[7]);   // (no 3D yet: any valid texture; uHave3d 0 keeps it unread)
    int mirrorNow = 0;   // the boot screens: no 3D shown
    {   // the title screen: the logo (gpu2d: the top screen's BG2), 1:1, centred on the screen; "touch to start"
        // (BG3) moved right by the side bar's width, into the screen's bottom-right corner
        extern int gpu2d_gc_logo(const uint32_t **px, int *dx, int *dy, int *prompt, int *mirror);
        const uint32_t *lp;
        int dx, dy, prompt, mirror;
        int on = gpu2d_gc_logo(&lp, &dx, &dy, &prompt, &mirror);
        if (on) {
            // Two textures, used in turn: the rows written since a texture was last brought up to date are copied
            // into it (glTexSubImage2D, row runs) - not the whole picture re-created each frame (glTexImage2D
            // allocates new texture memory every call, and a menu changes its rows every frame).
            static GLuint logoTex[2];
            static unsigned char pend[2][192];
            static int cur;
            glActiveTexture(GL_TEXTURE0 + 9);
            const int first = !logoTex[0];
            if (first) {
                glGenTextures(2, logoTex);
                for (int i = 0; i < 2; i++) {
                    glBindTexture(GL_TEXTURE_2D, logoTex[i]);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 192, 0, GL_RGBA, GL_UNSIGNED_BYTE, lp);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                }
            }
            extern void gpu2d_gc_logo_take_rows(unsigned char *rows);
            unsigned char rows[192];
            gpu2d_gc_logo_take_rows(rows);
            int any = 0;
            for (int y = 0; y < 192; y++) { pend[0][y] |= rows[y]; pend[1][y] |= rows[y]; any |= rows[y]; }
            if (first) memset(pend, 0, sizeof pend);
            else if (any) {
                cur ^= 1;   // (the other one: the GPU may still be reading the one shown last)
                glBindTexture(GL_TEXTURE_2D, logoTex[cur]);
                for (int y = 0; y < 192;) {
                    if (!pend[cur][y]) { y++; continue; }
                    int e = y;
                    while (e < 192 && pend[cur][e]) e++;
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, y, 256, e - y, GL_RGBA, GL_UNSIGNED_BYTE, lp + y * 256);
                    y = e;
                }
                memset(pend[cur], 0, sizeof pend[cur]);
            }
            sGcLogoTex = logoTex[cur];
            glBindTexture(GL_TEXTURE_2D, sGcLogoTex);
        }
        sGcU.logoOn = on && !mirror ? 1.0f : 0.0f;
        sGcLogoPx = on && !mirror ? (const unsigned *)lp : NULL;
        sGcU.mirror = mirror == 1 ? 1.0f : 0.0f;   // the boot screens: the top screen's picture instead
        sOvlNow = mirror == 2 && on;
        sGcU.sky = on && mirror == 3 ? 1.0f : 0.0f;   // the top screen's sky where the backdrop would be (gpu2d sSkyOn)
        if (sGcU.sky > 0.5f) __atomic_fetch_add(&sSkyFrames, 1, __ATOMIC_RELAXED);
        mirrorNow = on && mirror == 1;   // a menu: drawn over the composed picture below (its own small pass)
        sGcU.ofs[0] = (float)dx; sGcU.ofs[1] = (float)dy;
        sGcU.bg3 = prompt ? (float)GC_R : 0.0f;   // (to the screen's right edge)
        // the HUD clock's sprites moved right by the side bar's width, to the screen's right edge (gpu2d)
        extern int gpu2d_gc_objmove(int rect[4]);
        int r[4];
        int mv = gpu2d_gc_objmove(r);
        for (int i = 0; i < 4; i++) sGcU.mv[i] = (float)r[i];
        sGcU.objShift = mv ? (float)GC_R : 0.0f;
    }
    {   // this frame's per-line parameters, decoded once per line
        static unsigned char lt[192][8][4];
        gc_line_texels(gc->par, GPU2D_GC_PAR * 4, (float)GC_W, lt);
        glActiveTexture(GL_TEXTURE0 + 10);
        glBindTexture(GL_TEXTURE_2D, sGcLp[gc->set & 1]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 8, 192, GL_RGBA, GL_UNSIGNED_BYTE, lt);
    }
    glActiveTexture(GL_TEXTURE0);

    glBindFramebuffer(GL_FRAMEBUFFER, sGcFbo);   // the RGBA canvas texture is its colour attachment
    glViewport(0, 0, GC_W, 192);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);
    glDepthMask(GL_FALSE);
    sGcU.have3d = tex3d ? 1.0f : 0.0f;
    sGcU.ovlOn = ovl ? 1.0f : 0.0f;
    static const float kQuad[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, sGcFbo);
    glViewport(0, 0, GC_W, 192);
    if (probe && sGcTileProg) {   // both ways, each waited for: keep the faster for the frames that follow
        glFinish();
        uint64_t ta = host_plat_time_ns();
        gc_draw(gc, kQuad, 0);
        glFinish();
        uint64_t tb = host_plat_time_ns();
        static int why[8];
        memset(why, 0, sizeof why);
        gGcTileWhy = why;
        gc_draw(gc, kQuad, 1);
        gGcTileWhy = NULL;
        memcpy(sGcWhy, why, sizeof why);
        sGcWhole[0] = sGcU.have3d; sGcWhole[1] = sGcU.logoOn; sGcWhole[2] = sGcU.mirror; sGcWhole[3] = sGcU.bg3;
        glFinish();
        uint64_t tc = host_plat_time_ns();
        sGcOneNs = tb - ta; sGcTileNs = tc - tb;
        sGcTiled = sGcTileNs < sGcOneNs;
    } else
        gc_draw(gc, kQuad, sGcTileProg ? 2 : 0);

    if (sOvlNow) {   // the menu over the scene, in the DS columns of the canvas (row 0 = DS line 0, as the compose)
        if (!sOvlProg) {
            sOvlProg = cached_program(kMenuVertexShaderCg, kMenuFragmentShaderCg, "menu");
            if (sOvlProg) {
                sOvlPos = glGetAttribLocation(sOvlProg, "aPos");
                sOvlTex = glGetAttribLocation(sOvlProg, "aTex");
                glUseProgram(sOvlProg);
                glUniform1i(glGetUniformLocation(sOvlProg, "uTex"), 0);
                if (sOvlPos < 0 || sOvlTex < 0) sOvlProg = 0;
            }
            fprintf(stderr, "vita_gl: menu layer %s\n", sOvlProg ? "ready" : "unavailable");
            if (!sOvlProg) sOvlProg = (GLuint)-1;
        }
        if (sOvlProg != (GLuint)-1) {
            static const float kTex[8] = {0, 0, 1, 0, 0, 1, 1, 1};
            glViewport(GC_L, 0, 256, 192);
            glUseProgram(sOvlProg);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, sGcLogoTex);
            glEnableVertexAttribArray((GLuint)sOvlPos);
            glEnableVertexAttribArray((GLuint)sOvlTex);
            glVertexAttribPointer((GLuint)sOvlPos, 2, GL_FLOAT, GL_FALSE, 0, kQuad);
            glVertexAttribPointer((GLuint)sOvlTex, 2, GL_FLOAT, GL_FALSE, 0, kTex);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glDisableVertexAttribArray((GLuint)sOvlPos);
            glDisableVertexAttribArray((GLuint)sOvlTex);
        }
    }
    if (probe) { tp = host_plat_time_ns(); glFinish(); gVitaGpuComposeNs = host_plat_time_ns() - tp; }
    gVitaComposeNs += host_plat_time_ns() - t0;
    glUseProgram(0);
    glDepthMask(GL_TRUE);
    glViewport(0, 0, SCREEN_W, SCREEN_H);
    if (!sCopyProg) present_init();
    if (sCopyProg != (GLuint)-1) {
        const int has3d = tex3d && !mirrorNow;
        present_draw(gc, tex3d, has3d);
        if (probe) {
            tp = host_plat_time_ns(); glFinish(); gVitaGpuPresentNs = host_plat_time_ns() - tp;
            gVitaGpuProbe = 0;
            __atomic_store_n(&gVitaGpuProbed, 1, __ATOMIC_RELEASE);
        }
        present_swap();
    } else
        blit_canvas(sGcFbo, 0, 0, GC_W, 192);
    if (first) { first = 0; fprintf(stderr, "vita_gl: GPU compose: first frame presented\n"); }
}

// Present one frame: upload the composited DS canvas (XRGB8888 == GL_BGRA bytes, a zero-conversion upload) and
// blit the sub-rectangle [sx,sy,sw,sh] of it to the whole screen (30:17).
void vita_gl_present_canvas(const void *pixels, int canvas_w, int canvas_h, int sx, int sy, int sw, int sh) {
    if (!sReady) return;

    ensure_tex(canvas_w, canvas_h, pixels);
    blit_canvas(sFbo, sx, sy, sw, sh);
}

static void blit_canvas(GLuint fbo, int sx, int sy, int sw, int sh) {
    // The whole 30:17 screen: the GPU-composed canvas is already 30:17 (3D view across it, the 2D layers unstretched
    // in the middle; vita_gl_present_gc), so it scales evenly to 960x544. (The CPU fallback canvas is stretched.)
    const int dx = 0, dy = 0, dw = SCREEN_W, dh = SCREEN_H;
    // The read source is our FBO (canvas texture); the draw target is the default framebuffer (the screen). The
    // blit covers every pixel, so no clear. vitaGL's blit samples the FBO's attachment and flips V.
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    // vitaGL flips V when reading an FBO as a texture, which would show the canvas upside-down; invert the source
    // Y range (srcY0=bottom, srcY1=top) to cancel that flip so row 0 of the canvas lands at the top of the screen.
    glBlitFramebuffer(sx, sy + sh, sx + sw, sy,
                      dx, dy, dx + dw, dy + dh,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);

    present_swap();
}
// On the render thread the swap is done after the thread has told the game it is finished with the frame's data
// (render.c render_worker_main): a swap that waits for the display (the queue full) then no longer holds up the
// game's next frame - the game thread was losing a frame each time (~once a second, swap ~20 ms).
int gVitaDeferSwap;
static int sSwapPending;
static void do_swap(void);
void vita_gl_swap_pending(void) {
    if (!sSwapPending) return;
    sSwapPending = 0;
    do_swap();
}
static void present_swap(void) {
    if (gVitaDeferSwap) { sSwapPending = 1; return; }
    do_swap();
}
static void do_swap(void) {
    static int shown;
    if (!shown) { shown = 1; fprintf(stderr, "vita_gl: first picture on screen\n"); }
    uint64_t ts = host_plat_time_ns();
    vglSwapBuffers(GL_FALSE);
    gVitaSwapNs += host_plat_time_ns() - ts;
    gVitaFrameSwapNs = host_plat_time_ns() - ts;
    gVitaPresents++;
    {   // refreshes the screen showed the previous picture again (visible stutter), by the display's vblank count
        extern unsigned host_vita_vcount(void);
        static unsigned last;
        unsigned vc = host_vita_vcount();
        if (last && vc - last > 1) { gVitaRepeats += vc - last - 1; gVitaRepeatsTotal += vc - last - 1; }
        last = vc;
    }
}

void vita_gl_shutdown(void) {
    // vitaGL has no teardown entry point - GXM and its pools are reclaimed when the process exits, which on the
    // Vita is the only way out of the game. Just mark the context down so present becomes a no-op.
    sReady = 0;
}
