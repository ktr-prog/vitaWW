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
unsigned long long gVitaComposeNs, gVitaSwapNs, gVitaPresents, gVitaFrameSwapNs;   // present split (reported with the 3D stats)
int gVitaGc;   // GPU compose active (also tells the 3D renderer to skip its readback)
static GLuint sGcProg, sGcTex[2][8], sGcCanvas, sGcFbo;   // two layer sets (alternating per frame: one may still be read by the GPU) + the RGBA canvas
static GLint sGcAttr = -1, sGcHave3d = -1, sGcOvlOn = -1;
// The composed canvas is the Vita screen's 30:17 at the DS's 192 lines (339 / 192 = 1.7656, 960 / 544 = 1.7647: the
// canvas pixels come out square on the screen): GC_W columns (the 3D view across all of them, the 2D layers
// 1:1 in the middle 256), in a texture GC_TEX_W wide (a multiple of 32 for the GPU surface).
enum { GC_W = 339, GC_TEX_W = 352 };
enum { GC_L = (GC_W - 256) / 2, GC_R = GC_W - 256 - GC_L };   // the side bars beside the DS's 256 columns: 41 left, 42 right
static GLint sGcLogoOn = -1, sGcLogoOfs = -1, sGcBg3Shift = -1, sGcMirror = -1, sGcObjMove = -1, sGcObjShift = -1;
static GLuint sBlitProg;
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
static GLint sBlitPos = -1, sBlitTex = -1, sBlitP = -1;
static GLuint sMaskProg, sGcCanvas2, sGcFbo2;   // the 3D mark + brightness pass (gpu2d_compose.h kMask*)
static GLint sMaskPos = -1, sMaskP = -1;
// GPU time probe: one frame in 300 is drawn with the GPU waited for after each pass (that frame is a little late).
uint64_t gVitaGpu3dNs, gVitaGpuComposeNs, gVitaGpuPresentNs;
int gVitaGpuProbe, gVitaGpuProbed;
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
        static int compilerUp;   // the Cg compiler module, loaded before the first compile (vita_gl3d.c loads it
        if (!compilerUp) {       // only with the first 3D frame, which can come after the first picture)
            extern int shark_init(const char *path);
            for (int i = 0; kShaderCompilerPaths[i] && !compilerUp; i++) compilerUp = shark_init(kShaderCompilerPaths[i]) >= 0;
            fprintf(stderr, "vita_gl: shader compiler %s\n", compilerUp ? "loaded" : "could not be loaded");
        }
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
int vita_gl_gc_init(void) {
    if (sGcState) return sGcState > 0;
    sGcState = -1;
    if (!sReady) return 0;
    // Cg sources compiled as is (GL_CG_*: not through vitaGL's GLSL translator). Compiling takes seconds on the
    // Vita, so the linked program is kept in ux0:data/vitaWW (keyed by a hash of the sources) and loaded from there
    // on later starts. Each step is logged, so a failure on a device shows where it happened.
    sGcProg = cached_program(kGcVertexShaderCg, kGcFragmentShaderCg, "compose");
    if (!sGcProg) return 0;
    sGcAttr = glGetAttribLocation(sGcProg, "aPos");
    fprintf(stderr, "vita_gl: GPU compose: linked (aPos at %d), creating textures\n", (int)sGcAttr);
    if (sGcAttr < 0) return 0;
    sGcHave3d = glGetUniformLocation(sGcProg, "uHave3d");
    sGcOvlOn = glGetUniformLocation(sGcProg, "uOvlOn");
    glUseProgram(sGcProg);
    glUniform1f(glGetUniformLocation(sGcProg, "uW"), (float)GC_W);
    glUniform1f(glGetUniformLocation(sGcProg, "u3dCols"), (float)GC_W);   // the 3D is drawn GC_W wide (vita_gl3d.c)
    // The 3D colour buffer is the screen's 960 columns for the canvas's GC_W (vita_gl3d.c): the compose's sample of
    // canvas column x lands in screen column (x + 0.5) * 960 / GC_W, the 3D pixel under the middle of that column.
    glUniform1f(glGetUniformLocation(sGcProg, "u3dTexW"), (float)GC_W);
    static const char *const kNames[10] = {"uBg0", "uBg1", "uBg2", "uBg3", "uObj", "uAux", "uPar", "uOvl", "u3d", "uLogo"};
    for (int i = 0; i < 10; i++) glUniform1i(glGetUniformLocation(sGcProg, kNames[i]), i);
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
    glGenTextures(1, &sGcCanvas);
    glBindTexture(GL_TEXTURE_2D, sGcCanvas);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, GC_TEX_W, 192, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &sGcFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, sGcFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sGcCanvas, 0);
    glGenTextures(1, &sGcCanvas2);
    glBindTexture(GL_TEXTURE_2D, sGcCanvas2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, GC_TEX_W, 192, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &sGcFbo2);
    glBindFramebuffer(GL_FRAMEBUFFER, sGcFbo2);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sGcCanvas2, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    sGcState = 1;
    gVitaGc = 1;
    fprintf(stderr, "vita_gl: GPU compose on (2D layers + 3D blended on the GPU)\n");
    return 1;
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
    glBindTexture(GL_TEXTURE_2D, tex3d);
    int mirrorNow = 0;   // the boot screens: no 3D shown
    {   // the title screen: the logo (gpu2d: the top screen's BG2), 1:1, centred on the screen; "touch to start"
        // (BG3) moved right by the side bar's width, into the screen's bottom-right corner
        extern int gpu2d_gc_logo(const uint32_t **px, int *dx, int *dy, int *prompt, int *mirror);
        const uint32_t *lp;
        int dx, dy, prompt, mirror;
        int on = gpu2d_gc_logo(&lp, &dx, &dy, &prompt, &mirror);
        if (on) {
            glActiveTexture(GL_TEXTURE0 + 9);
            int first = !sGcLogoTex;
            if (first) glGenTextures(1, &sGcLogoTex);
            glBindTexture(GL_TEXTURE_2D, sGcLogoTex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 192, 0, GL_RGBA, GL_UNSIGNED_BYTE, lp);
            if (first) {
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            }
        }
        glUseProgram(sGcProg);
        glUniform1f(sGcLogoOn, on && !mirror ? 1.0f : 0.0f);
        glUniform1f(sGcMirror, mirror == 1 ? 1.0f : 0.0f);   // the boot screens: the top screen's picture instead
        sOvlNow = mirror == 2 && on;
        mirrorNow = on && mirror == 1;   // a menu: drawn over the composed picture below (its own small pass)
        glUniform2f(sGcLogoOfs, (float)dx, (float)dy);
        glUniform1f(sGcBg3Shift, prompt ? (float)GC_R : 0.0f);   // (to the screen's right edge)
        // the HUD clock's sprites moved right by the side bar's width, to the screen's right edge (gpu2d)
        extern int gpu2d_gc_objmove(int rect[4]);
        int r[4];
        int mv = gpu2d_gc_objmove(r);
        glUniform4f(sGcObjMove, (float)r[0], (float)r[1], (float)r[2], (float)r[3]);
        glUniform1f(sGcObjShift, mv ? (float)GC_R : 0.0f);
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
    glUseProgram(sGcProg);
    glUniform1f(sGcHave3d, tex3d ? 1.0f : 0.0f);
    glUniform1f(sGcOvlOn, ovl ? 1.0f : 0.0f);
    static const float kQuad[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray((GLuint)sGcAttr);
    glVertexAttribPointer((GLuint)sGcAttr, 2, GL_FLOAT, GL_FALSE, 0, kQuad);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableVertexAttribArray((GLuint)sGcAttr);
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
    // The 3D marks and the brightness, at the canvas's resolution, into the second canvas the present reads.
    if (!sMaskProg) {
        sMaskProg = cached_program(kMaskVertexShaderCg, kMaskFragmentShaderCg, "mask");
        if (sMaskProg) {
            sMaskPos = glGetAttribLocation(sMaskProg, "aPos");
            glUseProgram(sMaskProg);
            glUniform1i(glGetUniformLocation(sMaskProg, "uTex"), 0);
            glUniform1i(glGetUniformLocation(sMaskProg, "u3d"), 1);
            glUniform1i(glGetUniformLocation(sMaskProg, "uPar"), 2);
            sMaskP = glGetUniformLocation(sMaskProg, "uP");
            if (sMaskPos < 0) sMaskProg = 0;
        }
        fprintf(stderr, "vita_gl: full-resolution 3D %s\n", sMaskProg ? "on" : "unavailable (the mask shader failed)");
        if (!sMaskProg) sMaskProg = (GLuint)-1;
    }
    const int masked = sMaskProg != (GLuint)-1;
    if (masked) {
        glBindFramebuffer(GL_FRAMEBUFFER, sGcFbo2);
        glViewport(0, 0, GC_W, 192);
        glUseProgram(sMaskProg);
        glUniform4f(sMaskP, tex3d && !mirrorNow ? 1.0f : 0.0f, (float)GC_W, mirrorNow ? 0.0f : 1.0f, (float)GC_TEX_W);
        glActiveTexture(GL_TEXTURE0 + 1);
        glBindTexture(GL_TEXTURE_2D, tex3d);
        glActiveTexture(GL_TEXTURE0 + 2);
        glBindTexture(GL_TEXTURE_2D, t[6]);   // uPar
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sGcCanvas);
        glEnableVertexAttribArray((GLuint)sMaskPos);
        glVertexAttribPointer((GLuint)sMaskPos, 2, GL_FLOAT, GL_FALSE, 0, kQuad);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDisableVertexAttribArray((GLuint)sMaskPos);
    }
    if (probe) { tp = host_plat_time_ns(); glFinish(); gVitaGpuComposeNs = host_plat_time_ns() - tp; }
    gVitaComposeNs += host_plat_time_ns() - t0;
    glUseProgram(0);
    glDepthMask(GL_TRUE);
    glViewport(0, 0, SCREEN_W, SCREEN_H);
    if (!sBlitProg) {
        sBlitProg = cached_program(kBlitVertexShaderCg, kBlitFragmentShaderCg, "blit");
        if (sBlitProg) {
            sBlitPos = glGetAttribLocation(sBlitProg, "aPos");
            sBlitTex = glGetAttribLocation(sBlitProg, "aTex");
            glUseProgram(sBlitProg);
            glUniform1i(glGetUniformLocation(sBlitProg, "uTex"), 0);
            glUniform1i(glGetUniformLocation(sBlitProg, "u3d"), 1);
            sBlitP = glGetUniformLocation(sBlitProg, "uP");
            glUniform4f(glGetUniformLocation(sBlitProg, "uSz"), (float)GC_TEX_W, 192.0f, (float)SCREEN_W / GC_W,
                        (float)SCREEN_H / 192.0f);
            glUseProgram(0);
            if (sBlitPos < 0 || sBlitTex < 0) sBlitProg = 0;
            fprintf(stderr, "vita_gl: sharp-bilinear present %s\n", sBlitProg ? "on" : "unavailable (linear blit)");
        }
        if (!sBlitProg) sBlitProg = (GLuint)-1;
    }
    if (sBlitProg != (GLuint)-1) {
        // The canvas (GC_W of its GC_TEX_W columns; row 0 = DS line 0 at the top) across the whole screen.
        static const float kPos[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
        static const float kTex[8] = {0, 1, (float)GC_W / GC_TEX_W, 1, 0, 0, (float)GC_W / GC_TEX_W, 0};
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glUseProgram(sBlitProg);
        glUniform4f(sBlitP, masked && tex3d && !mirrorNow ? 1.0f : 0.0f, (float)GC_W, 0.0f, 0.0f);
        glActiveTexture(GL_TEXTURE0 + 1);
        glBindTexture(GL_TEXTURE_2D, tex3d);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, masked ? sGcCanvas2 : sGcCanvas);
        glEnableVertexAttribArray((GLuint)sBlitPos);
        glEnableVertexAttribArray((GLuint)sBlitTex);
        glVertexAttribPointer((GLuint)sBlitPos, 2, GL_FLOAT, GL_FALSE, 0, kPos);
        glVertexAttribPointer((GLuint)sBlitTex, 2, GL_FLOAT, GL_FALSE, 0, kTex);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDisableVertexAttribArray((GLuint)sBlitPos);
        glDisableVertexAttribArray((GLuint)sBlitTex);
        glUseProgram(0);
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
        if (last && vc - last > 1) gVitaRepeats += vc - last - 1;
        last = vc;
    }
}

void vita_gl_shutdown(void) {
    // vitaGL has no teardown entry point - GXM and its pools are reclaimed when the process exits, which on the
    // Vita is the only way out of the game. Just mark the context down so present becomes a no-op.
    sReady = 0;
}
