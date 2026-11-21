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

// Present one frame: upload the composited DS canvas (XRGB8888 == GL_BGRA bytes, a zero-conversion upload) and
// blit the sub-rectangle [sx,sy,sw,sh] of it to the screen, scaled to FIT (letterbox, 4:3, black bars).
void vita_gl_present_canvas(const void *pixels, int canvas_w, int canvas_h, int sx, int sy, int sw, int sh) {
    if (!sReady) return;

    ensure_tex(canvas_w, canvas_h, pixels);

    // FIT: largest scale that still fits both axes (letterbox)
    float scale = (float)SCREEN_W / sw;
    if ((float)SCREEN_H / sh < scale) scale = (float)SCREEN_H / sh;
    int dw = (int)(sw * scale + 0.5f), dh = (int)(sh * scale + 0.5f);
    int dx = (SCREEN_W - dw) / 2, dy = (SCREEN_H - dh) / 2;

    // Clear the whole screen black (precompiled clear shader), then blit the canvas region onto it. The read
    // source is our FBO (canvas texture); the draw target is the default framebuffer (the screen). vitaGL's blit
    // samples the FBO's attachment and flips V so the top-of-canvas row lands at the top of the screen.
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, sFbo);
    // vitaGL flips V when reading an FBO as a texture, which would show the canvas upside-down; invert the source
    // Y range (srcY0=bottom, srcY1=top) to cancel that flip so row 0 of the canvas lands at the top of the screen.
    glBlitFramebuffer(sx, sy + sh, sx + sw, sy,
                      dx, dy, dx + dw, dy + dh,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);

    vglSwapBuffers(GL_FALSE);
}

void vita_gl_shutdown(void) {
    // vitaGL has no teardown entry point - GXM and its pools are reclaimed when the process exits, which on the
    // Vita is the only way out of the game. Just mark the context down so present becomes a no-op.
    sReady = 0;
}
