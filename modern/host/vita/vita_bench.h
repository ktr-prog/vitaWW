// TEST BUILD ONLY: GPU timings of compose and present variants, measured in the probe frames (vita_gl3d.c starts a
// probe) and logged. Each variant: drawn three times with the GPU waited for before and after, the fastest kept.
// Included by vita_gl.c (uses its statics). The compose variants draw into the canvas before the real compose
// (which then overwrites it); the present variants draw into an offscreen 960x544 picture, not the screen.
#include "vita_bench_old.h"
static const char kBenchBlitqVertexCg[] =
    "void main(float2 aPos, float2 aTex, float2 aT3, out float4 oPos : POSITION,\n"
    "          out float2 vTex : TEXCOORD0, out float2 vT3 : TEXCOORD1) {\n"
    "  oPos = float4(aPos, 0.0, 1.0); vTex = aTex; vT3 = aT3; }\n";
static const char kBenchBlitqFragmentCg[] =
    "float4 main(float2 vTex : TEXCOORD0, float2 vT3 : TEXCOORD1, uniform sampler2D uTex,\n"
    "            uniform sampler2D u3d, uniform sampler2D uLp, uniform float4 uSz, uniform float4 uP) : COLOR {\n"
    "  float2 tx = vTex * uSz.xy;\n"
    "  float2 fl = floor(tx);\n"
    "  float2 d = tx - fl - 0.5;\n"
    "  float2 r = 0.5 - 0.5 / uSz.zw;\n"
    "  float2 f2 = (d - clamp(d, -r, r)) * uSz.zw + 0.5;\n"
    "  float4 cv = tex2D(uTex, (fl + f2) / uSz.xy);\n"
    "  float4 h = tex2D(u3d, vT3);\n"
    "  float2 mb = floor(tex2D(uLp, float2(2.5 / 8.0, (fl.y + 0.5) / 192.0)).zw * 255.0 + 0.5);\n"
    "  float3 c = floor(h.rgb * 63.75 + 0.125);\n"
    "  float3 cb = mb.x < 1.5 ? c + floor(((63.0 - c) * mb.y) / 16.0) : c - floor((c * mb.y + 15.0) / 16.0);\n"
    "  c = mb.x > 0.5 && mb.x < 2.5 && mb.y > 0.5 ? cb : c;\n"
    "  float3 h8 = (c * 4.0 + floor(c / 16.0)) / 255.0;\n"
    "  float f = uP.x > 0.5 && h.a > 0.016 ? cv.a : 1.0;\n"
    "  return float4(lerp(h8, cv.rgb, f), 1.0);\n"
    "}\n";
static const char kBenchTrivVertexCg[] =
    "void main(float2 aPos, float2 aTex, out float4 oPos : POSITION, out float2 vTex : TEXCOORD0) {\n"
    "  oPos = float4(aPos, 0.0, 1.0); vTex = aTex; }\n";
static const char kBenchTrivFragmentCg[] =
    "float4 main(float2 vTex : TEXCOORD0, uniform sampler2D uTex) : COLOR { return tex2D(uTex, vTex); }\n";

static GLuint sBenchOld, sBenchLines, sBenchBlitq, sBenchTriv, sBenchTex, sBenchFbo;
static int sBenchState;   // 0 untried, 1 ready, -1 failed

#define BENCH_TIME(var, code)                                                       \
    do {                                                                            \
        uint64_t best_ = ~0ull;                                                     \
        for (int r_ = 0; r_ < 3; r_++) {                                            \
            glFinish();                                                             \
            uint64_t t_ = host_plat_time_ns();                                      \
            code;                                                                   \
            glFinish();                                                             \
            uint64_t d_ = host_plat_time_ns() - t_;                                 \
            if (d_ < best_) best_ = d_;                                             \
        }                                                                           \
        var = best_;                                                                \
    } while (0)

static void bench_quad(GLuint prog, const float *pos, const float *tex, const float *t3) {
    GLint a0 = glGetAttribLocation(prog, "aPos"), a1 = tex ? glGetAttribLocation(prog, "aTex") : -1;
    GLint a2 = t3 ? glGetAttribLocation(prog, "aT3") : -1;
    glUseProgram(prog);
    glEnableVertexAttribArray((GLuint)a0);
    glVertexAttribPointer((GLuint)a0, 2, GL_FLOAT, GL_FALSE, 0, pos);
    if (a1 >= 0) { glEnableVertexAttribArray((GLuint)a1); glVertexAttribPointer((GLuint)a1, 2, GL_FLOAT, GL_FALSE, 0, tex); }
    if (a2 >= 0) { glEnableVertexAttribArray((GLuint)a2); glVertexAttribPointer((GLuint)a2, 2, GL_FLOAT, GL_FALSE, 0, t3); }
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableVertexAttribArray((GLuint)a0);
    if (a1 >= 0) glDisableVertexAttribArray((GLuint)a1);
    if (a2 >= 0) glDisableVertexAttribArray((GLuint)a2);
}

static void bench_run(const Gpu2dGc *gc, unsigned tex3d, int mirrorNow) {
    if (sBenchState < 0) return;
    if (!sBenchState) {
        sBenchState = -1;
        fprintf(stderr, "bench: compiling the variants (first start only)\n");
        sBenchOld = cached_program(kOldGcVertexShaderCg, kOldGcFragmentShaderCg, "bench-old");
        sBenchLines = cached_program(kGcVertexShaderCg, kGcFragmentShaderCg, "compose");
        sBenchBlitq = cached_program(kBenchBlitqVertexCg, kBenchBlitqFragmentCg, "bench-blitq");
        sBenchTriv = cached_program(kBenchTrivVertexCg, kBenchTrivFragmentCg, "bench-triv");
        if (!sBenchOld || !sBenchLines || !sBenchBlitq || !sBenchTriv || sBlitProg == 0 || sBlitProg == (GLuint)-1) {
            fprintf(stderr, "bench: a variant failed to build; no timings\n");
            return;
        }
        gc_static_uniforms(sBenchOld);
        gc_static_uniforms(sBenchLines);
        glUseProgram(sBenchBlitq);
        glUniform1i(glGetUniformLocation(sBenchBlitq, "uTex"), 0);
        glUniform1i(glGetUniformLocation(sBenchBlitq, "u3d"), 1);
        glUniform1i(glGetUniformLocation(sBenchBlitq, "uLp"), 10);
        glUniform4f(glGetUniformLocation(sBenchBlitq, "uSz"), (float)GC_TEX_W, 192.0f, (float)SCREEN_W / GC_W,
                    (float)SCREEN_H / 192.0f);
        glUseProgram(sBenchTriv);
        glUniform1i(glGetUniformLocation(sBenchTriv, "uTex"), 0);
        glUseProgram(0);
        glGenTextures(1, &sBenchTex);
        glBindTexture(GL_TEXTURE_2D, sBenchTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCREEN_W, SCREEN_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glGenFramebuffers(1, &sBenchFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, sBenchFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sBenchTex, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        sBenchState = 1;
        fprintf(stderr, "bench: ready\n");
    }
    static const float kQ[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
    static const float kTexQ[8] = {0, 1, (float)GC_W / GC_TEX_W, 1, 0, 0, (float)GC_W / GC_TEX_W, 0};
    static const float kT3Q[8] = {0, 0, 1, 0, 0, 1, 1, 1};
    uint64_t cOld, cLines, cTex, pLines, pQuad, pTriv;

    // compose variants, into the canvas (the units 0..10 are bound as for the real compose)
    glBindFramebuffer(GL_FRAMEBUFFER, sGcFbo);
    glViewport(0, 0, GC_W, 192);
    gc_frame_uniforms(sBenchOld);
    gc_frame_uniforms(sBenchLines);
    gc_frame_uniforms(sGcProg);
    BENCH_TIME(cOld, bench_quad(sBenchOld, kQ, NULL, NULL));
    BENCH_TIME(cTex, bench_quad(sGcProg, kQ, NULL, NULL));
    static float gcPos[GC_VERTS][2], gcL[6][GC_VERTS][4];
    gc_line_quads(gc->par, GPU2D_GC_PAR * 4, (float)GC_W, gcPos, gcL);
    GLint la[7];
    la[0] = glGetAttribLocation(sBenchLines, "aPos");
    for (int k = 0; k < 6; k++) { char an[4] = {'a', 'L', (char)('0' + k), 0}; la[k + 1] = glGetAttribLocation(sBenchLines, an); }
    BENCH_TIME(cLines, {
        glUseProgram(sBenchLines);
        for (int k = 0; k < 7; k++)
            if (la[k] >= 0) {
                glEnableVertexAttribArray((GLuint)la[k]);
                glVertexAttribPointer((GLuint)la[k], k ? 4 : 2, GL_FLOAT, GL_FALSE, 0, k ? (const void *)gcL[k - 1] : (const void *)gcPos);
            }
        glDrawArrays(GL_TRIANGLES, 0, GC_VERTS);
        for (int k = 0; k < 7; k++) if (la[k] >= 0) glDisableVertexAttribArray((GLuint)la[k]);
    });
    // the real compose's picture for the present variants to read
    bench_quad(sGcProg, kQ, NULL, NULL);

    // present variants, into the offscreen 960x544 picture
    glBindFramebuffer(GL_FRAMEBUFFER, sBenchFbo);
    glViewport(0, 0, SCREEN_W, SCREEN_H);
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, tex3d ? tex3d : sGcCanvas);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sGcCanvas);
    static float pv[4][GC_VERTS][2];
    present_line_quads(gc->par, GPU2D_GC_PAR * 4, (float)GC_W, (float)GC_TEX_W, pv);
    glUseProgram(sBlitProg);
    glUniform4f(sBlitP, tex3d && !mirrorNow ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
    BENCH_TIME(pLines, {
        glUseProgram(sBlitProg);
        for (int k = 0; k < 4; k++) {
            glEnableVertexAttribArray((GLuint)sBlitAttr[k]);
            glVertexAttribPointer((GLuint)sBlitAttr[k], 2, GL_FLOAT, GL_FALSE, 0, pv[k]);
        }
        glDrawArrays(GL_TRIANGLES, 0, GC_VERTS);
        for (int k = 0; k < 4; k++) glDisableVertexAttribArray((GLuint)sBlitAttr[k]);
    });
    glUseProgram(sBenchBlitq);
    glUniform4f(glGetUniformLocation(sBenchBlitq, "uP"), tex3d && !mirrorNow ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
    BENCH_TIME(pQuad, bench_quad(sBenchBlitq, kQ, kTexQ, kT3Q));
    BENCH_TIME(pTriv, bench_quad(sBenchTriv, kQ, kTexQ, NULL));
    glUseProgram(0);

    // back to the real compose's state: units 0 and 1 hold BG0 and BG1, the canvas is drawn next
    GLuint *t = sGcTex[gc->set & 1];
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, t[1]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, t[0]);
    fprintf(stderr, "bench (GPU, us): compose old %llu, per-line quads %llu, line-parameter texture %llu | present "
            "(offscreen 960x544) per-line quads %llu, one quad %llu, plain copy %llu\n",
            (unsigned long long)(cOld / 1000), (unsigned long long)(cLines / 1000), (unsigned long long)(cTex / 1000),
            (unsigned long long)(pLines / 1000), (unsigned long long)(pQuad / 1000), (unsigned long long)(pTriv / 1000));
}
