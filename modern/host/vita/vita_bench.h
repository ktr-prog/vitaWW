// TEST BUILD ONLY: GPU timings measured in the probe frames (vita_gl3d.c starts a probe) and logged. Each item:
// drawn three times with the GPU waited for before and after, the fastest kept. Included by vita_gl.c (uses its
// statics). The compose items draw into the canvas before the real compose (which then overwrites it); the present
// steps draw into their real targets before the real present (which redraws them).
//  - the compose (old one, the current one) and five diagnostic shaders that tell texture reads from arithmetic
//    and whether a branch not taken saves its cost on this GPU
//  - the present's steps
#include "vita_bench_old.h"
static const char kBenchVertexCg[] =
    "void main(float2 aPos, out float4 oPos : POSITION) { oPos = float4(aPos, 0.0, 1.0); }\n";
#define BENCH_UV "  float2 uv = float2((floor(fc.x) - 41.0 + 0.5) / 256.0, (floor(fc.y) + 0.5) / 192.0);\n"
static const char kBench3dCg[] =   // the 3D sample only
    "float4 main(float4 fc : WPOS, uniform sampler2D u3d) : COLOR {\n" BENCH_UV
    "  return tex2D(u3d, float2((floor(fc.x) + 0.5) / 339.0, uv.y));\n}\n";
static const char kBenchParCg[] =   // + the six line-parameter reads
    "float4 main(float4 fc : WPOS, uniform sampler2D u3d, uniform sampler2D uLp) : COLOR {\n" BENCH_UV
    "  float4 s = tex2D(u3d, float2((floor(fc.x) + 0.5) / 339.0, uv.y));\n"
    "  for (int i = 0; i < 6; i++) s += tex2D(uLp, float2((i + 0.5) / 8.0, uv.y));\n"
    "  return s;\n}\n";
static const char kBenchFetchCg[] =   // every texture read the compose makes, summed (no arithmetic)
    "float4 main(float4 fc : WPOS, uniform sampler2D u3d, uniform sampler2D uLp, uniform sampler2D uBg0,\n"
    "            uniform sampler2D uBg1, uniform sampler2D uBg2, uniform sampler2D uBg3, uniform sampler2D uObj,\n"
    "            uniform sampler2D uAux, uniform sampler2D uLogo) : COLOR {\n" BENCH_UV
    "  float4 s = tex2D(u3d, float2((floor(fc.x) + 0.5) / 339.0, uv.y));\n"
    "  for (int i = 0; i < 6; i++) s += tex2D(uLp, float2((i + 0.5) / 8.0, uv.y));\n"
    "  s += tex2D(uBg0, uv) + tex2D(uBg1, uv) + tex2D(uBg2, uv) + tex2D(uBg3, uv) + tex2D(uObj, uv);\n"
    "  s += tex2D(uAux, uv) + tex2D(uAux, float2(0.5 / 256.0, uv.y)) + tex2D(uLogo, uv);\n"
    "  return s;\n}\n";
#define BENCH_HEAVY \
    "    float4 a = q;\n" \
    "    for (int i = 0; i < 48; i++) a = floor(a * 1.37 + 0.5) - floor(a / 3.0);\n" \
    "    q = a;\n"
static const char kBenchBranchCg[] =   // 3D sample + ~200 operations in a branch never taken
    "float4 main(float4 fc : WPOS, uniform sampler2D u3d) : COLOR {\n" BENCH_UV
    "  float4 q = tex2D(u3d, float2((floor(fc.x) + 0.5) / 339.0, uv.y));\n"
    "  if (q.a > 2.0) {\n" BENCH_HEAVY "  }\n"
    "  return q;\n}\n";
static const char kBenchAluCg[] =   // the same operations always run
    "float4 main(float4 fc : WPOS, uniform sampler2D u3d) : COLOR {\n" BENCH_UV
    "  float4 q = tex2D(u3d, float2((floor(fc.x) + 0.5) / 339.0, uv.y));\n"
    "  {\n" BENCH_HEAVY "  }\n"
    "  return q;\n}\n";

enum { BENCH_N = 5 };
static GLuint sBenchOld, sBenchDiag[BENCH_N];
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
        var = best_ / 1000;                                                         \
    } while (0)

static void bench_quad(GLuint prog) {
    GLint a0 = glGetAttribLocation(prog, "aPos");
    glUseProgram(prog);
    glEnableVertexAttribArray((GLuint)a0);
    glVertexAttribPointer((GLuint)a0, 2, GL_FLOAT, GL_FALSE, 0, kFullPos);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableVertexAttribArray((GLuint)a0);
}

static uint64_t sBenchC[2 + BENCH_N];
static void bench_run(const Gpu2dGc *gc, unsigned tex3d, int mirrorNow) {
    (void)gc; (void)tex3d; (void)mirrorNow;
    if (sBenchState < 0) return;
    if (!sBenchState) {
        sBenchState = -1;
        fprintf(stderr, "bench: compiling the variants (first start only)\n");
        sBenchOld = cached_program(kOldGcVertexShaderCg, kOldGcFragmentShaderCg, "bench-old");
        const char *src[BENCH_N] = {kBench3dCg, kBenchParCg, kBenchFetchCg, kBenchBranchCg, kBenchAluCg};
        const char *nm[BENCH_N] = {"bench-3d", "bench-par", "bench-fetch", "bench-branch", "bench-alu"};
        for (int i = 0; i < BENCH_N; i++) {
            sBenchDiag[i] = cached_program(kBenchVertexCg, src[i], nm[i]);
            if (!sBenchDiag[i]) { fprintf(stderr, "bench: %s failed; no timings\n", nm[i]); return; }
            gc_static_uniforms(sBenchDiag[i]);
        }
        if (!sBenchOld) { fprintf(stderr, "bench: old compose failed; no timings\n"); return; }
        gc_static_uniforms(sBenchOld);
        sBenchState = 1;
        fprintf(stderr, "bench: ready\n");
    }
    if (!sGcLogoTex) {   // (the fetch diagnostic reads the logo unit: never texture 0)
        glActiveTexture(GL_TEXTURE0 + 9);
        glBindTexture(GL_TEXTURE_2D, sGcTex[gc->set & 1][7]);
        glActiveTexture(GL_TEXTURE0);
    }
    // compose items, into the canvas (the units 0..10 are bound as for the real compose)
    glBindFramebuffer(GL_FRAMEBUFFER, sGcFbo);
    glViewport(0, 0, GC_W, 192);
    gc_frame_uniforms(sBenchOld);
    gc_frame_uniforms(sGcProg);
    BENCH_TIME(sBenchC[0], bench_quad(sBenchOld));
    BENCH_TIME(sBenchC[1], bench_quad(sGcProg));
    for (int i = 0; i < BENCH_N; i++) BENCH_TIME(sBenchC[2 + i], bench_quad(sBenchDiag[i]));
}

static void bench_present(const Gpu2dGc *gc, unsigned tex3d, int has3d) {
    if (sBenchState <= 0) return;
    uint64_t pre, p3d = 0, fade = 0, fin, all;
    BENCH_TIME(pre, present_pre());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, SCREEN_W, SCREEN_H);
    if (has3d) {
        BENCH_TIME(p3d, present_3d(tex3d));
        BENCH_TIME(fade, present_fades(gc));
    }
    BENCH_TIME(fin, present_final(has3d));
    BENCH_TIME(all, present_draw(gc, tex3d, has3d));
    // how much of the picture the compose marked as exactly the 3D (shown at full resolution), and the 3D's alpha
    static uint8_t px[GC_W * 192 * 4];
    glBindFramebuffer(GL_FRAMEBUFFER, sGcFbo);
    glReadPixels(0, 0, GC_W, 192, GL_RGBA, GL_UNSIGNED_BYTE, px);
    int marked = 0;
    for (int i = 0; i < GC_W * 192; i++) marked += px[i * 4 + 3] < 128;
    int a3[4] = {0, 0, 0, 0};   // the 3D buffer's alpha: 0, 1..247, 248..254, 255
    if (tex3d) {
        static uint8_t hp[SCREEN_W * 4];
        extern GLuint vita_gl3d_fbo_of(GLuint tex);
        GLuint f3 = vita_gl3d_fbo_of(tex3d);
        if (f3) {
            glBindFramebuffer(GL_FRAMEBUFFER, f3);
            for (int y = 8; y < SCREEN_H; y += 16) {
                glReadPixels(0, y, SCREEN_W, 1, GL_RGBA, GL_UNSIGNED_BYTE, hp);
                for (int x = 0; x < SCREEN_W; x++) { int a = hp[x * 4 + 3]; a3[a == 0 ? 0 : a < 248 ? 1 : a < 255 ? 2 : 3]++; }
            }
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    fprintf(stderr, "bench: full-resolution 3D on %d%% of the picture; 3D alpha (sampled): 0 %d, 1-247 %d, 248-254 %d, "
            "255 %d\n", marked * 100 / (GC_W * 192), a3[0], a3[1], a3[2], a3[3]);
    glUseProgram(0);
    fprintf(stderr, "bench (GPU, us): compose old %llu, current %llu | diagnostics: 3D read only %llu, + 6 parameter "
            "reads %llu, all the compose's reads %llu, 200 ops in a branch not taken %llu, 200 ops %llu | present: "
            "3x canvas %llu, 3D copy %llu, fades %llu, final blend %llu, all %llu\n",
            (unsigned long long)sBenchC[0], (unsigned long long)sBenchC[1], (unsigned long long)sBenchC[2],
            (unsigned long long)sBenchC[3], (unsigned long long)sBenchC[4], (unsigned long long)sBenchC[5],
            (unsigned long long)sBenchC[6], (unsigned long long)pre, (unsigned long long)p3d,
            (unsigned long long)fade, (unsigned long long)fin, (unsigned long long)all);
}
