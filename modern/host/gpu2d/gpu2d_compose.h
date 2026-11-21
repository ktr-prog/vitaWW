// The GPU compose shader (GLSL ES 1.00): engine A's per-pixel layer priority, windows, colour effects, the 3D
// layer and master brightness - compose_line_fast + render_line's tail in gpu2d.c, in float arithmetic that is
// exact for these small integers (every division is by a power of two and is floored like the C shifts).
// Inputs (gpu2d.h Gpu2dGc), all 256x192 LUMINANCE_ALPHA textures holding the 16-bit values (L = low byte):
//   uBg0..uBg3 BG lines, uObj sprite colours, uAux window mask + sprite attributes; uPar the per-line parameters
//   (GPU2D_GC_PAR RGBA texels per line, 8x192); u3d the 3D colour buffer (RGBA8; 6-bit colour = byte >> 2,
//   alpha = byte >> 3), DS line y in row 191 - y as the GL renderer leaves it; uOvl the FPS overlay rows.
// It is drawn over a uW x 192 target (uW = 256, or wider for a 16:9 screen): gl_FragCoord gives the pixel, row =
// DS line. Wider than 256: the 3D layer (rendered with a 16:9 camera) is stretched across the whole width, the 2D
// layers stay 1:1 in the middle 256 columns; to their sides there is no 2D (window mask: BG0 + effects only), and
// during the iris wipe (uPar texel 5: radius, on) the circle continues there (outside it: black, as the wipe). (Identifiers avoid s, t, st,
// ts, ss, tt: vitaGL's GLSL-to-Cg header defines them as macros.)
#ifndef GPU2D_COMPOSE_H
#define GPU2D_COMPOSE_H

static const char kGcVertexShader[] =
    "attribute vec2 aPos;\n"
    "void main() { gl_Position = vec4(aPos, 0.0, 1.0); }\n";

// The fragment program's body, shared by the GLSL (host check) and Cg (Vita) versions below: one entry function
// compose(fragment position) that returns the colour.
#define GC_FRAGMENT_BODY \
    "uniform sampler2D uBg0;\n" \
    "uniform sampler2D uBg1;\n" \
    "uniform sampler2D uBg2;\n" \
    "uniform sampler2D uBg3;\n" \
    "uniform sampler2D uObj;\n" \
    "uniform sampler2D uAux;\n" \
    "uniform sampler2D uPar;\n" \
    "uniform sampler2D u3d;\n" \
    "uniform sampler2D uOvl;\n" \
    "uniform float uHave3d;\n" \
    "uniform float uOvlOn;\n" \
    "uniform float uW;\n" \
    "float by(float tv) { return floor(tv * 255.0 + 0.5); }\n" \
    "float bitOf(float v, float p) { return gmod(floor(v / p), 2.0); }\n"   \
    "vec3 col15(float lo, float hi) {\n" \
    "  return vec3(gmod(lo, 32.0), floor(lo / 32.0) + gmod(hi, 4.0) * 8.0, gmod(floor(hi / 4.0), 32.0)) * 2.0;\n" \
    "}\n" \
    "float lay2bit(float l) { return l < 0.5 ? 1.0 : l < 1.5 ? 2.0 : l < 2.5 ? 4.0 : l < 3.5 ? 8.0 : l < 4.5 ? 16.0 : l < 5.5 ? 32.0 : 128.0; }\n" \
    "vec3 blendc(vec3 a, vec3 b, float eva, float evb) { return min(floor((a * eva + b * evb + 8.0) / 16.0), 63.0); }\n" \
    "vec3 blend3d(vec3 a, vec3 b, float al) { return floor((a * (al + 1.0) + b * (31.0 - al) + 16.0) / 32.0); }\n" \
    "vec3 brighten(vec3 c, float ev, float bias) { return c + floor(((63.0 - c) * ev + bias) / 16.0); }\n" \
    "vec3 darken(vec3 c, float ev, float bias) { return c - floor((c * ev + bias) / 16.0); }\n" \
    "vec4 compose(vec2 fc) {\n" \
    "  float cx = floor(fc.x), y = floor(fc.y);\n" \
    "  float x = cx - floor((uW - 256.0) / 2.0);\n" \
    "  float in2d = x >= 0.0 && x < 256.0 ? 1.0 : 0.0;\n" \
    "  float x3 = floor((cx + 0.5) * 256.0 / uW);\n" \
    "  vec2 uv = vec2((x + 0.5) / 256.0, (y + 0.5) / 192.0);\n" \
    "  float py = (y + 0.5) / 192.0;\n" \
    "  vec4 p0 = texture2D(uPar, vec2(0.5 / 8.0, py));\n" \
    "  vec4 p1 = texture2D(uPar, vec2(1.5 / 8.0, py));\n" \
    "  vec4 p2 = texture2D(uPar, vec2(2.5 / 8.0, py));\n" \
    "  vec4 p3 = texture2D(uPar, vec2(3.5 / 8.0, py));\n" \
    "  vec4 p4 = texture2D(uPar, vec2(4.5 / 8.0, py));\n" \
    "  float pr0 = by(p0.r), pr1 = by(p0.g), pr2 = by(p0.b), pr3 = by(p0.a);\n" \
    "  float first = by(p1.r), second = by(p1.g), fl = by(p1.b), evy = by(p1.a);\n" \
    "  float effect = gmod(fl, 4.0), objOn = bitOf(fl, 4.0), on3d = bitOf(fl, 8.0);\n" \
    "  float eva = by(p2.r), evb = by(p2.g), mbm = by(p2.b), mbf = by(p2.a);\n" \
    "  vec3 bd = vec3(by(p3.r), by(p3.g), by(p3.b));\n" \
    "  float hofs = by(p4.r) + by(p4.g) * 256.0 - 256.0;\n" \
    "  vec4 ax = texture2D(uAux, uv);\n" \
    "  float sideM = 33.0;\n" \
    "  vec4 p5 = texture2D(uPar, vec2(5.5 / 8.0, py));\n" \
    "  if (in2d < 0.5 && by(p5.b) > 0.5) {\n" \
    "    float ir = by(p5.r) + by(p5.g) * 256.0, dd = floor(abs(2.0 * y + 1.0 - 192.0) / 2.0);\n" \
    "    float hr = dd <= ir ? floor(sqrt(ir * ir - dd * dd)) : -1.0;\n" \
    "    if (!(hr >= 0.0 && x >= 128.0 - hr && x < 128.0 + hr)) return vec4(0.0, 0.0, 0.0, 1.0);\n" \
    "  }\n" \
    "  float m = in2d > 0.5 ? by(ax.r) : sideM, oa = in2d > 0.5 ? by(ax.a) : 0.0;\n" \
    "  float k1 = 100.0, l1 = 5.0, a1 = 0.0; vec3 c1 = bd;\n" \
    "  float k2 = 200.0, l2 = 7.0; vec3 c2 = bd;\n" \
    "  float k; vec3 c;\n" \
    "  vec4 tv = texture2D(uObj, uv);\n" \
    "  if (in2d > 0.5 && objOn > 0.5 && by(tv.a) >= 128.0 && bitOf(m, 16.0) > 0.5) {\n" \
    "    k = gmod(oa, 4.0) * 8.0; c = col15(by(tv.r), by(tv.a));\n" \
    "    k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 4.0; a1 = 0.0;\n" \
    "  }\n" \
    "  if (pr0 < 3.5 && bitOf(m, 1.0) > 0.5) {\n" \
    "    float on = 0.0, al = 0.0;\n" \
    "    if (on3d > 0.5) {\n" \
    "      float sx = x3 + hofs;\n" \
    "      if (uHave3d > 0.5 && sx >= 0.0 && sx < 256.0) {\n" \
    "        vec4 q = texture2D(u3d, vec2((sx + 0.5) / 256.0, (191.0 - y + 0.5) / 192.0));\n" \
    "        al = floor(by(q.a) / 8.0);\n" \
    "        c = floor(vec3(by(q.r), by(q.g), by(q.b)) / 4.0);\n" \
    "        on = al > 0.5 ? 1.0 : 0.0;\n" \
    "      }\n" \
    "    } else {\n" \
    "      tv = texture2D(uBg0, uv); on = in2d > 0.5 && by(tv.a) >= 128.0 ? 1.0 : 0.0; c = col15(by(tv.r), by(tv.a));\n" \
    "    }\n" \
    "    if (on > 0.5) {\n" \
    "      k = pr0 * 8.0 + 1.0;\n" \
    "      if (k < k1) { k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 0.0; a1 = al; }\n" \
    "      else if (k < k2) { k2 = k; c2 = c; l2 = 0.0; }\n" \
    "    }\n" \
    "  }\n" \
    "  if (pr1 < 3.5 && bitOf(m, 2.0) > 0.5) {\n" \
    "    tv = texture2D(uBg1, uv);\n" \
    "    if (by(tv.a) >= 128.0) {\n" \
    "      k = pr1 * 8.0 + 2.0; c = col15(by(tv.r), by(tv.a));\n" \
    "      if (k < k1) { k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 1.0; a1 = 0.0; }\n" \
    "      else if (k < k2) { k2 = k; c2 = c; l2 = 1.0; }\n" \
    "    }\n" \
    "  }\n" \
    "  if (pr2 < 3.5 && bitOf(m, 4.0) > 0.5) {\n" \
    "    tv = texture2D(uBg2, uv);\n" \
    "    if (by(tv.a) >= 128.0) {\n" \
    "      k = pr2 * 8.0 + 3.0; c = col15(by(tv.r), by(tv.a));\n" \
    "      if (k < k1) { k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 2.0; a1 = 0.0; }\n" \
    "      else if (k < k2) { k2 = k; c2 = c; l2 = 2.0; }\n" \
    "    }\n" \
    "  }\n" \
    "  if (pr3 < 3.5 && bitOf(m, 8.0) > 0.5) {\n" \
    "    tv = texture2D(uBg3, uv);\n" \
    "    if (by(tv.a) >= 128.0) {\n" \
    "      k = pr3 * 8.0 + 4.0; c = col15(by(tv.r), by(tv.a));\n" \
    "      if (k < k1) { k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 3.0; a1 = 0.0; }\n" \
    "      else if (k < k2) { k2 = k; c2 = c; l2 = 3.0; }\n" \
    "    }\n" \
    "  }\n" \
    "  vec3 o = c1;\n" \
    "  float below2nd = bitOf(second, lay2bit(l2));\n" \
    "  float kind = gmod(floor(oa / 4.0), 4.0), oal = floor(oa / 16.0);\n" \
    "  if (l1 > 3.5 && l1 < 4.5 && kind > 0.5 && below2nd > 0.5) {\n" \
    "    if (kind > 1.5) o = blendc(c1, c2, oal + 1.0, 15.0 - oal); else o = blendc(c1, c2, eva, evb);\n" \
    "  } else if (a1 > 0.5 && below2nd > 0.5) {\n" \
    "    if (a1 < 30.5) o = blend3d(c1, c2, a1);\n" \
    "  } else if (bitOf(m, 32.0) > 0.5 && bitOf(first, lay2bit(l1)) > 0.5) {\n" \
    "    if (effect > 0.5 && effect < 1.5) { if (below2nd > 0.5) o = blendc(c1, c2, eva, evb); }\n" \
    "    else if (effect > 1.5 && effect < 2.5) o = brighten(c1, evy, 8.0);\n" \
    "    else if (effect > 2.5) o = darken(c1, evy, 7.0);\n" \
    "  }\n" \
    "  if (mbm > 0.5 && mbm < 1.5 && mbf > 0.5) o = brighten(o, mbf, 0.0);\n" \
    "  else if (mbm > 1.5 && mbm < 2.5 && mbf > 0.5) o = darken(o, mbf, 15.0);\n" \
    "  vec3 o8 = (o * 4.0 + floor(o / 16.0)) / 255.0;\n" \
    "  if (uOvlOn > 0.5 && y < 16.0 && in2d > 0.5) {\n" \
    "    vec4 ov = texture2D(uOvl, vec2(uv.x, (y + 0.5) / 16.0));\n" \
    "    if (ov.a > 0.5) o8 = ov.rgb;\n" \
    "  }\n" \
    "  return vec4(o8, 1.0);\n" \
    "}\n"

#define GC_GMOD "float gmod(float a, float b) { return a - b * floor(a / b); }\n"
static const char kGcFragmentShader[] =   // GLSL ES 1.00
    "precision highp float;\n" GC_GMOD GC_FRAGMENT_BODY
    "void main() { gl_FragColor = compose(gl_FragCoord.xy); }\n";
// Cg for vitaGL (GL_CG_*_SHADER_EXT: compiled by the Cg compiler as is, not through vitaGL's GLSL translator).
static const char kGcFragmentShaderCg[] =
    "#define vec2 float2\n#define vec3 float3\n#define vec4 float4\n#define texture2D tex2D\n" GC_GMOD GC_FRAGMENT_BODY
    "float4 main(float4 fc : WPOS) : COLOR { return compose(fc.xy); }\n";
static const char kGcVertexShaderCg[] =
    "void main(float2 aPos, out float4 oPos : POSITION) { oPos = float4(aPos, 0.0, 1.0); }\n";

#endif
