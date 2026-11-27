// The GPU compose shader (GLSL ES 1.00): engine A's per-pixel layer priority, windows, colour effects, the 3D
// layer and master brightness - compose_line_fast + render_line's tail in gpu2d.c, in float arithmetic that is
// exact for these small integers (every division is by a power of two and is floored like the C shifts).
// Inputs (gpu2d.h Gpu2dGc), all 256x192 LUMINANCE_ALPHA textures holding the 16-bit values (L = low byte):
//   uBg0..uBg3 BG lines, uObj sprite colours, uAux window mask + sprite attributes; uPar the per-line parameters
//   (GPU2D_GC_PAR RGBA texels per line, 8x192); u3d the 3D colour buffer (RGBA8; 6-bit colour = byte >> 2,
//   alpha = byte >> 3), DS line y in row 191 - y as the GL renderer leaves it; uOvl the FPS overlay rows.
// u3d holds u3dCols columns of 3D (in a texture u3dTexW wide) spread across the uW canvas columns.
// It is drawn over a uW x 192 target (uW = 256, or wider for a 16:9 screen): gl_FragCoord gives the pixel, row =
// DS line. Wider than 256: the 3D layer (rendered with a 16:9 camera) is stretched across the whole width, the 2D
// layers stay 1:1 in the middle 256 columns. Beside them: the 3D view, the window mask of the nearest edge column,
// and only BG layers that are one opaque colour across the line (uPar texel 5 alpha: a curtain or solid background
// carries on; a HUD does not); uLogo: the title screen's logo (the top screen's BG2, RGBA 256x192), drawn 1:1
// in the 2D area over the 3D / backdrop, under the main screen's own 2D layers and before its brightness; during the iris wipe (texel 5: radius, on) the circle continues there (outside: black). (Identifiers avoid s, t, st,
// ts, ss, tt: vitaGL's GLSL-to-Cg header defines them as macros.)
#ifndef OLD_GPU2D_COMPOSE_H
#define OLD_GPU2D_COMPOSE_H

static const char kOldGcVertexShader[] =
    "attribute vec2 aPos;\n"
    "void main() { gl_Position = vec4(aPos, 0.0, 1.0); }\n";

// The fragment program's body, shared by the GLSL (host check) and Cg (Vita) versions below: one entry function
// compose(fragment position) that returns the colour.
#define OLD_GC_FRAGMENT_BODY \
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
    "uniform float u3dCols;\n" \
    "uniform float u3dTexW;\n" \
    "uniform sampler2D uLogo;\n" \
    "uniform float uLogoOn;\n" \
    "uniform float uMirror;\n" \
    "uniform vec2 uLogoOfs;\n" \
    "uniform float uBg3Shift;\n" \
    "uniform vec4 uObjMove;\n" \
    "uniform float uObjShift;\n" \
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
    "  float x3 = floor((cx + 0.5) * u3dCols / uW);\n" \
    "  vec2 uv = vec2((x + 0.5) / 256.0, (y + 0.5) / 192.0);\n" \
    "  if (uMirror > 0.5) return in2d > 0.5 ? vec4(texture2D(uLogo, uv).rgb, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);\n" \
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
    "  vec2 euv = vec2(x < 0.0 ? 0.5 / 256.0 : 255.5 / 256.0, uv.y);\n" \
    "  vec4 p5 = texture2D(uPar, vec2(5.5 / 8.0, py));\n" \
    "  float uni = by(p5.a);\n" \
    "  float sideM = by(texture2D(uAux, euv).r);\n" \
    "  if (in2d < 0.5 && by(p5.b) > 0.5) {\n" \
    "    float ir = by(p5.r) + by(p5.g) * 256.0, dd = floor(abs(2.0 * y + 1.0 - 192.0) / 2.0);\n" \
    "    float hr = dd <= ir ? floor(sqrt(ir * ir - dd * dd)) : -1.0;\n" \
    "    if (!(hr >= 0.0 && x >= 128.0 - hr && x < 128.0 + hr)) return vec4(0.0, 0.0, 0.0, 1.0);\n" \
    "    sideM = 33.0;\n" \
    "  }\n" \
    "  vec2 luv = in2d > 0.5 ? uv : euv;\n" \
    "  float use0 = in2d > 0.5 ? 1.0 : bitOf(uni, 1.0), use1 = in2d > 0.5 ? 1.0 : bitOf(uni, 2.0);\n" \
    "  float use2 = in2d > 0.5 ? 1.0 : bitOf(uni, 4.0), use3 = in2d > 0.5 ? 1.0 : bitOf(uni, 8.0);\n" \
    "  vec2 uv3 = luv;\n" \
    "  if (uBg3Shift > 0.5) {\n" \
    "    float x3b = x - uBg3Shift;\n" \
    "    use3 = x3b >= 0.0 && x3b < 256.0 ? 1.0 : 0.0;\n" \
    "    uv3 = vec2((x3b + 0.5) / 256.0, uv.y);\n" \
    "  }\n" \
    "  float m = in2d > 0.5 ? by(ax.r) : sideM, oa = in2d > 0.5 ? by(ax.a) : 0.0;\n" \
    "  float k1 = 100.0, l1 = 5.0, a1 = 0.0; vec3 c1 = bd;\n" \
    "  float k2 = 200.0, l2 = 7.0; vec3 c2 = bd;\n" \
    "  float k; vec3 c;\n" \
    "  vec2 ouv = uv; float oin = in2d, om = m;\n" \
    "  if (uObjShift > 0.5 && y >= uObjMove.y && y < uObjMove.w) {\n" \
    "    float xs = x - uObjShift;\n" \
    "    if (xs >= uObjMove.x && xs < uObjMove.z) {\n" \
    "      ouv = vec2((xs + 0.5) / 256.0, uv.y); oin = 1.0;\n" \
    "      vec4 axs = texture2D(uAux, ouv); om = by(axs.r); oa = by(axs.a);\n" \
    "    } else if (x >= uObjMove.x && x < uObjMove.z) oin = 0.0;\n" \
    "  }\n" \
    "  vec4 tv = texture2D(uObj, ouv);\n" \
    "  if (oin > 0.5 && objOn > 0.5 && by(tv.a) >= 128.0 && bitOf(om, 16.0) > 0.5) {\n" \
    "    k = gmod(oa, 4.0) * 8.0; c = col15(by(tv.r), by(tv.a));\n" \
    "    k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 4.0; a1 = 0.0;\n" \
    "  }\n" \
    "  if (pr0 < 3.5 && bitOf(m, 1.0) > 0.5) {\n" \
    "    float on = 0.0, al = 0.0;\n" \
    "    if (on3d > 0.5) {\n" \
    "      float sx = x3 + floor(hofs * u3dCols / 256.0);\n" \
    "      if (uHave3d > 0.5 && sx >= 0.0 && sx < u3dCols) {\n" \
    "        vec4 q = texture2D(u3d, vec2((sx + 0.5) / u3dTexW, (191.0 - y + 0.5) / 192.0));\n" \
    "        al = floor(by(q.a) / 8.0);\n" \
    "        c = floor(vec3(by(q.r), by(q.g), by(q.b)) / 4.0);\n" \
    "        on = al > 0.5 ? 1.0 : 0.0;\n" \
    "      }\n" \
    "    } else {\n" \
    "      tv = texture2D(uBg0, luv); on = use0 > 0.5 && by(tv.a) >= 128.0 ? 1.0 : 0.0; c = col15(by(tv.r), by(tv.a));\n" \
    "    }\n" \
    "    if (on > 0.5) {\n" \
    "      k = pr0 * 8.0 + 1.0;\n" \
    "      if (k < k1) { k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 0.0; a1 = al; }\n" \
    "      else if (k < k2) { k2 = k; c2 = c; l2 = 0.0; }\n" \
    "    }\n" \
    "  }\n" \
    "  if (pr1 < 3.5 && bitOf(m, 2.0) > 0.5 && use1 > 0.5) {\n" \
    "    tv = texture2D(uBg1, luv);\n" \
    "    if (by(tv.a) >= 128.0) {\n" \
    "      k = pr1 * 8.0 + 2.0; c = col15(by(tv.r), by(tv.a));\n" \
    "      if (k < k1) { k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 1.0; a1 = 0.0; }\n" \
    "      else if (k < k2) { k2 = k; c2 = c; l2 = 1.0; }\n" \
    "    }\n" \
    "  }\n" \
    "  if (pr2 < 3.5 && bitOf(m, 4.0) > 0.5 && use2 > 0.5) {\n" \
    "    tv = texture2D(uBg2, luv);\n" \
    "    if (by(tv.a) >= 128.0) {\n" \
    "      k = pr2 * 8.0 + 3.0; c = col15(by(tv.r), by(tv.a));\n" \
    "      if (k < k1) { k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 2.0; a1 = 0.0; }\n" \
    "      else if (k < k2) { k2 = k; c2 = c; l2 = 2.0; }\n" \
    "    }\n" \
    "  }\n" \
    "  if (pr3 < 3.5 && bitOf(m, 8.0) > 0.5 && use3 > 0.5) {\n" \
    "    tv = texture2D(uBg3, uv3);\n" \
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
    "  if (uLogoOn > 0.5 && (l1 > 4.5 || (l1 < 0.5 && on3d > 0.5))) {\n" \
    "    vec2 lu = uv - uLogoOfs / vec2(256.0, 192.0);\n" \
    "    vec4 lg = texture2D(uLogo, lu);\n" \
    "    if (lg.a > 0.5 && lu.x > 0.0 && lu.x < 1.0 && lu.y > 0.0 && lu.y < 1.0) o = floor(lg.rgb * 63.0 + 0.5);\n" \
    "  }\n" \
    "  vec3 o8 = (o * 4.0 + floor(o / 16.0)) / 255.0;\n" \
    "  if (uOvlOn > 0.5 && y < 16.0 && in2d > 0.5) {\n" \
    "    vec4 ov = texture2D(uOvl, vec2(uv.x, (y + 0.5) / 16.0));\n" \
    "    if (ov.a > 0.5) o8 = ov.rgb;\n" \
    "  }\n" \
    "  return vec4(o8, 1.0);\n" \
    "}\n"

#define OLD_GC_GMOD "float gmod(float a, float b) { return a - b * floor(a / b); }\n"
static const char kOldGcFragmentShader[] =   // GLSL ES 1.00
    "precision highp float;\n" OLD_GC_GMOD OLD_GC_FRAGMENT_BODY
    "void main() { gl_FragColor = compose(gl_FragCoord.xy); }\n";
// Cg for vitaGL (GL_CG_*_SHADER_EXT: compiled by the Cg compiler as is, not through vitaGL's GLSL translator).
static const char kOldGcFragmentShaderCg[] =
    "#define vec2 float2\n#define vec3 float3\n#define vec4 float4\n#define texture2D tex2D\n" OLD_GC_GMOD OLD_GC_FRAGMENT_BODY
    "float4 main(float4 fc : WPOS) : COLOR { return compose(fc.xy); }\n";
static const char kOldGcVertexShaderCg[] =
    "void main(float2 aPos, out float4 oPos : POSITION) { oPos = float4(aPos, 0.0, 1.0); }\n";

// The canvas to the screen, "sharp bilinear": each canvas pixel a crisp block, only the block edges blended (the
// 192 lines go to 544, a non-integer 2.83x, which plain linear filtering blurs and nearest makes uneven).
// The 3D at the screen's resolution: where a canvas pixel is exactly the compose's sample of the 3D (u3d, the 960x544
// colour buffer; the compose read it at the middle of the pixel's block, as below) with nothing over or blended into
// it, the screen pixel takes the 3D pixel under it instead, in the same DS colour depth. 2D and anything drawn or
// blended over the 3D keep the composed pixel. The master brightness (the fades) is applied here, after that, to
// whichever it is - so a fade darkens the full-resolution picture (the compose leaves it out). uP: x = 3D on,
// y = the canvas columns (GC_W), z = brightness on. uPar: the compose's per-line parameters (texel 2: b = mode, a = factor).
static const char kOldBlitVertexShaderCg[] =
    "void main(float2 aPos, float2 aTex, out float4 oPos : POSITION, out float2 vTex : TEXCOORD0) {\n"
    "  oPos = float4(aPos, 0.0, 1.0); vTex = aTex; }\n";
static const char kOldBlitFragmentShaderCg[] =
    "float4 main(float2 vTex : TEXCOORD0, uniform sampler2D uTex, uniform sampler2D u3d, uniform float4 uSz,\n"
    "            uniform sampler2D uPar, uniform float4 uP) : COLOR {\n"
    "  float2 tx = vTex * uSz.xy;\n"
    "  float2 fl = floor(tx);\n"
    "  float2 d = tx - fl - 0.5;\n"
    "  float2 r = 0.5 - 0.5 / uSz.zw;\n"
    "  float2 f = (d - clamp(d, -r, r)) * uSz.zw + 0.5;\n"
    "  float3 o = tex2D(uTex, (fl + f) / uSz.xy).rgb;\n"
    "  if (uP.x > 0.5) {\n"
    "    float3 cv = floor(tex2D(uTex, (fl + 0.5) / uSz.xy).rgb * 255.0 + 0.5);\n"
    "    float4 lo = floor(tex2D(u3d, float2((fl.x + 0.5) / uP.y, (191.0 - fl.y + 0.5) / 192.0)) * 255.0 + 0.5);\n"
    "    float4 hi = floor(tex2D(u3d, float2(tx.x / uP.y, 1.0 - vTex.y)) * 255.0 + 0.5);\n"
    "    float3 c6 = floor(lo.rgb / 4.0);\n"
    "    float3 dd = abs(cv - (c6 * 4.0 + floor(c6 / 16.0)));\n"
    "    if (lo.a > 247.5 && hi.a > 247.5 && dd.x + dd.y + dd.z < 0.5) {\n"
    "      float3 h6 = floor(hi.rgb / 4.0);\n"
    "      o = (h6 * 4.0 + floor(h6 / 16.0)) / 255.0;\n"
    "    }\n"
    "  }\n"
    "  float4 mb = floor(tex2D(uPar, float2(2.5 / 8.0, (fl.y + 0.5) / 192.0)) * 255.0 + 0.5);\n"
    "  if (uP.z > 0.5 && mb.a > 0.5 && mb.b > 0.5 && mb.b < 2.5) {\n"
    "    float3 c = floor(floor(o * 255.0 + 0.5) / 4.0);\n"
    "    c = mb.b < 1.5 ? c + floor(((63.0 - c) * mb.a) / 16.0) : c - floor((c * mb.a + 15.0) / 16.0);\n"
    "    o = (c * 4.0 + floor(c / 16.0)) / 255.0;\n"
    "  }\n"
    "  return float4(o, 1.0);\n"
    "}\n";

#endif
