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
#ifndef GPU2D_COMPOSE_H
#define GPU2D_COMPOSE_H

// Drawn as one quad per DS line: the line's parameters (decoded on the CPU from the bands' per-line parameter row,
// vita_gl.c gc_line_attribs) come in as six constant vectors, so the fragment program does not fetch and decode
// them for every pixel: L0 = BG priorities (4 = off); L1 = first/second target layers, colour effect, EVY; L2 = EVA,
// EVB, master-brightness mode, factor; L3 = backdrop (6-bit r, g, b), the 3D column offset (BG0 scroll, in canvas
// columns); L4 = sprites on, 3D on, uni (BG n one opaque colour across the line: carried beside the 2D area), iris on;
// L5.x = the iris circle's half width on this line (-1: none).
static const char kGcVertexShader[] =
    "attribute vec2 aPos;\n"
    "attribute vec4 aL0, aL1, aL2, aL3, aL4, aL5;\n"
    "varying vec4 vL0, vL1, vL2, vL3, vL4, vL5;\n"
    "void main() { gl_Position = vec4(aPos, 0.0, 1.0); vL0 = aL0; vL1 = aL1; vL2 = aL2; vL3 = aL3; vL4 = aL4; vL5 = aL5; }\n";

// The fragment program's body, shared by the GLSL (host check) and Cg (Vita) versions below: one entry function
// compose(fragment position) that returns the colour.
#define GC_FRAGMENT_BODY \
    "uniform sampler2D uBg0;\n" \
    "uniform sampler2D uBg1;\n" \
    "uniform sampler2D uBg2;\n" \
    "uniform sampler2D uBg3;\n" \
    "uniform sampler2D uObj;\n" \
    "uniform sampler2D uAux;\n" \
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
    "vec4 compose(vec2 fc, vec4 L0, vec4 L1, vec4 L2, vec4 L3, vec4 L4, vec4 L5) {\n" \
    "  L0 = floor(L0 + 0.5); L1 = floor(L1 + 0.5); L2 = floor(L2 + 0.5);\n" \
    "  L3 = floor(L3 + 0.5); L4 = floor(L4 + 0.5); L5 = floor(L5 + 0.5);\n" \
    "  float cx = floor(fc.x), y = floor(fc.y);\n" \
    "  float x = cx - floor((uW - 256.0) / 2.0);\n" \
    "  float in2d = x >= 0.0 && x < 256.0 ? 1.0 : 0.0;\n" \
    "  float x3 = floor((cx + 0.5) * u3dCols / uW);\n" \
    "  vec2 uv = vec2((x + 0.5) / 256.0, (y + 0.5) / 192.0);\n" \
    "  if (uMirror > 0.5) return in2d > 0.5 ? vec4(texture2D(uLogo, uv).rgb, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);\n" \
    "  float pr0 = L0.x, pr1 = L0.y, pr2 = L0.z, pr3 = L0.w;\n" \
    "  float first = L1.x, second = L1.y, effect = L1.z, evy = L1.w;\n" \
    "  float eva = L2.x, evb = L2.y, mbm = L2.z, mbf = L2.w;\n" \
    "  vec3 bd = L3.xyz;\n" \
    "  float objOn = L4.x, on3d = L4.y, uni = L4.z;\n" \
    "  vec4 ax = texture2D(uAux, uv);\n" \
    "  vec2 euv = vec2(x < 0.0 ? 0.5 / 256.0 : 255.5 / 256.0, uv.y);\n" \
    "  float sideM = by(texture2D(uAux, euv).r);\n" \
    "  if (in2d < 0.5 && L4.w > 0.5) {\n" \
    "    float hr = L5.x;\n" \
    "    if (!(hr >= 0.0 && x >= 128.0 - hr && x < 128.0 + hr)) {\n" \
    "      float bk = L2.z > 0.5 && L2.z < 1.5 ? floor(63.0 * L2.w / 16.0) : 0.0;\n" \
    "      return vec4(vec3(bk * 4.0 + floor(bk / 16.0)) / 255.0, 1.0);\n" \
    "    }\n" \
    "    sideM = 33.0;\n" \
    "  }\n" \
    "  float axl = by(ax.r);\n" \
    "  float covMain = in2d > 0.5 ? (axl > 127.5 ? 1.0 : 0.0) : (gmod(uni, 8.0) > 0.5 ? 1.0 : 0.0);\n" \
    "  float covBg3 = in2d > 0.5 ? bitOf(axl, 64.0) : bitOf(uni, 8.0);\n" \
    "  vec2 luv = in2d > 0.5 ? uv : euv;\n" \
    "  float use0 = in2d > 0.5 ? 1.0 : bitOf(uni, 1.0), use1 = in2d > 0.5 ? 1.0 : bitOf(uni, 2.0);\n" \
    "  float use2 = in2d > 0.5 ? 1.0 : bitOf(uni, 4.0), use3 = in2d > 0.5 ? 1.0 : bitOf(uni, 8.0);\n" \
    "  vec2 uv3 = luv;\n" \
    "  if (uBg3Shift > 0.5) {\n" \
    "    float x3b = x - uBg3Shift;\n" \
    "    use3 = x3b >= 0.0 && x3b < 256.0 ? 1.0 : 0.0;\n" \
    "    uv3 = vec2((x3b + 0.5) / 256.0, uv.y);\n" \
    "    covBg3 = use3 > 0.5 ? bitOf(by(texture2D(uAux, uv3).r), 64.0) : 0.0;\n" \
    "  }\n" \
    "  float m = in2d > 0.5 ? axl : sideM, oa = in2d > 0.5 ? by(ax.a) : 0.0;\n" \
    /* Fast path: nothing but the 3D (or, where it has no pixel, the backdrop) can show here - no 2D pixel, the \
       line's colour effects leave the 3D and the backdrop as they are (L5.y, gc_line_params), no logo, overlay or \
       translucent 3D pixel - so the result is that colour with the master brightness: the rest is skipped (a \
       branch not taken costs nothing on the Vita's GPU; the arithmetic below is what costs). */ \
    "  if (L5.y > 0.5 && covMain < 0.5 && covBg3 < 0.5 && bitOf(m, 1.0) > 0.5 && uHave3d > 0.5 &&\n" \
    "      !(uOvlOn > 0.5 && y < 16.0 && in2d > 0.5) && !(uObjShift > 0.5 && y >= uObjMove.y && y < uObjMove.w)) {\n" \
    "    vec4 q = texture2D(u3d, vec2((x3 + 0.5) / u3dTexW, (191.0 - y + 0.5) / 192.0));\n" \
    "    float qa = floor(by(q.a) / 8.0);\n" \
    "    float lgOn = 0.0;\n" \
    "    if (uLogoOn > 0.5) {\n" \
    "      vec2 qlu = uv - uLogoOfs / vec2(256.0, 192.0);\n" \
    "      vec4 qlg = texture2D(uLogo, qlu);\n" \
    "      lgOn = qlg.a > 0.5 && qlu.x > 0.0 && qlu.x < 1.0 && qlu.y > 0.0 && qlu.y < 1.0 ? 1.0 : 0.0;\n" \
    "    }\n" \
    "    if ((qa > 30.5 || qa < 0.5) && lgOn < 0.5) {\n" \
    "      vec3 qo = qa > 30.5 ? floor(vec3(by(q.r), by(q.g), by(q.b)) / 4.0) : bd;\n" \
    "      if (mbm > 0.5 && mbm < 1.5 && mbf > 0.5) qo = brighten(qo, mbf, 0.0);\n" \
    "      else if (mbm > 1.5 && mbm < 2.5 && mbf > 0.5) qo = darken(qo, mbf, 15.0);\n" \
    "      return vec4((qo * 4.0 + floor(qo / 16.0)) / 255.0, qa > 30.5 ? 0.0 : 1.0);\n" \
    "    }\n" \
    "  }\n" \
    "  float k1 = 100.0, l1 = 5.0, a1 = 0.0; vec3 c1 = bd;\n" \
    "  float k2 = 200.0, l2 = 7.0; vec3 c2 = bd;\n" \
    "  float k; vec3 c; vec4 tv;\n" \
    "  vec2 ouv = uv; float oin = in2d, om = m;\n" \
    "  if (uObjShift > 0.5 && y >= uObjMove.y && y < uObjMove.w) {\n" \
    "    covMain = 1.0;\n" \
    "    float xs = x - uObjShift;\n" \
    "    if (xs >= uObjMove.x && xs < uObjMove.z) {\n" \
    "      ouv = vec2((xs + 0.5) / 256.0, uv.y); oin = 1.0;\n" \
    "      vec4 axs = texture2D(uAux, ouv); om = by(axs.r); oa = by(axs.a);\n" \
    "    } else if (x >= uObjMove.x && x < uObjMove.z) oin = 0.0;\n" \
    "  }\n" \
    "  if (covMain > 0.5) {\n" \
    "    tv = texture2D(uObj, ouv);\n" \
    "    if (oin > 0.5 && objOn > 0.5 && by(tv.a) >= 128.0 && bitOf(om, 16.0) > 0.5) {\n" \
    "      k = gmod(oa, 4.0) * 8.0; c = col15(by(tv.r), by(tv.a));\n" \
    "      k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 4.0; a1 = 0.0;\n" \
    "    }\n" \
    "  }\n" \
    "  if (pr0 < 3.5 && bitOf(m, 1.0) > 0.5) {\n" \
    "    float on = 0.0, al = 0.0;\n" \
    "    if (on3d > 0.5) {\n" \
    "      float sx = x3 + L3.w;\n" \
    "      if (uHave3d > 0.5 && sx >= 0.0 && sx < u3dCols) {\n" \
    "        vec4 q = texture2D(u3d, vec2((sx + 0.5) / u3dTexW, (191.0 - y + 0.5) / 192.0));\n" \
    "        al = floor(by(q.a) / 8.0);\n" \
    "        c = floor(vec3(by(q.r), by(q.g), by(q.b)) / 4.0);\n" \
    "        on = al > 0.5 ? 1.0 : 0.0;\n" \
    "      }\n" \
    "    } else if (covMain > 0.5) {\n" \
    "      tv = texture2D(uBg0, luv); on = use0 > 0.5 && by(tv.a) >= 128.0 ? 1.0 : 0.0; c = col15(by(tv.r), by(tv.a));\n" \
    "    }\n" \
    "    if (on > 0.5) {\n" \
    "      k = pr0 * 8.0 + 1.0;\n" \
    "      if (k < k1) { k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 0.0; a1 = al; }\n" \
    "      else if (k < k2) { k2 = k; c2 = c; l2 = 0.0; }\n" \
    "    }\n" \
    "  }\n" \
    "  if (covMain > 0.5 && pr1 < 3.5 && bitOf(m, 2.0) > 0.5 && use1 > 0.5) {\n" \
    "    tv = texture2D(uBg1, luv);\n" \
    "    if (by(tv.a) >= 128.0) {\n" \
    "      k = pr1 * 8.0 + 2.0; c = col15(by(tv.r), by(tv.a));\n" \
    "      if (k < k1) { k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 1.0; a1 = 0.0; }\n" \
    "      else if (k < k2) { k2 = k; c2 = c; l2 = 1.0; }\n" \
    "    }\n" \
    "  }\n" \
    "  if (covMain > 0.5 && pr2 < 3.5 && bitOf(m, 4.0) > 0.5 && use2 > 0.5) {\n" \
    "    tv = texture2D(uBg2, luv);\n" \
    "    if (by(tv.a) >= 128.0) {\n" \
    "      k = pr2 * 8.0 + 3.0; c = col15(by(tv.r), by(tv.a));\n" \
    "      if (k < k1) { k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 2.0; a1 = 0.0; }\n" \
    "      else if (k < k2) { k2 = k; c2 = c; l2 = 2.0; }\n" \
    "    }\n" \
    "  }\n" \
    "  if (covBg3 > 0.5 && pr3 < 3.5 && bitOf(m, 8.0) > 0.5 && use3 > 0.5) {\n" \
    "    tv = texture2D(uBg3, uv3);\n" \
    "    if (by(tv.a) >= 128.0) {\n" \
    "      k = pr3 * 8.0 + 4.0; c = col15(by(tv.r), by(tv.a));\n" \
    "      if (k < k1) { k2 = k1; c2 = c1; l2 = l1; k1 = k; c1 = c; l1 = 3.0; a1 = 0.0; }\n" \
    "      else if (k < k2) { k2 = k; c2 = c; l2 = 3.0; }\n" \
    "    }\n" \
    "  }\n" \
    "  vec3 o = c1;\n" \
    "  float pure = l1 < 0.5 && on3d > 0.5 && L3.w == 0.0 ? 1.0 : 0.0;\n" \
    "  float below2nd = bitOf(second, lay2bit(l2));\n" \
    "  float kind = gmod(floor(oa / 4.0), 4.0), oal = floor(oa / 16.0);\n" \
    "  if (l1 > 3.5 && l1 < 4.5 && kind > 0.5 && below2nd > 0.5) {\n" \
    "    if (kind > 1.5) o = blendc(c1, c2, oal + 1.0, 15.0 - oal); else o = blendc(c1, c2, eva, evb);\n" \
    "  } else if (a1 > 0.5 && below2nd > 0.5) {\n" \
    "    if (a1 < 30.5) { o = blend3d(c1, c2, a1); pure = 0.0; }\n" \
    "  } else if (bitOf(m, 32.0) > 0.5 && bitOf(first, lay2bit(l1)) > 0.5) {\n" \
    "    if (effect > 0.5 && effect < 1.5) { if (below2nd > 0.5) { o = blendc(c1, c2, eva, evb); pure = 0.0; } }\n" \
    "    else if (effect > 1.5 && effect < 2.5) { o = brighten(c1, evy, 8.0); pure = 0.0; }\n" \
    "    else if (effect > 2.5) { o = darken(c1, evy, 7.0); pure = 0.0; }\n" \
    "  }\n" \
    "  if (uLogoOn > 0.5 && (l1 > 4.5 || (l1 < 0.5 && on3d > 0.5))) {\n" \
    "    vec2 lu = uv - uLogoOfs / vec2(256.0, 192.0);\n" \
    "    vec4 lg = texture2D(uLogo, lu);\n" \
    "    if (lg.a > 0.5 && lu.x > 0.0 && lu.x < 1.0 && lu.y > 0.0 && lu.y < 1.0) { o = floor(lg.rgb * 63.0 + 0.5); pure = 0.0; }\n" \
    "  }\n" \
    "  if (mbm > 0.5 && mbm < 1.5 && mbf > 0.5) o = brighten(o, mbf, 0.0);\n" \
    "  else if (mbm > 1.5 && mbm < 2.5 && mbf > 0.5) o = darken(o, mbf, 15.0);\n" \
    "  vec3 o8 = (o * 4.0 + floor(o / 16.0)) / 255.0;\n" \
    "  if (uOvlOn > 0.5 && y < 16.0 && in2d > 0.5) {\n" \
    "    vec4 ov = texture2D(uOvl, vec2(uv.x, (y + 0.5) / 16.0));\n" \
    "    if (ov.a > 0.5) { o8 = ov.rgb; pure = 0.0; }\n" \
    "  }\n" \
    "  return vec4(o8, 1.0 - pure);\n" \
    "}\n"

#define GC_GMOD "float gmod(float a, float b) { return a - b * floor(a / b); }\n"
static const char kGcFragmentShader[] =   // GLSL ES 1.00
    "precision highp float;\n" GC_GMOD "varying vec4 vL0, vL1, vL2, vL3, vL4, vL5;\n" GC_FRAGMENT_BODY
    "void main() { gl_FragColor = compose(gl_FragCoord.xy, vL0, vL1, vL2, vL3, vL4, vL5); }\n";
// Cg for vitaGL (GL_CG_*_SHADER_EXT: compiled by the Cg compiler as is, not through vitaGL's GLSL translator).
static const char kGcFragmentShaderCg[] =
    "#define vec2 float2\n#define vec3 float3\n#define vec4 float4\n#define texture2D tex2D\n" GC_GMOD GC_FRAGMENT_BODY
    "float4 main(float4 fc : WPOS, float4 vL0 : TEXCOORD0, float4 vL1 : TEXCOORD1, float4 vL2 : TEXCOORD2,\n"
    "            float4 vL3 : TEXCOORD3, float4 vL4 : TEXCOORD4, float4 vL5 : TEXCOORD5) : COLOR {\n"
    "  return compose(fc.xy, vL0, vL1, vL2, vL3, vL4, vL5);\n"
    "}\n";
static const char kGcVertexShaderCg[] =
    "void main(float2 aPos, float4 aL0, float4 aL1, float4 aL2, float4 aL3, float4 aL4, float4 aL5,\n"
    "          out float4 oPos : POSITION, out float4 vL0 : TEXCOORD0, out float4 vL1 : TEXCOORD1,\n"
    "          out float4 vL2 : TEXCOORD2, out float4 vL3 : TEXCOORD3, out float4 vL4 : TEXCOORD4,\n"
    "          out float4 vL5 : TEXCOORD5) {\n"
    "  oPos = float4(aPos, 0.0, 1.0); vL0 = aL0; vL1 = aL1; vL2 = aL2; vL3 = aL3; vL4 = aL4; vL5 = aL5;\n"
    "}\n";

// The canvas to the screen, "sharp bilinear": each canvas pixel a crisp block, only the block edges blended (the
// 192 lines go to 544, a non-integer 2.83x, which plain linear filtering blurs and nearest makes uneven).
// The 3D at the screen's resolution: the compose marks (alpha 0) the canvas pixels that are exactly the 3D - nothing
// drawn over or blended into it - and there the screen shows the 3D colour buffer (u3d, 960x544) pixel under it, in
// the DS colour depth, with the line's master brightness (the compose applies it to everything else). The mark is
// filtered like the colour, so the 2D's block edges blend into the full-resolution 3D. Where the 3D has no pixel at
// the screen's resolution (alpha 0) the canvas shows. Drawn as one quad per DS line: vMb = that line's brightness
// mode and factor. uP.x: 3D on. Straight-line code: every texture is read for every pixel.
static const char kBlitVertexShaderCg[] =
    "void main(float2 aPos, float2 aTex, float2 aT3, float2 aMb, out float4 oPos : POSITION,\n"
    "          out float2 vTex : TEXCOORD0, out float2 vT3 : TEXCOORD1, out float2 vMb : TEXCOORD2) {\n"
    "  oPos = float4(aPos, 0.0, 1.0); vTex = aTex; vT3 = aT3; vMb = aMb; }\n";
static const char kBlitFragmentShaderCg[] =
    "float4 main(float2 vTex : TEXCOORD0, float2 vT3 : TEXCOORD1, float2 vMb : TEXCOORD2, uniform sampler2D uTex,\n"
    "            uniform sampler2D u3d, uniform float4 uSz, uniform float4 uP) : COLOR {\n"
    "  float2 tx = vTex * uSz.xy;\n"
    "  float2 fl = floor(tx);\n"
    "  float2 d = tx - fl - 0.5;\n"
    "  float2 r = 0.5 - 0.5 / uSz.zw;\n"
    "  float2 f2 = (d - clamp(d, -r, r)) * uSz.zw + 0.5;\n"
    "  float4 cv = tex2D(uTex, (fl + f2) / uSz.xy);\n"
    "  float4 h = tex2D(u3d, vT3);\n"
    "  float2 mb = floor(vMb + 0.5);\n"
    "  float3 c = floor(h.rgb * 63.75 + 0.125);\n"
    "  float3 cb = mb.x < 1.5 ? c + floor(((63.0 - c) * mb.y) / 16.0) : c - floor((c * mb.y + 15.0) / 16.0);\n"
    "  c = mb.x > 0.5 && mb.x < 2.5 && mb.y > 0.5 ? cb : c;\n"
    "  float3 h8 = (c * 4.0 + floor(c / 16.0)) / 255.0;\n"
    "  float f = uP.x > 0.5 && h.a > 0.016 ? cv.a : 1.0;\n"
    "  return float4(lerp(h8, cv.rgb, f), 1.0);\n"
    "}\n";
// The compose's per-line inputs (kGcVertexShader L0..L5: 24 floats) from line y's parameter row p (GPU2D_GC_PAR RGBA
// texels, gpu2d.c gc_line). u3dCols: the canvas columns the 3D spreads over. The iris half width is worked out in
// float as the shader did it per pixel.
#include <math.h>
static inline void gc_line_params(const unsigned char *p, int y, float u3dCols, float *L) {
    int hofs = p[16] + p[17] * 256 - 256;
    float ir = (float)(p[20] + p[21] * 256), dd = floorf(fabsf(2.0f * (float)y + 1.0f - 192.0f) / 2.0f);
    float hr = dd <= ir ? floorf(sqrtf(ir * ir - dd * dd)) : -1.0f;
    const float v[24] = {
        p[0], p[1], p[2], p[3],
        p[4], p[5], (float)(p[6] & 3), p[7],
        p[8], p[9], p[10], p[11],
        p[12], p[13], p[14], floorf((float)hofs * u3dCols / 256.0f),
        (float)(p[6] >> 2 & 1), (float)(p[6] >> 3 & 1), p[23], p[22] ? 1.0f : 0.0f,
        hr, 0, 0, 0,
    };
    // L5.y: the compose's fast path may be taken on this line (the 3D shown, not shifted, and the line's colour effect
    // leaves a 3D or backdrop pixel unchanged: no effect, alpha blending (which never blends these two with each
    // other the way the compose resolves them), or a brightness change of 0, or neither is a first target)
    const int effect = p[6] & 3, on3d = p[6] >> 3 & 1;
    for (int i = 0; i < 24; i++) L[i] = v[i];
    if (on3d && v[15] == 0.0f && (effect <= 1 || p[7] == 0 || !(p[4] & 0x21))) L[21] = 1.0f;
}
// The compose's vertices: one quad (two triangles) per DS line over the whole canvas: GC_VERTS positions (pos) and
// the line's parameters as six arrays of 4 floats (lp[k] = attribute aLk; separate arrays, as vitaGL copies each
// attribute's stream on its own). Line y covers canvas row y (row 0 at NDC -1: the FBO convention the 3D buffer shows
// too, u3d row 191 - y).
enum { GC_VERTS = 192 * 6 };
static inline void gc_line_quads(const unsigned char *par, int parStride, float u3dCols, float (*pos)[2],
                                 float (*lp)[GC_VERTS][4]) {
    for (int y = 0; y < 192; y++) {
        float L[24];
        gc_line_params(par + y * parStride, y, u3dCols, L);
        const float y0 = -1.0f + 2.0f * (float)y / 192.0f, y1 = -1.0f + 2.0f * (float)(y + 1) / 192.0f;
        const float q[6][2] = {{-1, y0}, {1, y0}, {-1, y1}, {-1, y1}, {1, y0}, {1, y1}};
        for (int k = 0; k < 6; k++) {
            pos[y * 6 + k][0] = q[k][0]; pos[y * 6 + k][1] = q[k][1];
            for (int a = 0; a < 6; a++)
                for (int i = 0; i < 4; i++) lp[a][y * 6 + k][i] = L[a * 4 + i];
        }
    }
}
// The present's vertices: one quad per DS line over the whole screen (line y = screen rows y * 544 / 192 down), as
// four arrays of 2 floats (attributes aPos, aTex, aT3, aMb): position, canvas texture coordinate (texW: the canvas
// texture width, w: the canvas columns), 3D texture coordinate, and the line's master brightness mode and factor
// (par texel 2: bytes 10, 11).
static inline void present_line_quads(const unsigned char *par, int parStride, float w, float texW,
                                      float (*v)[GC_VERTS][2]) {
    for (int y = 0; y < 192; y++) {
        const unsigned char *p = par + y * parStride;
        const float ya = 1.0f - 2.0f * (float)y / 192.0f, yb = 1.0f - 2.0f * (float)(y + 1) / 192.0f;
        const float va = (float)y / 192.0f, vb = (float)(y + 1) / 192.0f, us = w / texW;
        const float q[6][3] = {{-1, ya, 0}, {1, ya, 1}, {-1, yb, 0}, {-1, yb, 0}, {1, ya, 1}, {1, yb, 1}};
        for (int k = 0; k < 6; k++) {
            const int n = y * 6 + k;
            const float vv = q[k][1] == ya ? va : vb;
            v[0][n][0] = q[k][0]; v[0][n][1] = q[k][1];
            v[1][n][0] = q[k][2] * us; v[1][n][1] = vv;            // canvas
            v[2][n][0] = q[k][2]; v[2][n][1] = 1.0f - vv;          // 3D colour buffer (DS line 0 at the top row: v 1)
            v[3][n][0] = p[10]; v[3][n][1] = p[11];                // brightness
        }
    }
}
// The compose drawn as one quad over the canvas, the per-line parameters read from a small texture the CPU fills
// once per frame (gc_line_texels: 8 RGBA texels per line, already decoded - no per-pixel decoding). (One quad per
// line carrying them as vertex attributes measured slower on the Vita: 1-pixel-high triangles.)
#define GC_TEX_FETCH \
    "  float lv = (floor(fc.y) + 0.5) / 192.0;\n" \
    "  vec4 L0 = texture2D(uLp, vec2(0.5 / 8.0, lv)) * 255.0;\n" \
    "  vec4 L1 = texture2D(uLp, vec2(1.5 / 8.0, lv)) * 255.0;\n" \
    "  vec4 L2 = texture2D(uLp, vec2(2.5 / 8.0, lv)) * 255.0;\n" \
    "  vec4 L3 = texture2D(uLp, vec2(3.5 / 8.0, lv)) * 255.0;\n" \
    "  vec4 L4 = texture2D(uLp, vec2(4.5 / 8.0, lv)) * 255.0;\n" \
    "  vec4 L5 = texture2D(uLp, vec2(5.5 / 8.0, lv)) * 255.0;\n" \
    "  float lfast = L3.w;\n" \
    "  L3.w = L5.x + L5.y * 256.0 - 1024.0;\n" \
    "  L5 = vec4(L5.z + L5.w * 256.0 - 1024.0, lfast, 0.0, 0.0);\n"
static const char kGcqFragmentShader[] =   // GLSL ES 1.00 (host check)
    "precision highp float;\n" GC_GMOD "uniform sampler2D uLp;\n" GC_FRAGMENT_BODY
    "void main() { vec2 fc = gl_FragCoord.xy;\n" GC_TEX_FETCH
    "  gl_FragColor = compose(fc, L0, L1, L2, L3, L4, L5); }\n";
static const char kGcqVertexShader[] =
    "attribute vec2 aPos;\nvoid main() { gl_Position = vec4(aPos, 0.0, 1.0); }\n";
static const char kGcqFragmentShaderCg[] =
    "#define vec2 float2\n#define vec3 float3\n#define vec4 float4\n#define texture2D tex2D\n" GC_GMOD
    "uniform sampler2D uLp;\n" GC_FRAGMENT_BODY
    "float4 main(float4 fc : WPOS) : COLOR {\n" GC_TEX_FETCH
    "  return compose(fc.xy, L0, L1, L2, L3, L4, L5);\n"
    "}\n";
static const char kGcqVertexShaderCg[] =
    "void main(float2 aPos, out float4 oPos : POSITION) { oPos = float4(aPos, 0.0, 1.0); }\n";
// That texture's contents: per line 8 RGBA8 texels = aL0..aL5's values (gc_line_params), with L3.w and L5.x (signed,
// beyond a byte) as value + 1024 in texel 5's bytes (lo, hi).
static inline void gc_line_texels(const unsigned char *par, int parStride, float u3dCols, unsigned char (*o)[8][4]) {
    for (int y = 0; y < 192; y++) {
        float L[24];
        gc_line_params(par + y * parStride, y, u3dCols, L);
        unsigned char *t = &o[y][0][0];
        for (int i = 0; i < 20; i++) t[i] = (unsigned char)L[i];
        t[15] = (unsigned char)L[21];   // (the fast-path flag: GC_TEX_FETCH)
        const int s = (int)L[15] + 1024, h = (L[20] > 30000.0f ? 30000 : (int)L[20]) + 1024;
        t[20] = (unsigned char)(s & 255); t[21] = (unsigned char)(s >> 8);
        t[22] = (unsigned char)(h & 255); t[23] = (unsigned char)(h >> 8);
        for (int i = 24; i < 32; i++) t[i] = 0;
    }
}

// (GLSL ES twins of the present, for the host check)
static const char kBlitVertexShader[] =
    "attribute vec2 aPos, aTex, aT3, aMb;\n"
    "varying vec2 vTex, vT3, vMb;\n"
    "void main() { gl_Position = vec4(aPos, 0.0, 1.0); vTex = aTex; vT3 = aT3; vMb = aMb; }\n";
static const char kBlitFragmentShader[] =
    "precision highp float;\n"
    "varying vec2 vTex, vT3, vMb;\n"
    "uniform sampler2D uTex, u3d;\n"
    "uniform vec4 uSz, uP;\n"
    "void main() {\n"
    "  vec2 tx = vTex * uSz.xy;\n"
    "  vec2 fl = floor(tx);\n"
    "  vec2 d = tx - fl - 0.5;\n"
    "  vec2 r = 0.5 - 0.5 / uSz.zw;\n"
    "  vec2 f2 = (d - clamp(d, -r, r)) * uSz.zw + 0.5;\n"
    "  vec4 cv = texture2D(uTex, (fl + f2) / uSz.xy);\n"
    "  vec4 h = texture2D(u3d, vT3);\n"
    "  vec2 mb = floor(vMb + 0.5);\n"
    "  vec3 c = floor(h.rgb * 63.75 + 0.125);\n"
    "  vec3 cb = mb.x < 1.5 ? c + floor(((63.0 - c) * mb.y) / 16.0) : c - floor((c * mb.y + 15.0) / 16.0);\n"
    "  c = mb.x > 0.5 && mb.x < 2.5 && mb.y > 0.5 ? cb : c;\n"
    "  vec3 h8 = (c * 4.0 + floor(c / 16.0)) / 255.0;\n"
    "  float f = uP.x > 0.5 && h.a > 0.016 ? cv.a : 1.0;\n"
    "  gl_FragColor = vec4(mix(h8, cv.rgb, f), 1.0);\n"
    "}\n";

#endif
