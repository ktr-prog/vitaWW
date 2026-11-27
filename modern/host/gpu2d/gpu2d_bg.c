#include <stdio.h>
#include <stdlib.h>

#include "gpu2d_internal.h"
unsigned long long gBgKindN[2][5][2];
uint64_t host_plat_time_ns(void);

enum BgKind { BG_NONE, BG_TEXT, BG_AFFINE, BG_EXTENDED, BG_LARGE };

static const u8 kBgKinds[8][4] = {
    {BG_TEXT, BG_TEXT, BG_TEXT, BG_TEXT},       {BG_TEXT, BG_TEXT, BG_TEXT, BG_AFFINE},
    {BG_TEXT, BG_TEXT, BG_AFFINE, BG_AFFINE},   {BG_TEXT, BG_TEXT, BG_TEXT, BG_EXTENDED},
    {BG_TEXT, BG_TEXT, BG_AFFINE, BG_EXTENDED}, {BG_TEXT, BG_TEXT, BG_EXTENDED, BG_EXTENDED},
    {BG_NONE, BG_NONE, BG_LARGE, BG_NONE},      {BG_NONE, BG_NONE, BG_NONE, BG_NONE},
};

static const u8 kZeroPalette[0x2000];

static u32 char_base(const Engine *e, u16 cnt) {
    u32 b = (cnt >> 2 & 15) * 0x4000u;
    if (e->num == 0) b += (reg32(e, REG_DISPCNT) >> 24 & 7) * 0x10000u;
    return b;
}
static u32 map_base(const Engine *e, u16 cnt) {
    u32 b = (cnt >> 8 & 31) * 0x800u;
    if (e->num == 0) b += (reg32(e, REG_DISPCNT) >> 27 & 7) * 0x10000u;
    return b;
}

static const u8 *ext_palette(const Engine *e, int n, u16 cnt) {
    if (!(reg32(e, REG_DISPCNT) & 1u << 30)) return NULL;
    int slot = n;
    if (n < 2 && (cnt & 0x2000)) slot += 2;
    return e->bgExtPal[slot] ? e->bgExtPal[slot] : kZeroPalette;
}

static u16 tile8_pixel(const Engine *e, u32 charBase, u16 entry, u32 px, u32 py, const u8 *ext) {
    if (entry & 0x400) px = 7 - px;
    if (entry & 0x800) py = 7 - py;
    u8 idx = bg8(e, charBase + (entry & 0x3ffu) * 64 + py * 8 + px);
    if (!idx) return 0;
    u16 c = ext ? pal16(ext, (entry >> 12) * 256u + idx) : pal16(e->palette, idx);
    return (u16)((c & 0x7fff) | PIX_OPAQUE);
}

static int text_line(Engine *e, int n, int line, u16 *dst) {
    u16 cnt = reg16(e, REG_BGCNT + 2 * n);
    u32 hofs = reg16(e, REG_BGHOFS + 4 * n) & 0x1ff, vofs = reg16(e, REG_BGVOFS + 4 * n) & 0x1ff;
    u32 cb = char_base(e, cnt), mb = map_base(e, cnt);
    u32 wide = cnt & 0x4000 ? 512 : 256, tall = cnt & 0x8000 ? 512 : 256;
    u32 y = (line + vofs) & (tall - 1);
    u32 rowBase = mb + (y >= 256 ? (wide == 512 ? 2 : 1) * 0x800u : 0) + (y & 255) / 8 * 64;
    int color256 = cnt & 0x80;
    const u8 *ext = color256 ? ext_palette(e, n, cnt) : NULL;
    gpu2d_feature(e, color256 ? (ext ? F_TEXT8_EXTPAL : F_TEXT8) : F_TEXT4);
    // Per tile: one pointer lookup for the whole row (a tile row is 4/8 bytes, 32/64-byte aligned, so it never
    // straddles a VRAM granule) instead of a granule lookup per byte; an all-transparent row is just zeroes.
    const u32 bgm = e->bgMask;
    int any = 0;
    for (int x = 0; x < SCREEN_W;) {
        u32 sx = (x + hofs) & (wide - 1);
        u16 entry = bg16(e, rowBase + (sx >= 256 ? 0x800u : 0) + (sx & 255) / 8 * 2);
        u32 tile = entry & 0x3ffu, py = entry & 0x800 ? 7 - (y & 7) : y & 7;
        int flip = entry & 0x400 ? 7 : 0;
        u32 px = sx & 7;
        int cnt8 = 8 - (int)px;
        if (cnt8 > SCREEN_W - x) cnt8 = SCREEN_W - x;
        u8 row[8];
        const u8 *pal;
        if (color256) {
            u32 a = (cb + tile * 64 + py * 8) & bgm;
            const u8 *t = e->bgGran[a >> GRANULE_SHIFT] + (a & (GRANULE - 1));
            u32 lo = (u32)t[0] | (u32)t[1] << 8 | (u32)t[2] << 16 | (u32)t[3] << 24;
            u32 hi = (u32)t[4] | (u32)t[5] << 8 | (u32)t[6] << 16 | (u32)t[7] << 24;
            if (!(lo | hi)) { for (int i = 0; i < cnt8; i++) dst[x + i] = 0; x += cnt8; continue; }
            for (u32 i = 0; i < 4; i++) { row[i] = (u8)(lo >> 8 * i); row[4 + i] = (u8)(hi >> 8 * i); }
            pal = ext ? ext + (entry >> 12) * 512u : e->palette;
        } else {
            u32 a = (cb + tile * 32 + py * 4) & bgm;
            const u8 *t = e->bgGran[a >> GRANULE_SHIFT] + (a & (GRANULE - 1));
            u32 w = (u32)t[0] | (u32)t[1] << 8 | (u32)t[2] << 16 | (u32)t[3] << 24;
            if (!w) { for (int i = 0; i < cnt8; i++) dst[x + i] = 0; x += cnt8; continue; }
            for (u32 i = 0; i < 8; i++) row[i] = (u8)(w >> 4 * i & 15);
            pal = e->palette + (entry >> 12) * 32u;
        }
        any = 1;
        for (int i = 0; i < cnt8; i++, px++, x++) {
            u8 idx = row[px ^ flip];
            dst[x] = idx ? (u16)((pal16(pal, idx) & 0x7fff) | PIX_OPAQUE) : 0;
        }
    }
    return any;
}

static void affine_start(const Engine *e, int n, int line, s32 *x, s32 *y, s32 *pa, s32 *pc) {
    u32 r = REG_BG2PA + (u32)(n - 2) * 0x10;
    *pa = (s16)reg16(e, r);
    s32 pb = (s16)reg16(e, r + 2);
    *pc = (s16)reg16(e, r + 4);
    s32 pd = (s16)reg16(e, r + 6);
    *x = e->refX[n - 2] + pb * (line - e->refLine);
    *y = e->refY[n - 2] + pd * (line - e->refLine);
}

static inline s32 affine_coord(s32 c, s32 w, int wrap) {
    if (wrap) return c & (w - 1);
    return c >= 0 && c < w ? c : -1;
}

static void affine_line(Engine *e, int n, int line, u16 *dst) {
    u16 cnt = reg16(e, REG_BGCNT + 2 * n);
    s32 size = 128 << (cnt >> 14), x0, y0, pa, pc;
    int wrap = cnt & 0x2000;
    u32 cb = char_base(e, cnt), mb = map_base(e, cnt);
    affine_start(e, n, line, &x0, &y0, &pa, &pc);
    gpu2d_feature(e, F_AFFINE);
    if (pc == 0) {
        // Not rotated (the menus, the top screen's zoomed pictures): the whole line reads one source row, so each
        // tile's map byte and its 8-byte pixel row are looked up once while x stays in that tile, not per pixel.
        // Same texel coordinates as below, so the same colours.
        const s32 ty = affine_coord(y0 >> 8, size, wrap);
        if (ty < 0) { memset(dst, 0, SCREEN_W * sizeof *dst); return; }
        const u32 rowMap = mb + (u32)(ty / 8 * (size / 8)), ty7 = (u32)(ty & 7) * 8;
        s32 lastCol = -1;
        const u8 *row = NULL;
        for (int x = 0; x < SCREEN_W; x++) {
            const s32 tx = affine_coord((x0 + pa * x) >> 8, size, wrap);
            if (tx < 0) { dst[x] = 0; continue; }
            if (tx >> 3 != lastCol) {
                lastCol = tx >> 3;
                const u32 a = (cb + bg8(e, rowMap + (u32)lastCol) * 64u + ty7) & e->bgMask;   // 8-aligned: one granule
                row = e->bgGran[a >> GRANULE_SHIFT] + (a & (GRANULE - 1));
            }
            const u8 idx = row[tx & 7];
            dst[x] = idx ? (u16)((pal16(e->palette, idx) & 0x7fff) | PIX_OPAQUE) : 0;
        }
        return;
    }
    for (int x = 0; x < SCREEN_W; x++) {
        s32 tx = affine_coord((x0 + pa * x) >> 8, size, wrap), ty = affine_coord((y0 + pc * x) >> 8, size, wrap);
        if (tx < 0 || ty < 0) { dst[x] = 0; continue; }
        u8 tile = bg8(e, mb + (u32)(ty / 8 * (size / 8) + tx / 8));
        u8 idx = bg8(e, cb + tile * 64u + (u32)(ty & 7) * 8 + (u32)(tx & 7));
        dst[x] = idx ? (u16)((pal16(e->palette, idx) & 0x7fff) | PIX_OPAQUE) : 0;
    }
}

static void bitmap_line(Engine *e, int n, int line, u16 *dst, u32 base, s32 w, s32 h, int direct) {
    u16 cnt = reg16(e, REG_BGCNT + 2 * n);
    s32 x0, y0, pa, pc;
    int wrap = cnt & 0x2000;
    affine_start(e, n, line, &x0, &y0, &pa, &pc);
    for (int x = 0; x < SCREEN_W; x++) {
        s32 tx = affine_coord((x0 + pa * x) >> 8, w, wrap), ty = affine_coord((y0 + pc * x) >> 8, h, wrap);
        if (tx < 0 || ty < 0) { dst[x] = 0; continue; }
        u32 at = (u32)(ty * w + tx);
        if (direct) {
            u16 c = bg16(e, base + at * 2);
            dst[x] = c & 0x8000 ? (u16)((c & 0x7fff) | PIX_OPAQUE) : 0;
        } else {
            u8 idx = bg8(e, base + at);
            dst[x] = idx ? (u16)((pal16(e->palette, idx) & 0x7fff) | PIX_OPAQUE) : 0;
        }
    }
}

static void extended_line(Engine *e, int n, int line, u16 *dst) {
    u16 cnt = reg16(e, REG_BGCNT + 2 * n);
    int sz = cnt >> 14;
    if (cnt & 0x80) {
        static const s32 kW[4] = {128, 256, 512, 512}, kH[4] = {128, 256, 256, 512};
        int direct = cnt & 4;
        gpu2d_feature(e, direct ? F_EXT_DIRECT : F_EXT_BITMAP8);
        bitmap_line(e, n, line, dst, (cnt >> 8 & 31) * 0x4000u, kW[sz], kH[sz], direct);
        return;
    }
    s32 size = 128 << sz, x0, y0, pa, pc;
    int wrap = cnt & 0x2000;
    u32 cb = char_base(e, cnt), mb = map_base(e, cnt);
    const u8 *ext = ext_palette(e, n, cnt);
    gpu2d_feature(e, ext ? F_EXT_TILES_EXTPAL : F_EXT_TILES);
    affine_start(e, n, line, &x0, &y0, &pa, &pc);
    for (int x = 0; x < SCREEN_W; x++) {
        s32 tx = affine_coord((x0 + pa * x) >> 8, size, wrap), ty = affine_coord((y0 + pc * x) >> 8, size, wrap);
        if (tx < 0 || ty < 0) { dst[x] = 0; continue; }
        u16 entry = bg16(e, mb + (u32)(ty / 8 * (size / 8) + tx / 8) * 2);
        dst[x] = tile8_pixel(e, cb, entry, (u32)(tx & 7), (u32)(ty & 7), ext);
    }
}

void gpu2d_render_bg_line(Engine *e, int n, int line) {
    u16 *dst = e->bg[n];
    u32 dispcnt = reg32(e, REG_DISPCNT);
    int mode = dispcnt & 7;
    u16 cnt = reg16(e, REG_BGCNT + 2 * n);
    e->bgAny[n] = 0;
    if (!(dispcnt & 0x100u << n) || (e->num == 1 && mode > 5) || (n == 0 && e->num == 0 && (dispcnt & 8))) {
        memset(dst, 0, sizeof e->bg[n]);
        return;
    }
    e->bgAny[n] = 1;   // (conservative for the non-text kinds)
    int mosaic = cnt & 0x40;
    u16 mos = reg16(e, REG_MOSAIC);
    if (mosaic) {
        gpu2d_feature(e, F_BG_MOSAIC);
        line -= line % ((mos >> 4 & 15) + 1);
    }
    // (the log: engine B's BG lines by kind, and their time - engine B only: it is drawn only for the menus)
    gBgKindN[e->num][kBgKinds[mode][n]][0]++;
    const uint64_t tk0 = e->num ? host_plat_time_ns() : 0;
    switch (kBgKinds[mode][n]) {
    case BG_TEXT: e->bgAny[n] = (u8)text_line(e, n, line, dst); break;
    case BG_AFFINE: affine_line(e, n, line, dst); break;
    case BG_EXTENDED: extended_line(e, n, line, dst); break;
    case BG_LARGE: {
        static const s32 kW[4] = {512, 1024, 512, 512}, kH[4] = {1024, 512, 256, 512};
        gpu2d_feature(e, F_LARGE_BITMAP);
        bitmap_line(e, n, line, dst, 0, kW[cnt >> 14], kH[cnt >> 14], 0);
        break;
    }
    default: memset(dst, 0, sizeof e->bg[n]); e->bgAny[n] = 0; break;
    }
    if (e->num) gBgKindN[1][kBgKinds[mode][n]][1] += host_plat_time_ns() - tk0;
    int mw = (mos & 15) + 1;
    if (mosaic && mw > 1)
        for (int x = 0; x < SCREEN_W; x++) dst[x] = dst[x - x % mw];
}
