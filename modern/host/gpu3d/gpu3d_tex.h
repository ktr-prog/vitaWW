
#ifndef GPU3D_TEX_H
#define GPU3D_TEX_H

#include <string.h>

#include "gpu3d_internal.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

enum { IMG_SLOT = 0x20000, PAL_SLOT = 0x4000 };

static void g3_tex_or(u8 *dst, const u8 *src, u32 n) {
    u32 i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; i + 16 <= n; i += 16) {
        uint8x16_t a = vld1q_u8(dst + i);
        uint8x16_t b = vld1q_u8(src + i);
        vst1q_u8(dst + i, vorrq_u8(a, b));
    }
#endif
    for (; i < n; i++) dst[i] |= src[i];
}
static const u8 *sImg[4][4];
static s32 sImgSrc[4][4], sPalSrc[8][3];   // bank << 20 | byte offset in the bank, per source
static s32 sImgLast[4] = {-1, -1, -1, -1}, sPalLast[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
static int sNImg[4];
static const u8 *sPal[8][3];
static int sNPal[8];
static const u32 kBankSize[9] = {0x20000, 0x20000, 0x20000, 0x20000, 0x10000, 0x4000, 0x4000, 0x8000, 0x4000};

static u8 sImgFlat[4 * IMG_SLOT] __attribute__((aligned(16)));
static u8 sPalFlat[8 * PAL_SLOT] __attribute__((aligned(16)));
static u8 sImgWasActive[4], sPalWasActive[8];

// Changed 1 KB blocks of sImgFlat/sPalFlat since the previous g3_tex_setup (for the GL texture cache).
enum { G3_TEX_BLK = 1024 };
u8 gG3TexDirtyImg[4 * IMG_SLOT / G3_TEX_BLK], gG3TexDirtyPal[8 * PAL_SLOT / G3_TEX_BLK];
static u8 sTexMerge[IMG_SLOT] __attribute__((aligned(16)));
// Per flat block: which bank (and offset) at which generation it was last brought up to date from. A texture or
// texture-palette bank cannot be written by the CPU while it is mapped as one, so while the same single bank
// feeds the slot at the same generation (host_vram_bank_gens, as of the snapshot) the block is known to be
// current and the compare is skipped. 0 = unknown. ACWW_TEXGEN=0 always compares.
u32 gG3BankGenFrame[9];
int gG3BankGenValid;
static u64 sImgBlkKey[4 * IMG_SLOT / G3_TEX_BLK], sPalBlkKey[8 * PAL_SLOT / G3_TEX_BLK];

// Bring d (n bytes) up to date with src block by block: compare, copy only changed blocks, flag them. Most frames
// only a few blocks of texture VRAM change, so this reads the data once instead of copying all of it every frame.
// Word-wise block compare (vectorizes to NEON); the Vita C library's memcmp is far slower on 640 KB per frame.
static int g3_blk_differs(const u8 *a, const u8 *b) {
    const u32 *x = (const u32 *)a, *y = (const u32 *)b;
    u32 acc = 0;
    // Prefetch the next block (the Vita's RAM latency dominates a plain streaming compare; memcpy prefetches too).
    for (int p = 0; p < G3_TEX_BLK; p += 64) { __builtin_prefetch(a + G3_TEX_BLK + p); __builtin_prefetch(b + G3_TEX_BLK + p); }
    for (int i = 0; i < G3_TEX_BLK / 4; i++) acc |= x[i] ^ y[i];
    return acc != 0;
}
static void g3_tex_sync(u8 *d, const u8 *src, u32 n, u8 *dirty) {
    for (u32 o = 0, b = 0; o < n; o += G3_TEX_BLK, b++)
        if (g3_blk_differs(d + o, src + o)) {
            memcpy(d + o, src + o, G3_TEX_BLK);
            dirty[b] = 1;
        }
}
// One texture or palette slot: zero when no bank maps it, the bank itself, or the OR of several banks.
// srcId/last: which bank and offset feeds the slot now and fed it last time. When a single bank keeps feeding the
// slot and the snapshot reported its changed blocks, only those blocks are copied (no compare pass).
static void g3_tex_slot(u8 *d, const u8 *const *src, const s32 *srcId, int nsrc, u32 n, u8 *wasActive, s32 *last,
                        u8 *dirty) {
    if (!nsrc) {
        if (*wasActive) { memset(d, 0, n); memset(dirty, 1, n / G3_TEX_BLK); }
        *wasActive = 0;
        *last = -1;
        return;
    }
    *wasActive = 1;
    if (nsrc == 1) {
        const u8 *bd = gG3BankDirtyFrame ? gG3BankDirtyFrame[srcId[0] >> 20] : 0;
        if (bd && *last == srcId[0]) {
            const u8 *f = bd + (srcId[0] & 0xfffff) / G3_TEX_BLK;
            for (u32 o = 0, b = 0; o < n; o += G3_TEX_BLK, b++)
                if (f[b]) { memcpy(d + o, src[0] + o, G3_TEX_BLK); dirty[b] = 1; }
        } else
            g3_tex_sync(d, src[0], n, dirty);
        *last = srcId[0];
        return;
    }
    *last = -1;
    memcpy(sTexMerge, src[0], n);
    for (int i = 1; i < nsrc; i++) g3_tex_or(sTexMerge, src[i], n);
    g3_tex_sync(d, sTexMerge, n, dirty);
}

static void g3_tex_map(void) {
    memset(sNImg, 0, sizeof sNImg);
    memset(sNPal, 0, sizeof sNPal);
    if (gG3Banks)
        for (int b = 0; b < 7; b++) {
            u8 c = gG3VramCntFrame[b];
            if (!(c & 0x80) || (c & 7) != 3 || !gG3Banks[b]) continue;
            int ofs = c >> 3 & 3;
            if (b < 4) { sImgSrc[ofs][sNImg[ofs]] = b << 20; sImg[ofs][sNImg[ofs]++] = gG3Banks[b]; }
            else if (b == 4)
                for (int s = 0; s < 4; s++) { sPalSrc[s][sNPal[s]] = b << 20 | s * PAL_SLOT; sPal[s][sNPal[s]++] = gG3Banks[b] + s * PAL_SLOT; }
            else {
                int s = (ofs & 1) + (ofs >> 1) * 4;
                sPalSrc[s][sNPal[s]] = b << 20; sPal[s][sNPal[s]++] = gG3Banks[b];
            }
        }
}
static void g3_tex_setup(void) {
    g3_tex_map();
    memset(sImgBlkKey, 0, sizeof sImgBlkKey);   // the full setup rewrites the flat images itself
    memset(sPalBlkKey, 0, sizeof sPalBlkKey);
    memset(gG3TexDirtyImg, 0, sizeof gG3TexDirtyImg);
    memset(gG3TexDirtyPal, 0, sizeof gG3TexDirtyPal);
    for (int s = 0; s < 4; s++)
        g3_tex_slot(sImgFlat + s * IMG_SLOT, sImg[s], sImgSrc[s], sNImg[s], IMG_SLOT, &sImgWasActive[s], &sImgLast[s],
                    gG3TexDirtyImg + s * (IMG_SLOT / G3_TEX_BLK));
    for (int s = 0; s < 8; s++)
        g3_tex_slot(sPalFlat + s * PAL_SLOT, sPal[s], sPalSrc[s], sNPal[s], PAL_SLOT, &sPalWasActive[s], &sPalLast[s],
                    gG3TexDirtyPal + s * (PAL_SLOT / G3_TEX_BLK));
    (void)kBankSize;
}

// Lazy variant for the GL renderer: map the slots now, and bring only the byte ranges of the textures actually
// drawn this frame up to date (g3_sync_img/g3_sync_pal, each 1 KB block at most once per frame), instead of
// comparing all 640 KB of texture VRAM every frame. Change flags are set for synced blocks that changed; a cached
// texture not drawn this frame is checked when it is next drawn.
static u8 sSyncedImg[4 * IMG_SLOT / G3_TEX_BLK], sSyncedPal[8 * PAL_SLOT / G3_TEX_BLK];
static inline u64 g3_blk_key(const s32 *src, int nsrc) {
    if (!gG3BankGenValid) return 0;
    if (!nsrc) return ~0ull;   // no bank in the slot: zeros, which never change
    if (nsrc != 1) return 0;
    return (u64)((u32)src[0] + 1) << 32 | gG3BankGenFrame[src[0] >> 20];
}
static const u8 kZeroBlk[G3_TEX_BLK] __attribute__((aligned(16)));
static void g3_tex_setup_lazy(void) {
    g3_tex_map();
    memset(gG3TexDirtyImg, 0, sizeof gG3TexDirtyImg);
    memset(gG3TexDirtyPal, 0, sizeof gG3TexDirtyPal);
    memset(sSyncedImg, 0, sizeof sSyncedImg);
    memset(sSyncedPal, 0, sizeof sSyncedPal);
}
static void g3_sync_block(u8 *flat, const u8 *const *src, int nsrc, u32 off, u8 *dirty) {
    static u8 tmp[G3_TEX_BLK] __attribute__((aligned(16)));
    const u8 *b;
    if (!nsrc) b = kZeroBlk;
    else if (nsrc == 1) b = src[0] + off;
    else {
        memcpy(tmp, src[0] + off, G3_TEX_BLK);
        for (int i = 1; i < nsrc; i++) g3_tex_or(tmp, src[i] + off, G3_TEX_BLK);
        b = tmp;
    }
    if (g3_blk_differs(flat, b)) { memcpy(flat, b, G3_TEX_BLK); *dirty = 1; }
}
static void g3_sync_img(u32 a, u32 n) {
    enum { NB = 4 * IMG_SLOT / G3_TEX_BLK, PER = IMG_SLOT / G3_TEX_BLK };
    if (!n) return;
    a &= 4 * IMG_SLOT - 1;
    u32 first = a / G3_TEX_BLK, cnt = (a % G3_TEX_BLK + n + G3_TEX_BLK - 1) / G3_TEX_BLK;
    if (cnt > NB) cnt = NB;
    for (u32 i = 0; i < cnt; i++) {
        u32 b = (first + i) % NB, s = b / PER;
        if (sSyncedImg[b]) continue;
        sSyncedImg[b] = 1;
        u64 key = g3_blk_key(sImgSrc[s], sNImg[s]);
        if (key && sImgBlkKey[b] == key) continue;
        g3_sync_block(sImgFlat + b * G3_TEX_BLK, sImg[s], sNImg[s], (b % PER) * G3_TEX_BLK, &gG3TexDirtyImg[b]);
        sImgBlkKey[b] = key;
    }
}
static void g3_sync_pal(u32 a, u32 n) {
    enum { NB = 8 * PAL_SLOT / G3_TEX_BLK, PER = PAL_SLOT / G3_TEX_BLK };
    if (!n) return;
    a &= 8 * PAL_SLOT - 1;
    u32 first = a / G3_TEX_BLK, cnt = (a % G3_TEX_BLK + n + G3_TEX_BLK - 1) / G3_TEX_BLK;
    if (cnt > NB) cnt = NB;
    for (u32 i = 0; i < cnt; i++) {
        u32 b = (first + i) % NB, s = b / PER;
        if (sSyncedPal[b]) continue;
        sSyncedPal[b] = 1;
        u64 key = g3_blk_key(sPalSrc[s], sNPal[s]);
        if (key && sPalBlkKey[b] == key) continue;
        g3_sync_block(sPalFlat + b * G3_TEX_BLK, sPal[s], sNPal[s], (b % PER) * G3_TEX_BLK, &gG3TexDirtyPal[b]);
        sPalBlkKey[b] = key;
    }
}
// The palette bytes a texture reads: [*lo, *lo + *n) (wrapping in the 128 KB palette space). A 4x4-compressed
// texture reads, for each 4x4 block, 4 to 8 bytes at the offset its palette index gives - not its whole 64 KB
// window, so a palette change elsewhere in that window (an animated palette of another texture) leaves it as it
// is. (Its index data must be current in sImgFlat.)
static void g3_pal_range(u32 tp, u32 pp, u32 *lo, u32 *n) {
    static const u32 palBytes[8] = {0, 64, 8, 32, 512, 0x10000, 16, 0};
    int fmt = tp >> 26 & 7;
    if (fmt != 5) { *lo = fmt == 2 ? pp << 3 : pp << 4; *n = palBytes[fmt]; return; }
    u32 w = 8u << (tp >> 20 & 7), h = 8u << (tp >> 23 & 7);
    u32 a = ((tp & 0xffff) << 3) & (4 * IMG_SLOT - 1);
    u32 mn = 0x3fff, mx = 0;
    for (u32 blk = 0, nb = w * h / 16; blk < nb; blk++) {   // the blocks' index entries, as the decoder reads them
        u32 addr = a + blk * 4;
        u32 slot1 = 0x20000 + ((addr & 0x1ffff) >> 1) + ((addr >> 17) == 2 ? 0x10000 : 0);
        u32 i = (u32)(sImgFlat[slot1 & 0x7ffff] | sImgFlat[(slot1 + 1) & 0x7ffff] << 8) & 0x3fff;
        if (i < mn) mn = i;
        if (i > mx) mx = i;
    }
    *lo = ((pp & 0x1fff) << 4) + mn * 4;
    *n = (mx - mn) * 4 + 8;
}
// VRAM byte ranges one texture reads: texels, the 4x4-compressed palette indices, and its palette.
static void g3_tex_sync_for(u32 tp, u32 pp) {
    static const u8 bpp[8] = {0, 8, 2, 4, 8, 2, 8, 16};
    static const u32 palBytes[8] = {0, 64, 8, 32, 512, 0x10000, 16, 0};
    int fmt = tp >> 26 & 7;
    if (!fmt) return;
    u32 w = 8u << (tp >> 20 & 7), h = 8u << (tp >> 23 & 7);
    u32 a = ((tp & 0xffff) << 3) & (4 * IMG_SLOT - 1);
    g3_sync_img(a, w * h * bpp[fmt] / 8);
    if (fmt == 5) g3_sync_img(0x20000 + (a & 0x1ffff) / 2 + (a >= 0x40000 ? 0x10000 : 0), w * h / 8);
    u32 plo, pn;
    g3_pal_range(tp, pp, &plo, &pn);   // (after the index data is synced)
    g3_sync_pal(plo, pn);
    (void)palBytes;
}

static inline u8 img8(u32 a) { return sImgFlat[a & 0x7ffff]; }
static inline u16 img16(u32 a) { return (u16)(img8(a) | img8(a + 1) << 8); }
static inline u32 img32(u32 a) { return (u32)img16(a) | (u32)img16(a + 2) << 16; }
static inline u16 pal16(u32 a) {
    a &= 0x1ffff;
    return (u16)(sPalFlat[a] | sPalFlat[(a + 1) & 0x1ffff] << 8);
}

static inline int wrap(int c, int size, int repeat, int flip) {
    if (!repeat) return c < 0 ? 0 : c >= size ? size - 1 : c;
    if (flip && (c & size)) return size - 1 - (c & (size - 1));
    return c & (size - 1);
}

static inline u32 g3_texel(u32 texparam, u32 texpal, int s, int t) {
    int fmt = texparam >> 26 & 7;
    int w = 8 << (texparam >> 20 & 7), h = 8 << (texparam >> 23 & 7);
    s = wrap(s, w, texparam >> 16 & 1, texparam >> 18 & 1);
    t = wrap(t, h, texparam >> 17 & 1, texparam >> 19 & 1);
    u32 base = (texparam & 0xffff) << 3;
    u32 palbase = (texpal & 0x1fff) << 4;
    int zero = texparam >> 29 & 1;
    u32 i = (u32)(t * w + s);
    switch (fmt) {
    case 1: {
        u8 v = img8(base + i);
        u32 a = v >> 5;
        return (pal16(palbase + (v & 31) * 2) & 0x7fffu) | ((a << 2) + (a >> 1)) << 16;
    }
    case 2: {
        u32 c = img8(base + (i >> 2)) >> ((i & 3) * 2) & 3;
        if (zero && !c) return 0;
        return (pal16(((texpal & 0x1fff) << 3) + c * 2) & 0x7fffu) | 31u << 16;
    }
    case 3: {
        u32 c = img8(base + (i >> 1)) >> ((i & 1) * 4) & 15;
        if (zero && !c) return 0;
        return (pal16(palbase + c * 2) & 0x7fffu) | 31u << 16;
    }
    case 4: {
        u32 c = img8(base + i);
        if (zero && !c) return 0;
        return (pal16(palbase + c * 2) & 0x7fffu) | 31u << 16;
    }
    case 5: {
        u32 block = (u32)((t >> 2) * (w >> 2) + (s >> 2));
        u32 addr = base + block * 4;
        u32 bits = img32(addr);
        u32 c = bits >> (((t & 3) * 4 + (s & 3)) * 2) & 3;
        u32 slot1 = 0x20000 + ((addr & 0x1ffff) >> 1) + ((addr >> 17) == 2 ? 0x10000 : 0);
        u16 info = img16(slot1);
        u32 pb = palbase + (u32)(info & 0x3fff) * 4;
        int mode = info >> 14;
        u16 c0 = pal16(pb), c1 = pal16(pb + 2);
        if (c == 3 && mode < 2) return 0;
        if (c < 2 || mode == 2 || (mode == 0 && c == 2)) return (pal16(pb + c * 2) & 0x7fffu) | 31u << 16;
        u32 r0 = c0 & 31, g0 = c0 >> 5 & 31, b0 = c0 >> 10 & 31, r1 = c1 & 31, g1 = c1 >> 5 & 31, b1 = c1 >> 10 & 31;
        u32 r, g, b;
        if (mode == 1) {
            r = (r0 + r1) / 2, g = (g0 + g1) / 2, b = (b0 + b1) / 2;
        } else if (c == 2) {
            r = (r0 * 5 + r1 * 3) / 8, g = (g0 * 5 + g1 * 3) / 8, b = (b0 * 5 + b1 * 3) / 8;
        } else {
            r = (r0 * 3 + r1 * 5) / 8, g = (g0 * 3 + g1 * 5) / 8, b = (b0 * 3 + b1 * 5) / 8;
        }
        return r | g << 5 | b << 10 | 31u << 16;
    }
    case 6: {
        u8 v = img8(base + i);
        return (pal16(palbase + (v & 7) * 2) & 0x7fffu) | (u32)(v >> 3) << 16;
    }
    case 7: {
        u16 v = img16(base + i * 2);
        return (v & 0x8000) ? (v & 0x7fffu) | 31u << 16 : 0;
    }
    default: return 31u << 16;
    }
}

#endif
