
#include <stdio.h>
#include <string.h>

#include "host.h"
#include "host_mem.h"

enum { BANK_TOTAL = 0xa4000, GRANULE = 0x4000 };
static const u32 kSize[9] = {0x20000, 0x20000, 0x20000, 0x20000, 0x10000, 0x4000, 0x4000, 0x8000, 0x4000};
static const u32 kLcdc[9] = {0x06800000, 0x06820000, 0x06840000, 0x06860000, 0x06880000, 0x06890000, 0x06894000,
                             0x06898000, 0x068a0000};
static u32 sOffset[9];
#if HOST_MEM_RUNTIME_ALIGN
static u8 (*sBanksBlock)[BANK_TOTAL];
#define sBanks (*sBanksBlock)
#else
static u8 sBanks[BANK_TOTAL] __attribute__((aligned(0x20000)));
#endif
static u8 sCnt[9];

u8 *host_vram_bank(int i) { return sBanks + sOffset[i]; }
u32 host_vram_bank_size(int i) { return kSize[i]; }

// Snapshot the whole VRAM bank block (for the render thread: the worker reads this stable copy while the main
// thread runs the next frame's VRAM writes). dst must hold host_vram_total() bytes; per-bank pointers into the
// copy are at the same offsets as the live banks, so host_vram_bank_offset(i) maps bank i within dst.
u32 host_vram_total(void) { return BANK_TOTAL; }
u32 host_vram_bank_offset(int i) { return sOffset[i]; }
void host_vram_snapshot(u8 *dst) { memcpy(dst, sBanks, BANK_TOTAL); }


static const struct { u32 base, size, align; } kWindows[] = {
    {0x06000000, 0x80000, 0x80000},
    {0x06200000, 0x20000, 0x20000},
    {0x06400000, 0x40000, 0x40000},
    {0x06600000, 0x20000, 0x20000},
    {0x06800000, 0xa4000, 0x20000},
};
#define NWIN (sizeof kWindows / sizeof kWindows[0])
#if HOST_MEM_RUNTIME_ALIGN
static u8 *sWinMem[NWIN];
#else
static u8 sWinBgA[0x80000] __attribute__((aligned(0x80000)));
static u8 sWinBgB[0x20000] __attribute__((aligned(0x20000)));
static u8 sWinObjA[0x40000] __attribute__((aligned(0x40000)));
static u8 sWinObjB[0x20000] __attribute__((aligned(0x20000)));
static u8 sWinLcdc[0xa4000] __attribute__((aligned(0x20000)));
static u8 *const sWinMem[NWIN] = {sWinBgA, sWinBgB, sWinObjA, sWinObjB, sWinLcdc};
#endif

enum { NGRAN = (0x80000 + 0x20000 + 0x40000 + 0x20000 + 0xa4000) / GRANULE };
static struct { u8 *win; s32 bank; u32 off; } sGran[NGRAN];

static u32 bank_address(int i, u8 c) {
    if (!(c & 0x80)) return 0;
    u32 mst = c & 7, ofs = c >> 3 & 3;
    switch (i) {
    case 0: case 1:
        if (mst == 0) return kLcdc[i];
        if (mst == 1) return 0x06000000 + 0x20000 * ofs;
        if (mst == 2) return 0x06400000 + 0x20000 * (ofs & 1);
        return 0;
    case 2: case 3:
        if (mst == 0) return kLcdc[i];
        if (mst == 1) return 0x06000000 + 0x20000 * ofs;
        if (mst == 4) return i == 2 ? 0x06200000 : 0x06600000;
        return 0;
    case 4:
        if (mst == 0) return kLcdc[i];
        if (mst == 1) return 0x06000000;
        if (mst == 2) return 0x06400000;
        return 0;
    case 5: case 6:
        if (mst == 0) return kLcdc[i];
        if (mst == 1) return 0x06000000 + 0x4000 * (ofs & 1) + 0x10000 * (ofs >> 1);
        if (mst == 2) return 0x06400000 + 0x4000 * (ofs & 1) + 0x10000 * (ofs >> 1);
        return 0;
    case 7:
        if (mst == 0) return kLcdc[i];
        if ((mst & 3) == 1) return 0x06200000;
        return 0;
    case 8:
        if (mst == 0) return kLcdc[i];
        if ((mst & 3) == 1) return 0x06208000;
        if ((mst & 3) == 2) return 0x06600000;
        return 0;
    }
    return 0;
}

// host_vram_flush() + host_vram_snapshot() in one pass: mapped granules are copied from their windows (the live
// data), the rest from bank storage. Bank storage itself is left as is (readers of it flush first).
//
// dst is expected to hold the previous snapshot: each 1 KB block is compared and copied only when it changed, and
// host_vram_snapshot_dirty() reports which blocks changed (the texture setup then copies only those). The first
// snapshot into a new dst copies and flags everything.
enum { SNAP_BLK = 1024 };
static u8 sSnapDirty[BANK_TOTAL / SNAP_BLK];
static const u8 *sSnapDst;
static int snap_blk_differs(const u8 *a, const u8 *b) {
    const u32 *x = (const u32 *)a, *y = (const u32 *)b;
    u32 acc = 0;
    for (int i = 0; i < SNAP_BLK / 4; i++) acc |= x[i] ^ y[i];
    return acc != 0;
}
static void snap_granule(u8 *dst, const u8 *src, u32 o, int full) {
    for (u32 k = 0; k < GRANULE; k += SNAP_BLK) {
        int d = full || snap_blk_differs(dst + o + k, src + k);
        if (d) memcpy(dst + o + k, src + k, SNAP_BLK);
        sSnapDirty[(o + k) / SNAP_BLK] = (u8)d;
    }
}
void host_vram_snapshot_live(u8 *dst) {
    static u8 covered[BANK_TOTAL / GRANULE];
    int full = dst != sSnapDst;
    sSnapDst = dst;
    memset(covered, 0, sizeof covered);
    for (int g = 0; g < NGRAN; g++)
        if (sGran[g].bank >= 0) {
            u32 o = sOffset[sGran[g].bank] + sGran[g].off;
            snap_granule(dst, sGran[g].win, o, full);
            covered[o / GRANULE] = 1;
        }
    for (u32 o = 0; o < BANK_TOTAL; o += GRANULE)
        if (!covered[o / GRANULE]) snap_granule(dst, sBanks + o, o, full);
}
// Changed 1 KB blocks of the last host_vram_snapshot_live(), indexed by bank-block offset / 1024.
const u8 *host_vram_snapshot_dirty(void) { return sSnapDirty; }

static void remap_full(void);
static int granule(u32 a) {
    int g = 0;
    for (unsigned w = 0; w < NWIN; w++) {
        if (a >= kWindows[w].base && a - kWindows[w].base < kWindows[w].size) return g + (int)((a - kWindows[w].base) / GRANULE);
        g += (int)(kWindows[w].size / GRANULE);
    }
    return -1;
}

void host_vram_flush(void) {
    for (int g = 0; g < NGRAN; g++)
        if (sGran[g].bank >= 0) memcpy(sBanks + sOffset[sGran[g].bank] + sGran[g].off, sGran[g].win, GRANULE);
}

// Bank control changed: rebuild the granule map and move data only for granules whose mapping changed (save the
// window to its old bank, then load it from the new bank or clear it). Granules that keep their mapping keep their
// live window data; bank storage for them is brought up to date by host_vram_flush(), which every reader of bank
// storage calls first. The game remaps banks several times per frame to upload textures, and copying/clearing all
// 1.6 MB of windows each time was a quarter of the game thread.
static void remap(void) {
    static int inited;
    if (!inited) { inited = 1; remap_full(); return; }
    static struct { s32 bank; u32 off; } old[NGRAN];
    for (int g = 0; g < NGRAN; g++) { old[g].bank = sGran[g].bank; old[g].off = sGran[g].off; sGran[g].bank = -1; }
    for (int i = 0; i < 9; i++) {
        u32 a = bank_address(i, sCnt[i]);
        if (!a) continue;
        for (u32 off = 0; off < kSize[i]; off += GRANULE) {
            int g = granule(a + off);
            if (g < 0) { fprintf(stderr, "vram: bank %c at %08x outside the windows\n", 'A' + i, a + off); continue; }
            sGran[g].bank = i;
            sGran[g].off = off;
        }
    }
#define CHANGED(g) (old[g].bank != sGran[g].bank || (sGran[g].bank >= 0 && old[g].off != sGran[g].off))
    for (int g = 0; g < NGRAN; g++)   // all saves before any load: a bank can move to other granules
        if (CHANGED(g) && old[g].bank >= 0) memcpy(sBanks + sOffset[old[g].bank] + old[g].off, sGran[g].win, GRANULE);
    for (int g = 0; g < NGRAN; g++) {
        if (!CHANGED(g)) continue;
        if (sGran[g].bank >= 0) memcpy(sGran[g].win, sBanks + sOffset[sGran[g].bank] + sGran[g].off, GRANULE);
        else memset(sGran[g].win, 0, GRANULE);
    }
#undef CHANGED
}

static void remap_full(void) {
    host_vram_flush();
    for (int g = 0; g < NGRAN; g++) sGran[g].bank = -1;
    for (int i = 0; i < 9; i++) {
        u32 a = bank_address(i, sCnt[i]);
        if (!a) continue;
        for (u32 off = 0; off < kSize[i]; off += GRANULE) {
            int g = granule(a + off);
            if (g < 0) { fprintf(stderr, "vram: bank %c at %08x outside the windows\n", 'A' + i, a + off); continue; }
            sGran[g].bank = i;
            sGran[g].off = off;
        }
    }
    for (int g = 0; g < NGRAN; g++) {
        if (sGran[g].bank >= 0) memcpy(sGran[g].win, sBanks + sOffset[sGran[g].bank] + sGran[g].off, GRANULE);
        else memset(sGran[g].win, 0, GRANULE);
    }
}

void host_vram_control(u32 addr, u8 value) {
    int i = addr == 0x04000248 ? 7 : addr == 0x04000249 ? 8 : (int)(addr - 0x04000240);
    if (i < 0 || i > 8 || addr == 0x04000247) return;
    if (sCnt[i] == value) return;
    sCnt[i] = value;
    HOST_PHASE(PH_VRAM, remap());
}

static u8 sScratch[0x80000];
void *HostVram_CpuPtr(unsigned int a) {
    for (unsigned w = 0; w < NWIN; w++)
        if (a >= kWindows[w].base && a - kWindows[w].base < kWindows[w].size) return sWinMem[w] + (a - kWindows[w].base);
    static int reported;
    if (!reported++) fprintf(stderr, "vram: CPU address %08x outside the VRAM windows (frame %llu)\n", a, (unsigned long long)host_frame());
    return sScratch;
}

void host_vram_init(void) {
#if HOST_MEM_RUNTIME_ALIGN
    sBanksBlock = HostMem_AlignedBlock(sizeof sBanks, 0x20000);
    for (unsigned w = 0; w < NWIN; w++) sWinMem[w] = HostMem_AlignedBlock(kWindows[w].size, kWindows[w].align);
#endif
    u32 total = 0;
    for (int i = 0; i < 9; i++) { sOffset[i] = total; total += kSize[i]; }
    int g = 0;
    for (unsigned w = 0; w < NWIN; w++)
        for (u32 off = 0; off < kWindows[w].size; off += GRANULE, g++) sGran[g].win = sWinMem[w] + off;
    remap();
}
