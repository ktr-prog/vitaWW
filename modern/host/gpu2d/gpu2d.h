
#ifndef GPU2D_H
#define GPU2D_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Gpu2dFrameInput {
    const uint8_t *bank[9];
    const uint8_t *palette;
    const uint8_t *oam;
    const uint32_t *lines3d;
    const uint32_t *lines3dTop;
    int topRows;
    void (*hblank)(int line, void *context);
    void *hblankContext;
    int defer3d;
    int gpu;
} Gpu2dFrameInput;

void gpu2d_reset(void);
int gpu2d_tall_3d_ok(void);
int gpu2d_main_on_bottom(void);
const uint32_t *gpu2d_canvas(int *rows);
void gpu2d_write8(uint32_t addr, uint8_t value);
void gpu2d_write16(uint32_t addr, uint16_t value);
void gpu2d_write32(uint32_t addr, uint32_t value);
uint32_t gpu2d_read32(uint32_t addr);

void gpu2d_render_frame(const Gpu2dFrameInput *in);
void gpu2d_render_finish(void);     // run the deferred engine-0 bands + canvas on the render thread
void gpu2d_set_defer_bands(int on); // when on, gpu2d_render_frame does only the main-thread prepass
const uint32_t *gpu2d_screen(int screen);

const uint32_t *gpu2d_deferred(int screen);
const uint32_t *gpu2d_deferred_lines(int screen);

#define GPU2D_BANK_OFFSET {0x00000u, 0x20000u, 0x40000u, 0x60000u, 0x80000u, 0x90000u, 0x94000u, 0x98000u, 0xa0000u}
enum { GPU2D_VRAM_ZERO = 0xa4000, GPU2D_VRAM_OVERLAP = 0xa8000, GPU2D_LINE_WORDS = 40, GPU2D_MAX_VERSIONS = 193 };
enum { G2L_REGS = 0, G2L_REF = 28, G2L_WIN = 32, G2L_OBJMOSAIC = 33, G2L_PALETTE = 34, G2L_OAM = 35, G2L_POWCNT = 36 };
enum { G2T_BG = 0, G2T_OBJ = 32, G2T_BGEXT = 48, G2T_OBJEXT = 52, G2T_LCDC = 53, G2T_SCREEN_A = 63 };
typedef struct Gpu2dGpuFrame {
    const uint8_t *overlap;
    uint32_t overlapSize;
    uint32_t tables[2][64];
    uint32_t lines[2][192][GPU2D_LINE_WORDS];
    const uint8_t *palettes;
    uint32_t paletteCount;
    const uint8_t *oams;
    uint32_t oamCount;
    uint32_t obj0[2][256];
} Gpu2dGpuFrame;
const Gpu2dGpuFrame *gpu2d_gpu_frame(void);

// GPU compose (the Vita): instead of compositing on the CPU, the bands write each line's layers - the four BG
// lines and the sprite colours as raw DS colours (bit 15 = opaque), the window mask + sprite attributes, and the
// line's compose registers - and the GPU blends them with the 3D image (gpu2d_compose.h has the shader).
enum { GPU2D_GC_PAR = 8 };   // parameter texels (RGBA8) per line
typedef struct Gpu2dGc {
    uint16_t bg[4][192 * 256];
    uint16_t obj[192 * 256];   // sprite colour | 0x8000 where a sprite pixel is (0 elsewhere)
    uint16_t aux[192 * 256];   // low byte: window mask; high byte: sprite priority | kind << 2 | alpha << 4
    uint8_t par[192][GPU2D_GC_PAR * 4];
    uint32_t used;             // bit n: BG n has an opaque pixel on some line; bit 4: sprites
} Gpu2dGc;
void gpu2d_set_gpu_compose(int on);       // compose on the GPU from now on (frames it can handle)
const Gpu2dGc *gpu2d_gc_frame(void);      // the finished frame's layers, or NULL when it was composed on the CPU

void gpu2d_set_diagnostics(int on);
void gpu2d_describe_pixel(int screen, int x, int y, char *buf, size_t size);
void gpu2d_print_features(void);

#ifdef __cplusplus
}
#endif

#endif
