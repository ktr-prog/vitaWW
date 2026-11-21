#ifndef NITRO_DMA_HOST_H
#define NITRO_DMA_HOST_H

#ifdef __cplusplus
extern "C" {
#endif
void HostDma_Addresses(unsigned int ch, __UINTPTR_TYPE__ src, __UINTPTR_TYPE__ dest);
// Feed a display list straight into the geometry engine (what the GX DMA would do), synchronously.
void HostGx_SendList(const void *src, unsigned int bytes);
#ifdef __cplusplus
}
#endif

#endif
