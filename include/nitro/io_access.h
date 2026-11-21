#ifndef NITRO_IO_ACCESS_H
#define NITRO_IO_ACCESS_H


#ifdef __cplusplus
extern "C" {
#endif
unsigned long long HostIo_Read(__UINTPTR_TYPE__ addr, unsigned int size);
unsigned long long HostIo_Write(__UINTPTR_TYPE__ addr, unsigned int size, unsigned long long value);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
template <class T> struct IoUnqual_ { typedef T type; };
template <class T> struct IoUnqual_<volatile T> { typedef T type; };
#define IO_T_(p) IoUnqual_<__typeof__(*(p))>::type
#else
#define IO_T_(p) __typeof__(((void)0, *(p)))
#endif
#define IO_ADDR_(p) ((__UINTPTR_TYPE__)(p))
#define IO_R(p) ((IO_T_(p))HostIo_Read(IO_ADDR_(p), sizeof(*(p))))
#if defined(HOST)
// GX command/FIFO stores (0x04000400-0x040005ff, 32-bit, constant address - the immediate G3_* calls) are queued
// in a small buffer here instead of a call per word; the host processes the queue in one loop before any other
// I/O access, DMA, vblank or GX read (HostGx_StageFlush), so the order of everything the game does is unchanged.
#ifdef __cplusplus
extern "C" {
#endif
extern unsigned int gHostGxStage[];
extern unsigned int gHostGxStageN;
void HostGx_StageFlush(void);
#ifdef __cplusplus
}
#endif
enum { HOST_GX_STAGE_MAX = 512 };
static inline __attribute__((always_inline)) unsigned int HostGx_Stage_(unsigned int a, unsigned int v) {
    unsigned int n = gHostGxStageN;
    gHostGxStage[n] = a;
    gHostGxStage[n + 1] = v;
    gHostGxStageN = n + 2;
    if (n + 2 >= HOST_GX_STAGE_MAX) HostGx_StageFlush();
    return v;
}
#define IO_GX_STAGE_(p) (__builtin_constant_p(IO_ADDR_(p)) && sizeof(*(p)) == 4 && IO_ADDR_(p) - 0x04000400u < 0x200u)
#define IO_W(p, v) (IO_GX_STAGE_(p) ? (IO_T_(p))HostGx_Stage_((unsigned int)IO_ADDR_(p), (unsigned int)(IO_T_(p))(v)) \
                                    : (IO_T_(p))HostIo_Write(IO_ADDR_(p), sizeof(*(p)), (unsigned long long)(IO_T_(p))(v)))
#else
#define IO_W(p, v) ((IO_T_(p))HostIo_Write(IO_ADDR_(p), sizeof(*(p)), (unsigned long long)(IO_T_(p))(v)))
#endif
#define IO_RMW_(p, op, v) IO_W(p, IO_R(p) op (v))
#define IO_OR(p, v) IO_RMW_(p, |, v)
#define IO_AND(p, v) IO_RMW_(p, &, v)
#define IO_XOR(p, v) IO_RMW_(p, ^, v)
#define IO_ADD(p, v) IO_RMW_(p, +, v)
#define IO_SUB(p, v) IO_RMW_(p, -, v)


#endif
