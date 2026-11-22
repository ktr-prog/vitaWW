// PS Vita entry point. The VPK's eboot.bin starts here with no arguments, so this finds the player's game
// under ux0:/data/vitaWW, sets up a log file next to it, and hands the path to host_main (modern/host/main.c).
// Everything after that - the window, input, audio, the frame loop - is the shared SDL frontend
// (modern/host/frontend_sdl.c), the same code the desktop and iOS builds use.
#define SDL_MAIN_HANDLED
#include <SDL.h>

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/reent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <psp2/kernel/processmgr.h>
#include <psp2/power.h>

#include "vita.h"

// CI writes build_id.h with the commit short SHA so every log says exactly which build produced it
#if defined(__has_include)
#  if __has_include("build_id.h")
#    include "build_id.h"
#  endif
#endif
#ifndef VITA_BUILD
#define VITA_BUILD "dev"
#endif

#define DATA_DIR "ux0:/data/vitaWW"
#define README   DATA_DIR "/Put your Animal Crossing Wild World .nds file here.txt"

// Enlarge the main thread's kernel-registered stack. The game, the SDL frontend and the DS-SDK threads all run
// as coroutines on this one OS thread, and host_plat_stack_alloc (modern/host/plat_posix.c) carves their stacks
// from the low end of this region so that syscalls made from any coroutine pass the Vita's stack-bounds gate.
// Holds the game coroutine (8 MB) + frontend coroutine (8 MB) + several DS threads (2 MB each) + headroom; any
// further DS-thread stacks fall back to the heap. The VPK asks for the extended-memory budget (ATTRIBUTE2=12,
// tools/vita_package.py) so this plus the 128 MB newlib heap fits.
unsigned int sceUserMainThreadStackSize = 32 * 1024 * 1024;

void vita_fatal_message(const char *text) {
    // the message box needs SDL's video subsystem; bring it up if the game never got that far
    SDL_InitSubSystem(SDL_INIT_VIDEO);
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "vitaWW", text, NULL);
}

static int ends_with_nds(const char *name) {
    size_t n = strlen(name);
    return n > 4 && strcasecmp(name + n - 4, ".nds") == 0;
}

// the first .nds in the folder, by name (the iOS glue likewise expects a single game file there)
static int find_rom(char *out, size_t cap) {
    DIR *d = opendir(DATA_DIR);
    if (!d) return 0;
    struct dirent *e;
    char best[256] = "";
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.' || !ends_with_nds(e->d_name)) continue;
        if (best[0] == 0 || strcmp(e->d_name, best) < 0)
            snprintf(best, sizeof best, "%s", e->d_name);
    }
    closedir(d);
    if (best[0] == 0) return 0;
    snprintf(out, cap, DATA_DIR "/%s", best);
    return 1;
}

// The log must never stall a frame: writing to ux0 takes milliseconds, and the game and render threads print
// a report every 60 frames. stderr stays a newlib FILE opened on the log file (freopen keeps the one shared
// FILE object every thread prints to), but its low-level write hook is replaced: lines are copied into a
// memory ring, and a low-priority thread writes the ring to the file a few times a second. If the ring is
// full (only possible if the card stalls for seconds), lines are dropped and the count is logged.
#define LOG_RING (1 << 17)
static char sLogRing[LOG_RING];
static unsigned sLogHead, sLogTail, sLogDropped;   // head: write position, tail: drained up to (both wrap)
static SDL_SpinLock sLogLock, sLogDrainLock;   // drain lock: the writer thread vs the exit flush
static int sLogFd = -1;

static void ring_put(const char *buf, unsigned n) {   // (sLogLock held)
    unsigned at = sLogHead & (LOG_RING - 1), first = n < LOG_RING - at ? n : LOG_RING - at;
    memcpy(sLogRing + at, buf, first);
    memcpy(sLogRing, buf + first, n - first);
    sLogHead += n;
}
static int sLogMidLine;   // the last write did not end its line
// Every line starts with the time since the process started ("[  12.345] "), so the log shows where time goes
// (the boot included).
static _READ_WRITE_RETURN_TYPE log_ring_write(struct _reent *r, void *cookie, const char *buf,
                                              _READ_WRITE_BUFSIZE_TYPE n) {
    (void)r; (void)cookie;
    char stamp[24];
    unsigned long long us = sceKernelGetProcessTimeWide();
    int sl = snprintf(stamp, sizeof stamp, "[%4u.%03u] ", (unsigned)(us / 1000000), (unsigned)(us / 1000 % 1000));
    SDL_AtomicLock(&sLogLock);
    unsigned lines = 0;
    for (_READ_WRITE_BUFSIZE_TYPE i = 0; i < n; i++) lines += buf[i] == '\n';
    unsigned need = (unsigned)n + (lines + 1) * (unsigned)sl;
    if (need > LOG_RING - (sLogHead - sLogTail)) sLogDropped++;
    else {
        const char *p = buf, *end = buf + n;
        while (p < end) {
            if (!sLogMidLine) ring_put(stamp, (unsigned)sl);
            const char *nl = memchr(p, '\n', (size_t)(end - p));
            const char *e = nl ? nl + 1 : end;
            ring_put(p, (unsigned)(e - p));
            sLogMidLine = !nl;
            p = e;
        }
    }
    SDL_AtomicUnlock(&sLogLock);
    return n;
}

// The writer reads [tail, head) straight out of the ring; producers never touch that range (they only fill the
// room before tail), so the lock is held just to read head and to advance tail - a writer preempted mid-write
// never blocks a game thread for longer than one line copy.
static void log_drain(void) {
    SDL_AtomicLock(&sLogDrainLock);
    SDL_AtomicLock(&sLogLock);
    unsigned h = sLogHead, t = sLogTail, dropped = sLogDropped;
    sLogDropped = 0;
    SDL_AtomicUnlock(&sLogLock);
    unsigned n = h - t, at = t & (LOG_RING - 1), first = n < LOG_RING - at ? n : LOG_RING - at;
    if (first) write(sLogFd, sLogRing + at, first);
    if (n > first) write(sLogFd, sLogRing, n - first);
    SDL_AtomicLock(&sLogLock);
    sLogTail = h;
    SDL_AtomicUnlock(&sLogLock);
    if (dropped) {
        char msg[64];
        int m = snprintf(msg, sizeof msg, "vita: log ring full, %u lines dropped\n", dropped);
        write(sLogFd, msg, (size_t)m);
    }
    SDL_AtomicUnlock(&sLogDrainLock);
}
static void log_final(void) { fflush(stderr); log_drain(); }

static int log_writer(void *unused) {
    (void)unused;
    SDL_SetThreadPriority(SDL_THREAD_PRIORITY_LOW);
    for (;;) {
        SDL_Delay(200);
        log_drain();
    }
    return 0;
}

static void open_log(void) {
    rename(DATA_DIR "/acww-log.txt", DATA_DIR "/acww-log-previous.txt");
    // (newlib on the Vita has no dup2: the game logs to stderr, so only stderr is redirected to the file)
    if (!freopen(DATA_DIR "/acww-log.txt", "w", stderr)) return;
    fprintf(stderr, "vita: vitaWW starting (build %s)\n", VITA_BUILD);
    fflush(stderr);
    sLogFd = stderr->_file;
    static char lineBuf[1024];
    setvbuf(stderr, lineBuf, _IOLBF, sizeof lineBuf);
    stderr->_write = log_ring_write;
    atexit(log_final);
    SDL_Thread *t = SDL_CreateThread(log_writer, "log writer", NULL);
    if (t) SDL_DetachThread(t);
}

static void ensure_readme(void) {
    FILE *f = fopen(README, "r");
    if (f) { fclose(f); return; }
    f = fopen(README, "w");
    if (!f) return;
    fputs("Copy your own dump of Animal Crossing: Wild World (USA, Rev 1) into ux0:/data/vitaWW -\n"
          "one file ending in .nds - then start vitaWW again.\n\n"
          "The game saves next to it (same name, ending in .sav). Keep a backup of that file somewhere safe.\n", f);
    fclose(f);
}

// read ux0:/data/vitaWW/acww-options.txt and set each "NAME=value" as an environment variable (host subsystems
// read ACWW_* from the environment); create a template if it is missing. Lets options be changed on the device
// without a rebuild.
static void read_options(void) {
    FILE *f = fopen(DATA_DIR "/acww-options.txt", "r");
    if (!f) {
        f = fopen(DATA_DIR "/acww-options.txt", "w");
        if (f) {
            fputs("# vitaWW options: one NAME=value per line; remove the # to enable, then restart vitaWW.\n"
                  "#ACWW_MUTE=1            # run without sound\n"
                  "#ACWW_OVLOG=1           # log overlay/module loads to acww-log.txt (boot progress)\n"
                  "#ACWW_IOLOG=4000000     # log I/O register writes at/above this address\n", f);
            fclose(f);
        }
        return;
    }
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (*s == '#' || *s == '\n' || *s == '\r' || *s == 0) continue;
        char *eq = strchr(s, '=');
        if (!eq) continue;
        *eq = 0;
        char *name = s, *val = eq + 1;
        char *end = val + strlen(val);
        while (end > val && (end[-1] == '\n' || end[-1] == '\r' || end[-1] == ' ' || end[-1] == '\t')) *--end = 0;
        setenv(name, val, 1);
        fprintf(stderr, "vita: option %s=%s\n", name, val);
    }
    fclose(f);
}

int main(int argc, char **argv) {
    SDL_SetMainReady();
    vita_io_init();  // must precede any file I/O: the game coroutine's file syscalls are marshaled to this worker
    // Highest clocks the system offers apps (the standard modes retail games can request; no plugin): 444 MHz ARM (+33% over the stock 333 for every thread), 222 MHz
    // bus, 222 MHz GPU (stock 111) and 166 MHz GPU crossbar.
    int wasArm = scePowerGetArmClockFrequency(), wasBus = scePowerGetBusClockFrequency();
    scePowerSetArmClockFrequency(444);
    scePowerSetBusClockFrequency(222);
    scePowerSetGpuClockFrequency(222);
    scePowerSetGpuXbarClockFrequency(166);
    mkdir("ux0:/data", 0777);
    mkdir(DATA_DIR, 0777);
    open_log();
    fprintf(stderr, "vita: arm clock %d -> %d MHz, bus %d -> %d MHz\n", wasArm, scePowerGetArmClockFrequency(),
            wasBus, scePowerGetBusClockFrequency());
    ensure_readme();
    chdir(DATA_DIR);
    read_options();
    fprintf(stderr, "vita: clocks now arm %d, bus %d, gpu %d, xbar %d MHz\n", scePowerGetArmClockFrequency(),
            scePowerGetBusClockFrequency(), scePowerGetGpuClockFrequency(), scePowerGetGpuXbarClockFrequency());
    // overlay logging is off by default; enable ACWW_OVLOG=1 in acww-options.txt for boot-progress diagnostics

    static char path[1024];
    if (!find_rom(path, sizeof path)) {
        fprintf(stderr, "vita: no .nds found in %s\n", DATA_DIR);
        vita_fatal_message("No game found.\n\nCopy your Animal Crossing: Wild World (USA) .nds file into\n"
                           "ux0:/data/vitaWW (for example with VitaShell), then start vitaWW again.");
        return 1;
    }
    fprintf(stderr, "vita: the game: %s\n", path);

    static char *args[3];
    args[0] = (argc > 0 && argv && argv[0]) ? argv[0] : "vitaWW";
    args[1] = path;
    args[2] = NULL;
    return host_main(2, args);
}
