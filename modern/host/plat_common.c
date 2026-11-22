#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <stdio.h>
#include <stdlib.h>

#include "plat.h"

// The whole file in one allocation, read in large blocks straight into it (stdio would go through its small
// buffer: on the Vita that was thousands of 1 KB card reads and seconds of black screen for the 32 MB game).
void *host_plat_read_file(const char *path, size_t *size) {
#ifndef O_BINARY
#define O_BINARY 0
#endif
    int fd = open(path, O_RDONLY | O_BINARY);   // (O_BINARY: no text-mode translation on Windows)
    if (fd < 0) return 0;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) { close(fd); return 0; }
    size_t len = (size_t)st.st_size;
    unsigned char *buf = malloc(len ? len : 1);
    if (!buf) { close(fd); return 0; }
    size_t got = 0;
    while (got < len) {
        size_t want = len - got > (4u << 20) ? (4u << 20) : len - got;
        ssize_t r = read(fd, buf + got, want);
        if (r <= 0) break;
        got += (size_t)r;
    }
    close(fd);
    if (got != len) { free(buf); return 0; }
    *size = len;
    return buf;
}
