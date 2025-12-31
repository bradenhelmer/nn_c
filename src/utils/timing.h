/*
 * timing.h - Timing utilities.
 */

#ifndef TIMING_H
#define TIMING_H
#include <time.h>

typedef struct {
    struct timespec start;
    double elapsed;
} Timer;

static inline void timer_start(Timer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

static inline void timer_stop(Timer *t) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    t->elapsed = (end.tv_sec - t->start.tv_sec) + (end.tv_nsec - t->start.tv_nsec) * 1e-9;
}

#endif /* ifndef TIMING_H */
