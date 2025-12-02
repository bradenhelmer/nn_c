/*
 * utils.c - Utility functions implementation
 */

#include "utils.h"
#include <math.h>
#include <stdlib.h>

int float_equals(float a, float b) {
    return fabs(a - b) < EPSILON;
}

int rand_range(int min, int max) {
    return min + rand() % (max - min + 1);
}

float rand_rangef(float min, float max) {
    float norm_rand = (float)rand() / (float)RAND_MAX;
    return min + norm_rand * (max - min);
}
