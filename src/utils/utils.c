/*
 * utils.c - Utility functions implementation
 */

#include "utils.h"
#include <math.h>

int float_equals(float a, float b) {
    return fabs(a - b) < EPSILON;
}
