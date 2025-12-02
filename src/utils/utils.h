/*
 * utils.h - Utility functions for the neural network library
 */

#ifndef UTILS_H
#define UTILS_H

#define EPSILON 1e-6

// Equality
int float_equals(float a, float b);

// Random
int rand_range(int min, int max);
float rand_rangef(float min, float max);

#endif /* UTILS_H */
