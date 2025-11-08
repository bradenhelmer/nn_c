/*
 * vector.c - Vector operations implementation
 */
#include "vector.h"
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

Vector *vector_create(int size) {
    Vector *v = (Vector *)malloc(sizeof(Vector));
    v->size = size;
    v->data = (float *)malloc(sizeof(float) * size);
    return v;
}

Vector *vector_zeros(int size) {
    Vector *v = vector_create(size);
    for (int i = 0; i < size; i++) {
        v->data[i] = 0.f;
    }
    return v;
}

Vector *vector_ones(int size) {
    Vector *v = vector_create(size);
    for (int i = 0; i < size; i++) {
        v->data[i] = 1.f;
    }
    return v;
}

Vector *vector_random(int size, float min, float max) {
    Vector *v = vector_create(size);

    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        float norm_rand = (float)rand() / (float)RAND_MAX;
        v->data[i] = min + norm_rand * (max - min);
    }
    return v;
}

void vector_free(Vector *v) {
    free(v->data);
    free(v);
}

void vector_add(Vector *result, const Vector *a, const Vector *b) {
    assert(a->size == b->size);
    assert(a->size == result->size);
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
}

void vector_subtract(Vector *result, const Vector *a, const Vector *b) {
    assert(a->size == b->size);
    assert(a->size == result->size);
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
}

void vector_scale(Vector *result, const Vector *v, float scalar) {
    assert(v->size == result->size);
    for (int i = 0; i < v->size; i++) {
        result->data[i] = v->data[i] * scalar;
    }
}

float vector_dot(const Vector *a, const Vector *b) {
    assert(a->size == b->size);
    float acc = 0.f;
    for (int i = 0; i < a->size; i++) {
        acc += a->data[i] * b->data[i];
    }
    return acc;
}

void vector_multiply(Vector *result, const Vector *a, const Vector *b) {
    assert(a->size == b->size);
    assert(a->size == result->size);
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
}

void vector_divide(Vector *result, const Vector *a, const Vector *b) {
    assert(a->size == b->size);
    assert(a->size == result->size);
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] / b->data[i];
    }
}

void vector_copy(Vector *dest, Vector *src) {
    assert(dest->size == src->size);
    memcpy(dest->data, src->data, sizeof(float) * src->size);
}

float vector_sum(const Vector *v) {
    float sum = 0.f;
    for (int i = 0; i < v->size; i++) {
        sum += v->data[i];
    }
    return sum;
}

float vector_min(const Vector *v) {
    float curr;
    float min = FLT_MAX;
    for (int i = 0; i < v->size; i++) {
        curr = v->data[i];
        if (curr < min) {
            min = curr;
        }
    }
    return min;
}

float vector_max(const Vector *v) {
    float curr;
    float max = FLT_MIN;
    for (int i = 0; i < v->size; i++) {
        curr = v->data[i];
        if (curr > max) {
            max = curr;
        }
    }
    return max;
}

void vector_print(const Vector *v) {
    int i;
    printf("[");
    for (i = 0; i < (v->size - 1); i++) {
        printf("%f,", v->data[i]);
    }
    printf("%f]", v->data[i]);
}
