/*
 * vector.c - Vector operations implementation
 */
#include "vector.h"
#include <stdlib.h>
#include <time.h>

Vector *vector_create(int size) {
    Vector *v = (Vector *) malloc(sizeof(Vector));
    v->size = size;
    v->data = (float *) malloc(sizeof(float) * size);
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
        float norm_rand = (float) rand() / (float) RAND_MAX;
        v->data[i] = min + norm_rand * (max - min);
    }
    return v;
}

void vector_free(Vector *v) {
    free(v->data);
    free(v);
}
