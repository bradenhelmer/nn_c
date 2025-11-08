/*
 * matrix.c - Matrix operations implementation
 */

#include "matrix.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>

Matrix *matrix_create(int rows, int cols) {
    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float *)malloc(sizeof(float) * rows * cols);
    return mat;
}

Matrix *matrix_zeros(int rows, int cols) {
    Matrix *mat = matrix_create(rows, cols);
    for (int m = 0; m < rows; m++) {
        for (int n = 0; n < cols; n++) {
            mat->data[m * cols + n] = 0.f;
        }
    }
    return mat;
}

Matrix *matrix_ones(int rows, int cols) {
    Matrix *mat = matrix_create(rows, cols);
    for (int m = 0; m < rows; m++) {
        for (int n = 0; n < cols; n++) {
            mat->data[m * cols + n] = 1.f;
        }
    }
    return mat;
}

Matrix *matrix_random(int rows, int cols, float min, float max) {
    Matrix *mat = matrix_create(rows, cols);

    srand(time(NULL));
    for (int m = 0; m < rows; m++) {
        for (int n = 0; n < cols; n++) {
            float norm_rand = (float)rand() / (float)RAND_MAX;
            mat->data[m * cols + n] = min + norm_rand * (max - min);
        }
    }
    return mat;
}

Matrix *matrix_identity(int rows, int cols) {
    assert(rows == cols);
    Matrix *mat = matrix_create(rows, cols);
    for (int m = 0; m < rows; m++) {
        for (int n = 0; n < cols; n++) {
            mat->data[m * cols + n] = (m == n) ? 1.f : 0.f;
        }
    }
    return mat;
}

void matrix_free(Matrix *m) {
    free(m->data);
    free(m);
}
