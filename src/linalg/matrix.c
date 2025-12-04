/*
 * matrix.c - Matrix operations implementation
 */

#include "matrix.h"
#include "../utils/utils.h"
#include "vector.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

Matrix *matrix_create(int rows, int cols) {
    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float *)calloc(rows * cols, sizeof(float));
    return mat;
}

Matrix *matrix_zeros(int rows, int cols) {
    return matrix_create(rows, cols);
}

Matrix *matrix_ones(int rows, int cols) {
    Matrix *m = matrix_create(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix_set(m, i, j, 1.f);
        }
    }
    return m;
}

Matrix *matrix_random(int rows, int cols, float min, float max) {
    Matrix *m = matrix_create(rows, cols);

    srand(time(NULL));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix_set(m, i, j, rand_rangef(min, max));
        }
    }
    return m;
}

Matrix *matrix_identity(int size) {
    Matrix *m = matrix_create(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix_set(m, i, j, (i == j) ? 1.f : 0.f);
        }
    }
    return m;
}

void matrix_free(Matrix *m) {
    if (m != NULL) {
        free(m->data);
        free(m);
    }
}

void matrix_multiply(Matrix *result, const Matrix *a, const Matrix *b) {
    assert(a->cols == b->rows);
    assert(result->rows == a->rows);
    assert(result->cols == b->cols);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float acc = 0.f;
            for (int k = 0; k < a->cols; k++) {
                acc += matrix_get(a, i, k) * matrix_get(b, k, j);
            }
            matrix_set(result, i, j, acc);
        }
    }
}

void matrix_transpose(Matrix *result, const Matrix *m) {
    assert(m->rows == result->rows);
    assert(m->cols == result->cols);

    // Swap dimensions first so matrix_set works correctly
    result->rows = m->cols;
    result->cols = m->rows;

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            matrix_set(result, j, i, matrix_get(m, i, j));
        }
    }
}

void matrix_add(Matrix *result, const Matrix *a, const Matrix *b) {
    assert(a->rows == b->rows && b->rows == result->rows);
    assert(a->cols == b->cols && b->cols == result->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            matrix_set(result, i, j, matrix_get(a, i, j) + matrix_get(b, i, j));
        }
    }
}

void matrix_subtract(Matrix *result, const Matrix *a, const Matrix *b) {
    assert(a->rows == b->rows && b->rows == result->rows);
    assert(a->cols == b->cols && b->cols == result->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            matrix_set(result, i, j, matrix_get(a, i, j) - matrix_get(b, i, j));
        }
    }
}

void matrix_scale(Matrix *result, const Matrix *m, float scalar) {
    assert(m->rows == result->rows);
    assert(m->cols == result->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            matrix_set(result, i, j, matrix_get(m, i, j) * scalar);
        }
    }
}

void matrix_fill(Matrix *m, float scalar) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            matrix_set(m, i, j, scalar);
        }
    }
}

void matrix_multiply_elementwise(Matrix *result, const Matrix *a, const Matrix *b) {
    assert(a->rows == b->rows && b->rows == result->rows);
    assert(a->cols == b->cols && b->cols == result->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            matrix_set(result, i, j, matrix_get(a, i, j) * matrix_get(b, i, j));
        }
    }
}

void matrix_add_scalar(Matrix *result, const Matrix *m, float scalar) {
    assert(m->rows == result->rows);
    assert(m->cols == result->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            matrix_set(result, i, j, matrix_get(m, i, j) + scalar);
        }
    }
}

void matrix_vector_multiply(Vector *result, const Matrix *m, const Vector *v) {
    assert(m->cols == v->size);
    assert(result->size == m->rows);
    for (int i = 0; i < m->rows; i++) {
        float acc = 0.f;
        for (int j = 0; j < m->cols; j++) {
            acc += matrix_get(m, i, j) * v->data[j];
        }
        result->data[i] = acc;
    }
}

void matrix_transpose_vector_multiply(Vector *result, const Matrix *m, const Vector *v) {
    assert(m->rows == v->size);
    Matrix *m_t = matrix_create(m->rows, m->cols);
    matrix_transpose(m_t, m);
    matrix_vector_multiply(result, m_t, v);
    matrix_free(m_t);
}

// Adds a vector to each row of the matrix
void matrix_add_vector(Matrix *result, const Matrix *m, const Vector *v) {
    assert(result->rows == m->rows);
    assert(result->cols == m->cols && m->cols == v->size);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            matrix_set(result, i, j, matrix_get(m, i, j) + v->data[j]);
        }
    }
}

void matrix_copy(Matrix *dest, const Matrix *src) {
    assert(dest->rows == src->rows);
    assert(dest->rows == src->rows);
    memcpy(dest->data, src->data, sizeof(float) * src->rows * src->cols);
}

void matrix_copy_vector_into_row(Matrix *m, const Vector *v, int row_idx) {
    assert(v->size == m->cols);
    assert(row_idx < m->rows);
    memcpy(&m->data[m->cols * row_idx], v->data, sizeof(float) * v->size);
}

// External definitions for inline functions (C99 fallback)
extern inline float matrix_get(const Matrix *m, int row, int col);
extern inline void matrix_set(Matrix *m, int row, int col, float value);

Vector *get_row_as_vector(Matrix *m, int row) {
    Vector *v = vector_create(m->cols);
    float *row_start_ptr = &m->data[row * m->cols];
    memcpy(v->data, row_start_ptr, sizeof(float) * m->cols);
    return v;
}

void matrix_print(const Matrix *m) {
    int i, j;
    printf("[");
    for (i = 0; i < m->rows - 1; i++) {
        for (j = 0; j < m->cols - 1; j++) {
            printf("%f,", matrix_get(m, i, j));
        }
        printf("%f,\n   ", matrix_get(m, i, j));
    }
    for (j = 0; j < m->cols - 1; j++) {
        printf("%f,", matrix_get(m, i, j));
    }
    printf("%f]", matrix_get(m, i, j));
}

void matrix_sum_rows(Vector *result, const Matrix *m) {
    assert(m->rows == result->size);
    for (int i = 0; i < m->rows; i++) {
        float row_sum = 0.f;
        for (int j = 0; j < m->cols; j++) {
            row_sum += matrix_get(m, i, j);
        }
        result->data[i] = row_sum;
    }
}
void matrix_sum_cols(Vector *result, const Matrix *m) {
    assert(m->cols == result->size);
    for (int j = 0; j < m->cols; j++) {
        float col_sum = 0.f;
        for (int i = 0; i < m->rows; i++) {
            col_sum += matrix_get(m, i, j);
        }
        result->data[j] = col_sum;
    }
}

int matrix_equals(const Matrix *a, const Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return 0;
    }
    for (int i = 0; i < a->rows * a->cols; i++) {
        if (fabs(a->data[i] - b->data[i]) >= 1e-6) {
            return 0;
        }
    }
    return 1;
}
