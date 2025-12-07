/*
 * matrix.h - Matrix data structure and operations for linear algebra
 * Provides core matrix operations including creation, multiplication,
 * transpose, element-wise operations, and matrix-vector operations
 * for neural network computations.
 */
#ifndef MATRIX_H
#define MATRIX_H

#include <assert.h>

// Forward declaration to avoid circular dependency
struct Vector;

typedef struct Matrix {
    float *data;
    int rows;
    int cols;
} Matrix;

// Creation and destruction
Matrix *matrix_create(int rows, int cols);
Matrix *matrix_zeros(int rows, int cols);
Matrix *matrix_ones(int rows, int cols);
Matrix *matrix_random(int rows, int cols, float min, float max);
Matrix *matrix_identity(int size);
void matrix_free(Matrix *m);

// Core linear algebra
void matrix_multiply(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_transpose(Matrix *result, const Matrix *m);
void matrix_add(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_subtract(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_scale(Matrix *result, const Matrix *m, float scalar);

// Element-wise operations
void matrix_multiply_elementwise(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_add_scalar(Matrix *result, const Matrix *m, float scalar);
void matrix_fill(Matrix *m, float scalar);

// Matrix-vector operations
void matrix_vector_multiply(struct Vector *result, const Matrix *m, const struct Vector *v);
void matrix_transpose_vector_multiply(struct Vector *result, const Matrix *m,
                                      const struct Vector *v);
void matrix_add_vector(Matrix *result, const Matrix *m, const struct Vector *v); // Broadcasting

// Utility functions
void matrix_copy(Matrix *dest, const Matrix *src);
void matrix_copy_vector_into_row(Matrix *m, const struct Vector *v, int row_idx);
void matrix_copy_row_to_vector(struct Vector *dest, const Matrix *src, int row_idx);
void matrix_print(const Matrix *m);
int matrix_equals(const Matrix *a, const Matrix *b);

// Inline accessor functions (defined in header for inlining)
inline float matrix_get(const Matrix *m, int row, int col) {
    assert(row < m->rows);
    assert(col < m->cols);
    return m->data[row * m->cols + col];
}

inline void matrix_set(Matrix *m, int row, int col, float value) {
    assert(row < m->rows);
    assert(col < m->cols);
    m->data[row * m->cols + col] = value;
}

// Other getters
struct Vector *get_row_as_vector(Matrix *m, int row);

// Special operations for neural networks
void matrix_sum_rows(struct Vector *result, const Matrix *m); // For gradient accumulation
void matrix_sum_cols(struct Vector *result, const Matrix *m); // For batch operations

#endif /* ifndef MATRIX_H */
