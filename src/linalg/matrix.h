/*
 * matrix.h - Matrix data structure and operations for linear algebra
 *
 * Provides core matrix operations including creation, multiplication,
 * transpose, element-wise operations, and matrix-vector operations
 * for neural network computations.
 */
#ifndef MATRIX_H
#define MATRIX_H
#include "vector.h"

typedef struct {
    float *data;
    int rows;
    int cols;
} Matrix;

// Creation and destruction
Matrix *matrix_create(int rows, int cols);
Matrix *matrix_zeros(int rows, int cols);
Matrix *matrix_ones(int rows, int cols);
Matrix *matrix_random(int rows, int cols, float min, float max);
Matrix *matrix_identity(int rows, int cols);
void matrix_free(Matrix *m);

// Core linear algebra
void matrix_multiply(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_transpose(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_add(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_subtract(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_scale(Matrix *result, const Matrix *m, float scalar);

// Element-wise operations
void matrix_multiply_elementwise(Matrix *result, const Matrix *a, const Matrix *b);
void matrix_add_scalar(Matrix *result, const Matrix *m, float scalar);

// Matrix-vector operations
void matrix_vector_multiply(Vector *result, const Matrix *m, const Vector *v);
void matrix_add_vector(Matrix *result, const Matrix *m, const Vector *v); // Broadcasting

// Utility functions
void matrix_copy(Matrix *dest, const Matrix *src);
float matrix_get(const Matrix *m, int row, int col);
void matrix_set(Matrix *m, int row, int col, float value);
void matrix_print(const Matrix *m);

// Special operations for neural networks
void matrix_sum_rows(Vector *result, const Matrix *m); // For gradient accumulation
void matrix_sum_cols(Vector *result, const Matrix *m); // For batch operations

#endif /* ifndef MATRIX_H */
