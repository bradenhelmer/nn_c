/*
 * vector.h - Vector data structure and operations for linear algebra
 *
 * Provides vector operations including creation, basic arithmetic,
 * element-wise operations, dot product, and utility functions for
 * neural network computations.
 */

#ifndef VECTOR_H
#define VECTOR_H

// Forward declaration to avoid circular dependency
struct Matrix;

typedef struct Vector {
    float *data;
    int size;
} Vector;

// Creation and destruction
Vector *vector_create(int size);
Vector *vector_zeros(int size);
Vector *vector_ones(int size);
Vector *vector_random(int size, float min, float max);
void vector_free(Vector *v);

// Basic operations
void vector_add(Vector *result, const Vector *a, const Vector *b);
void vector_subtract(Vector *result, const Vector *a, const Vector *b);
void vector_scale(Vector *result, const Vector *v, float scalar);
float vector_dot(const Vector *a, const Vector *b);
void vector_fill(Vector *v, float scalar);

// Element-wise operations
void vector_elementwise_multiply(Vector *result, const Vector *a,
                                 const Vector *b); // Hadamard product
void vector_elementwise_divide(Vector *result, const Vector *a, const Vector *b);

// Other
void vector_outer_product(struct Matrix *result, const Vector *a, const Vector *b);

// Utility functions
void vector_copy(Vector *dest, const Vector *src);
float vector_sum(const Vector *v);
float vector_min(const Vector *v);
float vector_max(const Vector *v);
void vector_print(const Vector *v);
int vector_equals(const Vector *a, const Vector *b);

#endif /* ifndef VECTOR_H */
