/*
 * matrix.h - Matrix data structure definition for linear algebra operations
 */

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    float *data;
    int rows;
    int cols;
} Matrix;

#endif /* ifndef MATRIX_H */
