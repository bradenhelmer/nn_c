/*
 * tensor.h
 *
 * Tensor declarations
 */
#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float *data;
    int *shape;   // e.g {32, 28, 28} for 32 channels of 28x28
    int *strides; // precomputed to faster indexing
    int ndim;     // Number of dimensions
    int size;     // Total elements
} Tensor;

// Lifecycle
Tensor *tensor_create(int ndim, int *shape);
Tensor *tensor_zeros(int ndim, int *shape);
void tensor_free(Tensor *t);

// 2D Accessors
float tensor_get2d(const Tensor *t, int i, int j);
void tensor_set2d(const Tensor *t, int i, int j, float value);
static inline int tensor_index2d(const Tensor *t, int i, int j) {
    return i * t->strides[0] + j;
}

// 3D Accessors
float tensor_get3d(const Tensor *t, int i, int j, int k);
void tensor_set3d(const Tensor *t, int i, int j, int k, float value);
static inline int tensor_index3d(const Tensor *t, int i, int j, int k) {
    return i * t->strides[0] + j * t->strides[1] + k;
}

// 4D Accessors
float tensor_get4d(const Tensor *t, int i, int j, int k, int l);
void tensor_set4d(const Tensor *t, int i, int j, int k, int l, float value);
static inline int tensor_index4d(const Tensor *t, int i, int j, int k, int l) {
    return i * t->strides[0] + j * t->strides[1] + k * t->strides[2] + l;
}

// Utilities
void tensor_fill(Tensor *t, float val);
void tensor_copy(Tensor *dest, const Tensor *src);
Tensor *tensor_clone(const Tensor *t);
void tensor_print_shape(const Tensor *t);
Tensor *tensor_flatten(Tensor *t);
Tensor *tensor_unflatten(Tensor *t, int c, int h, int w);

// For conv layers
Tensor *tensor_pad2d(const Tensor *t, int padding);
Tensor *tensor_unpad2d(const Tensor *t, int padding);

#endif /* ifndef TENSOR_H */
