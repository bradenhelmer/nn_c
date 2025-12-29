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
    int owner;    // Does this tensor own the data? For views
} Tensor;

// Lifecycle
Tensor *tensor_create1d(int x);
Tensor *tensor_create2d(int x, int y);
Tensor *tensor_create3d(int x, int y, int z);
Tensor *tensor_create4d(int x, int y, int z, int a);
Tensor *tensor_zeros(int ndim, int *shape);
Tensor *tensor_random(int ndim, int *shape, float min, float max);
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
void tensor_print(const Tensor *t);
Tensor *tensor_flatten(const Tensor *t);
Tensor *tensor_unflatten(const Tensor *t, int ndim, int *new_shape);
Tensor *tensor_view(const Tensor *t, int ndim, int *shape);
Tensor *tensor_reshape_inplace(Tensor *t, int ndim, int *new_shape);

// Row operations for 2D tensors (used in batch processing)
// Copies row `row_idx` from 2D tensor `src` into 1D tensor `dest`
void tensor_get_row(Tensor *dest, const Tensor *src, int row_idx);
// Copies 1D tensor `src` into row `row_idx` of 2D tensor `dest`
void tensor_set_row(Tensor *dest, const Tensor *src, int row_idx);

// Comparison
int tensor_equals(const Tensor *a, const Tensor *b);

// Scalar operations
void tensor_scale(Tensor *dest, const Tensor *src, float scalar);

// 1D tensor operations (vector-like)
float tensor_dot(const Tensor *a, const Tensor *b);
float tensor_sum(const Tensor *t);
float tensor_max(const Tensor *t);
int tensor_argmax(const Tensor *t);

// For conv layers
Tensor *tensor_pad2d(const Tensor *t, int padding);
Tensor *tensor_unpad2d(const Tensor *t, int padding);
float tensor_sum_2drow(const Tensor *t, int row_idx);

// Element-wise addition: dest = a + b
// Works for tensors of any dimension (shapes must match)
void tensor_add(Tensor *dest, const Tensor *a, const Tensor *b);

// Element-wise multiplication (Hadamard product): dest = a ⊙ b
// Works for tensors of any dimension (shapes must match)
void tensor_elementwise_mul(Tensor *dest, const Tensor *a, const Tensor *b);

// Matrix-vector multiplication: dest = mat * vec
// mat: 2D tensor of shape (m, n)
// vec: 1D tensor of shape (n,)
// dest: 1D tensor of shape (m,)
void tensor_matvec_mul(Tensor *dest, const Tensor *mat, const Tensor *vec);

// Transposed matrix-vector multiplication: dest = mat^T * vec
// mat: 2D tensor of shape (m, n)
// vec: 1D tensor of shape (m,)
// dest: 1D tensor of shape (n,)
// Used for backpropagation: computes gradient w.r.t. input
void tensor_matvec_mul_transpose(Tensor *dest, const Tensor *mat, const Tensor *vec);

// Tensor matrix multiplication: dest = a * b
// a: 2D tensor of shape (m, k)
// b: 2D tensor of shape (k, n)
// dest: 2D tensor of shape (m, n)
void tensor_matmul(Tensor *dest, const Tensor *a, const Tensor *b);

// Outer product: dest = a ⊗ b^T
// a: 1D tensor of shape (m,)
// b: 1D tensor of shape (n,)
// dest: 2D tensor of shape (m, n)
// Used for computing weight gradients: dW = dz ⊗ input
void tensor_outer_product(Tensor *dest, const Tensor *a, const Tensor *b);

// Transpose: returns t^T
// t: 2D tensor of shape (m, n)
// returns: 2D tensor of shape (n, m)
Tensor *tensor_transpose2d(const Tensor *t);

#endif /* ifndef TENSOR_H */
