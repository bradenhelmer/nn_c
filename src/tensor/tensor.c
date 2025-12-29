/*
 * tensor.c
 *
 * Tensor implementations.
 */
#include "tensor.h"
#include "../utils/utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline void _tensor_set_size_metadata(Tensor *t, int ndim, int *shape) {
    t->ndim = ndim;

    t->shape = (int *)malloc(ndim * sizeof(int));
    memcpy(t->shape, shape, ndim * sizeof(int));

    // Compute strides left to right
    t->strides = (int *)malloc(ndim * sizeof(int));
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * shape[i + 1];
    }

    t->size = t->strides[0] * shape[0];
}

static inline Tensor *_tensor_create(int ndim, int *shape) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    _tensor_set_size_metadata(t, ndim, shape);
    t->data = (float *)calloc(t->size, sizeof(float));
    t->owner = 1;
    return t;
}

Tensor *tensor_create1d(int x) {
    return _tensor_create(1, (int[]){x});
}

Tensor *tensor_create2d(int x, int y) {
    return _tensor_create(2, (int[]){x, y});
}

Tensor *tensor_create3d(int x, int y, int z) {
    return _tensor_create(3, (int[]){x, y, z});
}

Tensor *tensor_create4d(int x, int y, int z, int a) {
    return _tensor_create(4, (int[]){x, y, z, a});
}

Tensor *tensor_zeros(int ndim, int *shape) {
    return _tensor_create(ndim, shape);
}

Tensor *tensor_random(int ndim, int *shape, float min, float max) {
    Tensor *t = _tensor_create(ndim, shape);
    for (int i = 0; i < t->size; i++) {
        t->data[i] = rand_rangef(min, max);
    }
    return t;
}

void tensor_free(Tensor *t) {
    if (t->owner) {
        free(t->data);
    }
    free(t->shape);
    free(t->strides);
    free(t);
}

float tensor_get2d(const Tensor *t, int i, int j) {
    return t->data[tensor_index2d(t, i, j)];
}

void tensor_set2d(const Tensor *t, int i, int j, float value) {
    t->data[tensor_index2d(t, i, j)] = value;
}

float tensor_get3d(const Tensor *t, int i, int j, int k) {
    return t->data[tensor_index3d(t, i, j, k)];
}

void tensor_set3d(const Tensor *t, int i, int j, int k, float value) {
    t->data[tensor_index3d(t, i, j, k)] = value;
}

float tensor_get4d(const Tensor *t, int i, int j, int k, int l) {
    return t->data[tensor_index4d(t, i, j, k, l)];
}

void tensor_set4d(const Tensor *t, int i, int j, int k, int l, float value) {
    t->data[tensor_index4d(t, i, j, k, l)] = value;
}

void tensor_fill(Tensor *t, float val) {
    for (int i = 0; i < t->size; i++) {
        t->data[i] = val;
    }
}

void tensor_copy(Tensor *dest, const Tensor *src) {
    assert(dest->ndim == src->ndim);
    for (int shape = 0; shape < dest->ndim; shape++) {
        assert(dest->shape[shape] == src->shape[shape]);
    }
    memcpy(dest->data, src->data, sizeof(float) * src->size);
}

Tensor *tensor_clone(const Tensor *t) {
    Tensor *clone = _tensor_create(t->ndim, t->shape);
    memcpy(clone->data, t->data, sizeof(float) * t->size);
    return clone;
}

void tensor_print_shape(const Tensor *t) {
    int i;
    printf("Shape: {");
    for (i = 0; i < t->ndim - 1; i++) {
        printf("%d, ", t->shape[i]);
    }
    printf("%d}", t->shape[i]);
}

void tensor_print(const Tensor *t) {
    printf("[");
    for (int i = 0; i < t->size; i++) {
        printf("%.4f", t->data[i]);
        if (i < t->size - 1) {
            printf(", ");
        }
    }
    printf("]");
}

void tensor_get_row(Tensor *dest, const Tensor *src, int row_idx) {
    assert(src->ndim == 2);
    assert(dest->ndim == 1);
    assert(dest->shape[0] == src->shape[1]);
    int cols = src->shape[1];
    memcpy(dest->data, &src->data[row_idx * cols], cols * sizeof(float));
}

void tensor_set_row(Tensor *dest, const Tensor *src, int row_idx) {
    assert(dest->ndim == 2);
    assert(src->ndim == 1);
    assert(src->shape[0] == dest->shape[1]);
    int cols = dest->shape[1];
    memcpy(&dest->data[row_idx * cols], src->data, cols * sizeof(float));
}

int tensor_equals(const Tensor *a, const Tensor *b) {
    if (a->ndim != b->ndim || a->size != b->size) {
        return 0;
    }
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return 0;
        }
    }
    for (int i = 0; i < a->size; i++) {
        if (a->data[i] != b->data[i]) {
            return 0;
        }
    }
    return 1;
}

void tensor_scale(Tensor *dest, const Tensor *src, float scalar) {
    assert(dest->size == src->size);
    for (int i = 0; i < src->size; i++) {
        dest->data[i] = src->data[i] * scalar;
    }
}

float tensor_dot(const Tensor *a, const Tensor *b) {
    assert(a->ndim == 1 && b->ndim == 1);
    assert(a->shape[0] == b->shape[0]);
    float sum = 0.0f;
    for (int i = 0; i < a->shape[0]; i++) {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}

float tensor_sum(const Tensor *t) {
    float sum = 0.0f;
    for (int i = 0; i < t->size; i++) {
        sum += t->data[i];
    }
    return sum;
}

float tensor_max(const Tensor *t) {
    float max = t->data[0];
    for (int i = 1; i < t->size; i++) {
        if (t->data[i] > max) {
            max = t->data[i];
        }
    }
    return max;
}

int tensor_argmax(const Tensor *t) {
    int max_idx = 0;
    float max = t->data[0];
    for (int i = 1; i < t->size; i++) {
        if (t->data[i] > max) {
            max = t->data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

void tensor_add(Tensor *dest, const Tensor *a, const Tensor *b) {
    assert(dest->size == a->size && a->size == b->size);
    for (int i = 0; i < dest->size; i++) {
        dest->data[i] = a->data[i] + b->data[i];
    }
}

void tensor_elementwise_mul(Tensor *dest, const Tensor *a, const Tensor *b) {
    assert(dest->size == a->size && a->size == b->size);
    for (int i = 0; i < dest->size; i++) {
        dest->data[i] = a->data[i] * b->data[i];
    }
}

void tensor_matvec_mul(Tensor *dest, const Tensor *mat, const Tensor *vec) {
    assert(mat->ndim == 2);
    assert(vec->ndim == 1);
    assert(dest->ndim == 1);
    assert(mat->shape[1] == vec->shape[0]);
    assert(dest->shape[0] == mat->shape[0]);

    int m = mat->shape[0];
    int n = mat->shape[1];

    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += tensor_get2d(mat, i, j) * vec->data[j];
        }
        dest->data[i] = sum;
    }
}

void tensor_matvec_mul_transpose(Tensor *dest, const Tensor *mat, const Tensor *vec) {
    assert(mat->ndim == 2);
    assert(vec->ndim == 1);
    assert(dest->ndim == 1);
    assert(mat->shape[0] == vec->shape[0]);
    assert(dest->shape[0] == mat->shape[1]);

    int m = mat->shape[0];
    int n = mat->shape[1];

    // Zero dest first since we accumulate
    tensor_fill(dest, 0.0f);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dest->data[j] += tensor_get2d(mat, i, j) * vec->data[i];
        }
    }
}

void tensor_matmul(Tensor *dest, const Tensor *a, const Tensor *b) {
    assert(a->ndim == 2);
    assert(b->ndim == 2);
    assert(dest->ndim == 2);

    const int m = a->shape[0];
    const int n = b->shape[1];
    const int k = a->shape[1];

    assert(m == dest->shape[0]);
    assert(n == dest->shape[1]);
    assert(k == b->shape[0]);

    float *dest_base = dest->data;
    float *a_base = a->data;
    float *b_base = b->data;

    for (int row = 0; row < m; row++) {
        float *dest_row_ptr = dest_base + row * n;
        float *a_row_ptr = a_base + row * k;
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int inner = 0; inner < k; inner++) {
                sum += a_row_ptr[inner] * b_base[inner * n + col];
            }
            *dest_row_ptr++ = sum;
        }
    }
}

void tensor_outer_product(Tensor *dest, const Tensor *a, const Tensor *b) {
    assert(a->ndim == 1);
    assert(b->ndim == 1);
    assert(dest->ndim == 2);
    assert(dest->shape[0] == a->shape[0]);
    assert(dest->shape[1] == b->shape[0]);

    int m = a->shape[0];
    int n = b->shape[0];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            tensor_set2d(dest, i, j, a->data[i] * b->data[j]);
        }
    }
}

Tensor *tensor_transpose2d(const Tensor *t) {
    assert(t->ndim == 2);

    int m = t->shape[0];
    int n = t->shape[1];

    Tensor *transposed = tensor_create2d(n, m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float val = tensor_get2d(t, i, j);
            tensor_set2d(transposed, j, i, val);
        }
    }

    return transposed;
}

Tensor *tensor_pad2d(const Tensor *t, int padding) {
    int C = t->shape[0];
    int H = t->shape[1];
    int W = t->shape[2];

    Tensor *padded = tensor_create3d(C, H + 2 * padding, W + 2 * padding);

    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float val = tensor_get3d(t, c, h, w);
                tensor_set3d(padded, c, h + padding, w + padding, val);
            }
        }
    }

    return padded;
}

Tensor *tensor_unpad2d(const Tensor *t, int padding) {
    int C = t->shape[0];
    int H = t->shape[1];
    int W = t->shape[2];

    Tensor *unpadded = tensor_create3d(C, H - 2 * padding, W - 2 * padding);

    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H - 2 * padding; h++) {
            for (int w = 0; w < W - 2 * padding; w++) {
                float val = tensor_get3d(t, c, h + padding, w + padding);
                tensor_set3d(unpadded, c, h, w, val);
            }
        }
    }
    return unpadded;
}

Tensor *tensor_flatten(const Tensor *t) {
    Tensor *view = (Tensor *)malloc(sizeof(Tensor));
    view->data = t->data;
    view->ndim = 1;
    view->shape = (int *)malloc(sizeof(int));
    view->shape[0] = t->size;
    view->strides = (int *)malloc(sizeof(int));
    view->strides[0] = 1;
    view->size = t->size;
    view->owner = 0;
    return view;
}

Tensor *tensor_unflatten(const Tensor *t, int ndim, int *new_shape) {
    return tensor_view(t, ndim, new_shape);
}

static int _check_new_size(__attribute__((unused)) const Tensor *t, int ndim, int *new_shape) {
    int new_size = 1;
    for (int i = 0; i < ndim; i++) {
        new_size *= new_shape[i];
    }
    assert(new_size == t->size);
    return new_size;
}

Tensor *tensor_view(const Tensor *t, int ndim, int *new_shape) {
    _check_new_size(t, ndim, new_shape);
    Tensor *view = (Tensor *)malloc(sizeof(Tensor));
    _tensor_set_size_metadata(view, ndim, new_shape);
    view->data = t->data;
    view->owner = 0;
    return view;
}

Tensor *tensor_reshape_inplace(Tensor *t, int ndim, int *new_shape) {
    const int new_size = _check_new_size(t, ndim, new_shape);

    free(t->shape);
    free(t->strides);

    t->ndim = ndim;
    t->shape = (int *)malloc(ndim * sizeof(int));
    memcpy(t->shape, new_shape, ndim * sizeof(int));

    t->strides = (int *)malloc(ndim * sizeof(int));
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * new_shape[i + 1];
    }

    t->size = new_size;
    return t;
}

float tensor_sum_2drow(const Tensor *t, int row_idx) {
    assert(row_idx < t->shape[0]);
    const int cols = t->shape[1];
    float *row_ptr = t->data + row_idx + cols;
    float sum = 0.0f;
    for (int col = 0; col < cols; col++) {
        sum += *row_ptr;
        row_ptr++;
    }
    return sum;
}
