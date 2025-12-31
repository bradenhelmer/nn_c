/*
 * tensor.c
 *
 * Tensor implementations.
 */
#include "tensor.h"
#include "../utils/utils.h"
#include <assert.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

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

Tensor *tensor_create(int ndim, int *shape) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    _tensor_set_size_metadata(t, ndim, shape);
    t->data = (float *)calloc(t->size, sizeof(float));
    t->owner = 1;
    return t;
}

Tensor *tensor_create1d(int x) {
    return tensor_create(1, (int[]){x});
}

Tensor *tensor_create2d(int x, int y) {
    return tensor_create(2, (int[]){x, y});
}

Tensor *tensor_create3d(int x, int y, int z) {
    return tensor_create(3, (int[]){x, y, z});
}

Tensor *tensor_create4d(int x, int y, int z, int a) {
    return tensor_create(4, (int[]){x, y, z, a});
}

Tensor *tensor_zeros(int ndim, int *shape) {
    return tensor_create(ndim, shape);
}

Tensor *tensor_random(int ndim, int *shape, float min, float max) {
    Tensor *t = tensor_create(ndim, shape);
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
    Tensor *clone = tensor_create(t->ndim, t->shape);
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

static void _tensor_matvec_mul_simd(Tensor *dest, const Tensor *mat, const Tensor *vec) {
    const int m = mat->shape[0];
    const int n = mat->shape[1];
    float *dest_data = dest->data;
    float *mat_data = mat->data;
    float *vec_data = vec->data;
    for (int row = 0; row < m; row++) {
        float *mat_row_ptr = mat_data + row * n;

        // Vectorized accumulator
        __m512 sum_vec = _mm512_setzero_ps();

        // Process 16 at at time
        int col = 0;
        for (; col <= n - 16; col += 16) {
            __m512 mat_chunk = _mm512_loadu_ps(&mat_row_ptr[col]);
            __m512 vec_chunk = _mm512_loadu_ps(&vec_data[col]);
            sum_vec = _mm512_fmadd_ps(mat_chunk, vec_chunk, sum_vec);
        }

        // Horizontal sum: reduce 16 partial sums to 1
        __m256 low = _mm512_extractf32x8_ps(sum_vec, 0);
        __m256 high = _mm512_extractf32x8_ps(sum_vec, 1);
        __m256 sum256 = _mm256_hadd_ps(high, low); // 8 floats
        sum256 = _mm256_hadd_ps(sum256, sum256);   // 4 floats
        sum256 = _mm256_hadd_ps(sum256, sum256);   // 2 floats
        sum256 = _mm256_hadd_ps(sum256, sum256);   // 1 floats
        float sum = _mm256_cvtss_f32(sum256);

        for (; col < n; col++) {
            sum += mat_row_ptr[col] * vec_data[col];
        }
        dest_data[row] = sum;
    }
}

UNUSED static void _tensor_matvec_mul_trivial(Tensor *dest, const Tensor *mat, const Tensor *vec) {
    int m = mat->shape[0];
    int n = mat->shape[1];

    float *dest_data = dest->data;
    float *mat_data = mat->data;
    float *vec_data = vec->data;

    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        float *mat_row_ptr = mat_data + i * n;
        for (int j = 0; j < n; j++) {
            sum += mat_row_ptr[j] * vec_data[j];
        }
        dest_data[i] = sum;
    }
}

void tensor_matvec_mul(Tensor *dest, const Tensor *mat, const Tensor *vec) {
    assert(mat->ndim == 2);
    assert(vec->ndim == 1);
    assert(dest->ndim == 1);
    assert(mat->shape[1] == vec->shape[0]);
    assert(dest->shape[0] == mat->shape[0]);

    _tensor_matvec_mul_simd(dest, mat, vec);
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

    float *dest_data = dest->data;
    float *mat_data = mat->data;
    float *vec_data = vec->data;

    for (int i = 0; i < m; i++) {
        float *mat_row_ptr = mat_data + i * n;
        for (int j = 0; j < n; j++) {
            dest_data[j] += mat_row_ptr[j] * vec_data[i];
        }
    }
}

#define BLOCK_SIZE 64
#define MIN(a, b) ((a) < (b) ? (a) : (b))

UNUSED static void _tensor_matmul_tiled_simd(Tensor *dest, const Tensor *a, const Tensor *b) {
    const int m = a->shape[0];
    const int n = b->shape[1];
    const int k = a->shape[1];

    float *dest_base = dest->data;
    float *a_base = a->data;
    float *b_base = b->data;

    for (int tile_row = 0; tile_row < m; tile_row += BLOCK_SIZE) {
        for (int tile_col = 0; tile_col < n; tile_col += BLOCK_SIZE) {
            for (int tile_inner = 0; tile_inner < k; tile_inner += BLOCK_SIZE) {

                int row_end = MIN(tile_row + BLOCK_SIZE, m);
                int col_end = MIN(tile_col + BLOCK_SIZE, n);
                int inner_end = MIN(tile_inner + BLOCK_SIZE, k);
                int col_count = col_end - tile_col;

                for (int row = tile_row; row < row_end; row++) {
                    float *dest_row = dest_base + row * n + tile_col;

                    for (int inner = tile_inner; inner < inner_end; inner++) {
                        float a_val = a_base[row * k + inner];
                        float *b_row = b_base + inner * n + tile_col;

                        // Broadcast a_val to all 16 lanes
                        __m512 a_vec = _mm512_set1_ps(a_val);

                        // Process 16 at a time.
                        int col = 0;
                        for (; col < col_count - 16; col += 16) {
                            __m512 b_vec = _mm512_loadu_ps(&b_row[col]);
                            __m512 dest_vec = _mm512_loadu_ps(&dest_row[col]);
                            dest_vec = _mm512_fmadd_ps(a_vec, b_vec, dest_vec);
                            _mm512_storeu_ps(&dest_row[col], dest_vec);
                        }

                        for (; col < col_count; col++) {
                            dest_row[col] += a_val * b_row[col];
                        }
                    }
                }
            }
        }
    }
}

static void _tensor_matmul_tiled(Tensor *dest, const Tensor *a, const Tensor *b) {
    const int m = a->shape[0];
    const int n = b->shape[1];
    const int k = a->shape[1];
    float *dest_base = dest->data;
    float *a_base = a->data;
    float *b_base = b->data;
    for (int tile_row = 0; tile_row < m; tile_row += BLOCK_SIZE) {
        for (int tile_col = 0; tile_col < n; tile_col += BLOCK_SIZE) {
            for (int tile_inner = 0; tile_inner < k; tile_inner += BLOCK_SIZE) {
                int row_end = MIN(tile_row + BLOCK_SIZE, m);
                int col_end = MIN(tile_col + BLOCK_SIZE, n);
                int inner_end = MIN(tile_inner + BLOCK_SIZE, k);
                for (int row = tile_row; row < row_end; row++) {
                    for (int inner = tile_inner; inner < inner_end; inner++) {
                        float a_val = a_base[row * k + inner];
                        float *b_row = b_base + inner * n + tile_col;
                        float *c_row = dest_base + row * n + tile_col;
                        for (int col = 0; col < col_end - tile_col; col++) {
                            c_row[col] += a_val * b_row[col];
                        }
                    }
                }
            }
        }
    }
}

UNUSED static void _tensor_matmul_trivial(Tensor *dest, const Tensor *a, const Tensor *b) {

    const int m = a->shape[0];
    const int n = b->shape[1];
    const int k = a->shape[1];

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

void tensor_matmul(Tensor *dest, const Tensor *a, const Tensor *b) {
    assert(a->ndim == 2);
    assert(b->ndim == 2);
    assert(dest->ndim == 2);

    __attribute__((unused)) const int m = a->shape[0];
    __attribute__((unused)) const int n = b->shape[1];
    __attribute__((unused)) const int k = a->shape[1];

    assert(m == dest->shape[0]);
    assert(n == dest->shape[1]);
    assert(k == b->shape[0]);

    _tensor_matmul_tiled(dest, a, b);
}

void tensor_outer_product(Tensor *dest, const Tensor *a, const Tensor *b) {
    assert(a->ndim == 1);
    assert(b->ndim == 1);
    assert(dest->ndim == 2);
    assert(dest->shape[0] == a->shape[0]);
    assert(dest->shape[1] == b->shape[0]);

    int m = a->shape[0];
    int n = b->shape[0];

    float *a_data = a->data;
    float *b_data = b->data;
    float *dest_data = dest->data;

    for (int i = 0; i < m; i++) {
        float *dest_row_ptr = dest_data + i * n;
        for (int j = 0; j < n; j++) {
            dest_row_ptr[j] = a_data[i] * b_data[j];
        }
    }
}

void tensor_outer_product_accumulate(Tensor *dest, const Tensor *a, const Tensor *b) {
    int m = a->shape[0];
    int n = b->shape[0];
    float *dest_data = dest->data;
    float *a_data = a->data;
    float *b_data = b->data;

    for (int i = 0; i < m; i++) {
        float a_val = a_data[i];
        float *dest_row = dest_data + i * n;
        for (int j = 0; j < n; j++) {
            dest_row[j] += a_val * b_data[j];
        }
    }
}

Tensor *tensor_transpose2d(const Tensor *t) {
    assert(t->ndim == 2);

    int m = t->shape[0];
    int n = t->shape[1];

    Tensor *transposed = tensor_create2d(n, m);
    float *t_data = t->data;
    float *transposed_data = transposed->data;

    for (int i = 0; i < m; i++) {
        float *t_row_ptr = t_data + i * n;
        for (int j = 0; j < n; j++) {
            transposed_data[j * m + i] = t_row_ptr[j];
        }
    }

    return transposed;
}

Tensor *tensor_pad2d(const Tensor *t, int padding) {
    int C = t->shape[0];
    int H = t->shape[1];
    int W = t->shape[2];

    Tensor *padded = tensor_create3d(C, H + 2 * padding, W + 2 * padding);

    float *t_data = t->data;
    const int t_c_stride = t->strides[0];
    const int t_h_stride = t->strides[1];

    float *padded_data = padded->data;
    const int padded_c_stride = padded->strides[0];
    const int padded_h_stride = padded->strides[1];

    for (int c = 0; c < C; c++) {
        float *t_c_base = t_data + c * t_c_stride;
        float *padded_c_base = padded_data + c * padded_c_stride;
        for (int h = 0; h < H; h++) {
            float *t_row_ptr = t_c_base + h * t_h_stride;
            float *padded_row_ptr = padded_c_base + (h + padding) * padded_h_stride;
            for (int w = 0; w < W; w++) {
                padded_row_ptr[w + padding] = t_row_ptr[w];
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

    float *t_data = t->data;
    const int t_c_stride = t->strides[0];
    const int t_h_stride = t->strides[1];

    float *padded_data = unpadded->data;
    const int unpadded_c_stride = unpadded->strides[0];
    const int unpadded_h_stride = unpadded->strides[1];

    for (int c = 0; c < C; c++) {
        float *t_c_base = t_data + c * t_c_stride;
        float *unpadded_c_base = padded_data + c * unpadded_c_stride;
        for (int h = 0; h < H - 2 * padding; h++) {
            float *t_row_ptr = t_c_base + (h + padding) * t_h_stride;
            float *unpadded_row_ptr = unpadded_c_base + h * unpadded_h_stride;
            for (int w = 0; w < W - 2 * padding; w++) {
                unpadded_row_ptr[w] = t_row_ptr[w + padding];
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
    float *row_ptr = t->data + row_idx * cols;
    float sum = 0.0f;
    for (int col = 0; col < cols; col++) {
        sum += *row_ptr;
        row_ptr++;
    }
    return sum;
}
