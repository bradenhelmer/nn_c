/*
 * test_im2col.c - Tests for convolutional layer im2col operations.
 */

#include "nn/layer.h"
#include "tensor/tensor.h"
#include "test_runner.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdio.h>

void test_im2col() {

    Tensor *X = tensor_create3d(1, 2, 2);
    X->data[0] = 1.0f;
    X->data[1] = 2.0f;
    X->data[2] = 3.0f;
    X->data[3] = 4.0f;

    Tensor *X_pad = tensor_pad2d(X, 1);
    Tensor *converted = im2col(X_pad, 3, 1);

    assert(float_equals(converted->data[0], 0.0f));
    assert(float_equals(converted->data[1], 0.0f));
    assert(float_equals(converted->data[2], 0.0f));
    assert(float_equals(converted->data[3], 1.0f));
    assert(float_equals(converted->data[4], 0.0f));
    assert(float_equals(converted->data[5], 0.0f));
    assert(float_equals(converted->data[6], 1.0f));
    assert(float_equals(converted->data[7], 2.0f));
    assert(float_equals(converted->data[8], 0.0f));
    assert(float_equals(converted->data[9], 0.0f));
    assert(float_equals(converted->data[10], 2.0f));
    assert(float_equals(converted->data[11], 0.0f));
    assert(float_equals(converted->data[12], 0.0f));
    assert(float_equals(converted->data[13], 1.0f));
    assert(float_equals(converted->data[14], 0.0f));
    assert(float_equals(converted->data[15], 3.0f));
    assert(float_equals(converted->data[16], 1.0f));
    assert(float_equals(converted->data[17], 2.0f));
    assert(float_equals(converted->data[18], 3.0f));
    assert(float_equals(converted->data[19], 4.0f));
    assert(float_equals(converted->data[20], 2.0f));
    assert(float_equals(converted->data[21], 0.0f));
    assert(float_equals(converted->data[22], 4.0f));
    assert(float_equals(converted->data[23], 0.0f));
    assert(float_equals(converted->data[24], 0.0f));
    assert(float_equals(converted->data[25], 3.0f));
    assert(float_equals(converted->data[26], 0.0f));
    assert(float_equals(converted->data[27], 0.0f));
    assert(float_equals(converted->data[28], 3.0f));
    assert(float_equals(converted->data[29], 4.0f));
    assert(float_equals(converted->data[30], 0.0f));
    assert(float_equals(converted->data[31], 0.0f));
    assert(float_equals(converted->data[32], 4.0f));
    assert(float_equals(converted->data[33], 0.0f));
    assert(float_equals(converted->data[34], 0.0f));
    assert(float_equals(converted->data[35], 0.0f));

    tensor_free(X);
    tensor_free(X_pad);
    tensor_free(converted);

    TEST_PASSED;
}

void run_im2col_tests(void) {
    test_im2col();
}
