/*
 * test_tensor.c - Comprehensive tests for tensor operations
 */

#include "../src/tensor/tensor.h"
#include "../src/utils/utils.h"
#include "test_runner.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Creation and Destruction Tests
// =============================================================================

void test_tensor_create_3d() {
    Tensor *t = tensor_create3d(2, 3, 4);

    assert(t != NULL);
    assert(t->ndim == 3);
    assert(t->shape[0] == 2);
    assert(t->shape[1] == 3);
    assert(t->shape[2] == 4);
    assert(t->size == 24);
    assert(t->data != NULL);

    // Check strides (row-major order)
    assert(t->strides[0] == 12); // 3 * 4
    assert(t->strides[1] == 4);  // 4
    assert(t->strides[2] == 1);

    tensor_free(t);
    TEST_PASSED;
}

void test_tensor_create_4d() {
    Tensor *t = tensor_create4d(2, 3, 4, 5);

    assert(t != NULL);
    assert(t->ndim == 4);
    assert(t->shape[0] == 2);
    assert(t->shape[1] == 3);
    assert(t->shape[2] == 4);
    assert(t->shape[3] == 5);
    assert(t->size == 120);
    assert(t->data != NULL);

    // Check strides (row-major order)
    assert(t->strides[0] == 60); // 3 * 4 * 5
    assert(t->strides[1] == 20); // 4 * 5
    assert(t->strides[2] == 5);  // 5
    assert(t->strides[3] == 1);

    tensor_free(t);
    TEST_PASSED;
}

void test_tensor_zeros() {
    Tensor *t = tensor_create3d(2, 2, 2);

    assert(t != NULL);
    assert(t->ndim == 3);
    assert(t->size == 8);

    // Verify all elements are zero-initialized
    for (int i = 0; i < t->size; i++) {
        assert(float_equals(t->data[i], 0.0f));
    }

    tensor_free(t);
    TEST_PASSED;
}

void test_tensor_create_1d() {
    Tensor *t = tensor_create1d(10);

    assert(t != NULL);
    assert(t->ndim == 1);
    assert(t->shape[0] == 10);
    assert(t->size == 10);
    assert(t->strides[0] == 1);

    tensor_free(t);
    TEST_PASSED;
}

void test_tensor_create_2d() {
    Tensor *t = tensor_create2d(3, 4);

    assert(t != NULL);
    assert(t->ndim == 2);
    assert(t->shape[0] == 3);
    assert(t->shape[1] == 4);
    assert(t->size == 12);
    assert(t->strides[0] == 4);
    assert(t->strides[1] == 1);

    tensor_free(t);
    TEST_PASSED;
}

// =============================================================================
// 3D Accessor Tests
// =============================================================================

void test_tensor_get_set_3d() {
    Tensor *t = tensor_create3d(2, 3, 4);

    // Set some values
    tensor_set3d(t, 0, 0, 0, 1.0f);
    tensor_set3d(t, 0, 1, 2, 5.5f);
    tensor_set3d(t, 1, 2, 3, 10.0f);

    // Get and verify
    assert(float_equals(tensor_get3d(t, 0, 0, 0), 1.0f));
    assert(float_equals(tensor_get3d(t, 0, 1, 2), 5.5f));
    assert(float_equals(tensor_get3d(t, 1, 2, 3), 10.0f));

    tensor_free(t);
    TEST_PASSED;
}

void test_tensor_index3d() {
    Tensor *t = tensor_create3d(2, 3, 4);

    // Test that indexing is correct for row-major order
    // Index = i * strides[0] + j * strides[1] + k
    assert(tensor_index3d(t, 0, 0, 0) == 0);
    assert(tensor_index3d(t, 0, 0, 1) == 1);
    assert(tensor_index3d(t, 0, 1, 0) == 4);
    assert(tensor_index3d(t, 1, 0, 0) == 12);
    assert(tensor_index3d(t, 1, 2, 3) == 23); // Last element

    tensor_free(t);
    TEST_PASSED;
}

void test_tensor_3d_all_elements() {
    Tensor *t = tensor_create3d(2, 3, 4);

    // Set all elements with unique values
    float val = 0.0f;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                tensor_set3d(t, i, j, k, val);
                val += 1.0f;
            }
        }
    }

    // Verify all elements
    val = 0.0f;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                assert(float_equals(tensor_get3d(t, i, j, k), val));
                val += 1.0f;
            }
        }
    }

    tensor_free(t);
    TEST_PASSED;
}

// =============================================================================
// 4D Accessor Tests
// =============================================================================

void test_tensor_get_set_4d() {
    Tensor *t = tensor_create4d(2, 3, 4, 5);

    // Set some values
    tensor_set4d(t, 0, 0, 0, 0, 1.0f);
    tensor_set4d(t, 0, 1, 2, 3, 7.5f);
    tensor_set4d(t, 1, 2, 3, 4, 15.0f);

    // Get and verify
    assert(float_equals(tensor_get4d(t, 0, 0, 0, 0), 1.0f));
    assert(float_equals(tensor_get4d(t, 0, 1, 2, 3), 7.5f));
    assert(float_equals(tensor_get4d(t, 1, 2, 3, 4), 15.0f));

    tensor_free(t);
    TEST_PASSED;
}

void test_tensor_index4d() {
    Tensor *t = tensor_create4d(2, 3, 4, 5);

    // Test that indexing is correct for row-major order
    assert(tensor_index4d(t, 0, 0, 0, 0) == 0);
    assert(tensor_index4d(t, 0, 0, 0, 1) == 1);
    assert(tensor_index4d(t, 0, 0, 1, 0) == 5);
    assert(tensor_index4d(t, 0, 1, 0, 0) == 20);
    assert(tensor_index4d(t, 1, 0, 0, 0) == 60);
    assert(tensor_index4d(t, 1, 2, 3, 4) == 119); // Last element

    tensor_free(t);
    TEST_PASSED;
}

void test_tensor_4d_all_elements() {
    Tensor *t = tensor_create4d(2, 2, 2, 2);

    // Set all elements with unique values
    float val = 0.0f;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 2; l++) {
                    tensor_set4d(t, i, j, k, l, val);
                    val += 1.0f;
                }
            }
        }
    }

    // Verify all elements
    val = 0.0f;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 2; l++) {
                    assert(float_equals(tensor_get4d(t, i, j, k, l), val));
                    val += 1.0f;
                }
            }
        }
    }

    tensor_free(t);
    TEST_PASSED;
}

// =============================================================================
// Utility Function Tests
// =============================================================================

void test_tensor_fill() {
    Tensor *t = tensor_create3d(3, 4, 5);

    tensor_fill(t, 3.14f);

    for (int i = 0; i < t->size; i++) {
        assert(float_equals(t->data[i], 3.14f));
    }

    tensor_free(t);
    TEST_PASSED;
}

void test_tensor_fill_zero() {
    Tensor *t = tensor_create3d(2, 2, 2);

    // First fill with non-zero
    tensor_fill(t, 5.0f);

    // Then fill with zero
    tensor_fill(t, 0.0f);

    for (int i = 0; i < t->size; i++) {
        assert(float_equals(t->data[i], 0.0f));
    }

    tensor_free(t);
    TEST_PASSED;
}

void test_tensor_copy() {
    Tensor *src = tensor_create3d(2, 3, 4);

    // Fill source with values
    for (int i = 0; i < src->size; i++) {
        src->data[i] = (float)i;
    }

    Tensor *dest = tensor_create3d(2, 3, 4);
    tensor_copy(dest, src);

    // Verify copy
    for (int i = 0; i < src->size; i++) {
        assert(float_equals(dest->data[i], src->data[i]));
    }

    tensor_free(src);
    tensor_free(dest);
    TEST_PASSED;
}

void test_tensor_clone() {
    Tensor *t = tensor_create3d(2, 3, 4);

    // Fill with values
    for (int i = 0; i < t->size; i++) {
        t->data[i] = (float)i * 0.5f;
    }

    Tensor *clone = tensor_clone(t);

    // Verify clone has same properties
    assert(clone->ndim == t->ndim);
    assert(clone->size == t->size);
    for (int i = 0; i < t->ndim; i++) {
        assert(clone->shape[i] == t->shape[i]);
        assert(clone->strides[i] == t->strides[i]);
    }

    // Verify data is copied
    for (int i = 0; i < t->size; i++) {
        assert(float_equals(clone->data[i], t->data[i]));
    }

    // Verify it's a deep copy (modifying clone doesn't affect original)
    clone->data[0] = 999.0f;
    assert(!float_equals(t->data[0], clone->data[0]));

    tensor_free(t);
    tensor_free(clone);
    TEST_PASSED;
}

void test_tensor_print_shape() {
    Tensor *t = tensor_create3d(2, 3, 4);

    printf("  Testing tensor_print_shape (visual check): ");
    tensor_print_shape(t);
    printf("\n");

    tensor_free(t);
    TEST_PASSED;
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

void test_tensor_single_element() {
    Tensor *t = tensor_create3d(1, 1, 1);

    assert(t->size == 1);
    tensor_set3d(t, 0, 0, 0, 42.0f);
    assert(float_equals(tensor_get3d(t, 0, 0, 0), 42.0f));

    tensor_free(t);
    TEST_PASSED;
}

void test_tensor_large() {
    // Test with a reasonably large tensor
    Tensor *t = tensor_create3d(32, 28, 28);

    assert(t != NULL);
    assert(t->size == 32 * 28 * 28);

    // Set and get corners
    tensor_set3d(t, 0, 0, 0, 1.0f);
    tensor_set3d(t, 31, 27, 27, 2.0f);
    assert(float_equals(tensor_get3d(t, 0, 0, 0), 1.0f));
    assert(float_equals(tensor_get3d(t, 31, 27, 27), 2.0f));

    tensor_free(t);
    TEST_PASSED;
}

void test_tensor_copy_independence() {
    Tensor *src = tensor_create2d(2, 2);
    tensor_fill(src, 5.0f);

    Tensor *dest = tensor_create2d(2, 2);
    tensor_copy(dest, src);

    // Modify source after copy
    tensor_fill(src, 10.0f);

    // Destination should be unchanged
    for (int i = 0; i < dest->size; i++) {
        assert(float_equals(dest->data[i], 5.0f));
    }

    tensor_free(src);
    tensor_free(dest);
    TEST_PASSED;
}

// =============================================================================
// Test Runner
// =============================================================================

void run_tensor_tests(void) {
    printf("\n=== Tensor Creation and Destruction Tests ===\n");
    test_tensor_create_3d();
    test_tensor_create_4d();
    test_tensor_zeros();
    test_tensor_create_1d();
    test_tensor_create_2d();

    printf("\n=== Tensor 3D Accessor Tests ===\n");
    test_tensor_get_set_3d();
    test_tensor_index3d();
    test_tensor_3d_all_elements();

    printf("\n=== Tensor 4D Accessor Tests ===\n");
    test_tensor_get_set_4d();
    test_tensor_index4d();
    test_tensor_4d_all_elements();

    printf("\n=== Tensor Utility Function Tests ===\n");
    test_tensor_fill();
    test_tensor_fill_zero();
    test_tensor_copy();
    test_tensor_clone();
    test_tensor_print_shape();

    printf("\n=== Tensor Edge Cases Tests ===\n");
    test_tensor_single_element();
    test_tensor_large();
    test_tensor_copy_independence();

    printf("\n=== All Tensor Tests Passed! ===\n");
}
