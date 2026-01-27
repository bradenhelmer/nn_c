#!/usr/bin/env python3
"""
Simple test runner. Requires: uv pip install -e .
"""

from tests.test_tensor import (
    test_tensor_creation,
    test_tensor_from_bytes,
    test_tensor_from_list,
    test_tensor_indexing,
    test_tensor_roundtrip,
)

if __name__ == "__main__":
    print("Running tensor tests...")

    test_tensor_creation()
    print("✓ test_tensor_creation")

    test_tensor_from_list()
    print("✓ test_tensor_from_list")

    test_tensor_from_bytes()
    print("✓ test_tensor_from_bytes")

    test_tensor_roundtrip()
    print("✓ test_tensor_roundtrip")

    test_tensor_indexing()
    print("✓ test_tensor_indexing")

    print("\nAll tests passed! ✨")
