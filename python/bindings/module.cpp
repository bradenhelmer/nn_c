/**
 * Python bindings for nn_core library using pybind11.
 *
 * Key patterns demonstrated:
 * 1. Wrapping C structs with custom deleters
 * 2. Exposing properties and methods
 * 3. Static factory methods (from_bytes, from_list)
 * 4. Buffer protocol for zero-copy (optional optimization)
 */

#include "activations/activations.h"
#include "nn/loss.h"
#include "tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector <-> Python list conversion

// Your C headers
extern "C" {
#include "core/tensor_internal.h"
}

namespace py = pybind11;

// =============================================================================
// Tensor Bindings
// =============================================================================

/**
 * Custom destructor wrapper.
 * pybind11 wants to call `delete ptr` but your C code uses tensor_free().
 * This tells pybind11 how to properly clean up.
 */
struct TensorDeleter {
    void operator()(Tensor *t) const {
        if (t) {
            tensor_free(t);
        }
    }
};

// Unique pointer type with custom deleter
using TensorPtr = std::unique_ptr<Tensor, TensorDeleter>;

/**
 * Helper: Convert C shape array to Python tuple
 */
py::tuple shape_to_tuple(const Tensor *t) {
    py::tuple shape(t->ndim);
    for (int i = 0; i < t->ndim; i++) {
        shape[i] = t->shape[i];
    }
    return shape;
}

/**
 * Helper: Compute total size from shape vector
 */
int compute_size(const std::vector<int> &shape) {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    return size;
}

void bind_tensor(py::module_ &m) {
    py::class_<Tensor, TensorPtr>(m, "Tensor")
        // Properties (read-only)
        .def_property_readonly("shape", &shape_to_tuple)
        .def_property_readonly("ndim", [](const Tensor *t) { return t->ndim; })
        .def_property_readonly("size", [](const Tensor *t) { return t->size; })

        // Constructor: Tensor([3, 28, 28]) creates zeroed tensor
        .def(py::init([](std::vector<int> shape) {
                 // Your tensor_create function signature may vary
                 // Adjust this to match your actual API
                 Tensor *t = tensor_create(shape.size(), shape.data());
                 return TensorPtr(t);
             }),
             py::arg("shape"))

        // Static factory: Tensor.from_bytes(data, shape)
        .def_static(
            "from_bytes",
            [](py::bytes data, std::vector<int> shape) {
                std::string_view view = data;
                int expected_size = compute_size(shape) * sizeof(float);

                if (static_cast<int>(view.size()) != expected_size) {
                    throw std::invalid_argument("Byte length mismatch: expected " +
                                                std::to_string(expected_size) + ", got " +
                                                std::to_string(view.size()));
                }

                Tensor *t = tensor_create(shape.size(), shape.data());
                std::memcpy(t->data, view.data(), view.size());
                return TensorPtr(t);
            },
            py::arg("data"), py::arg("shape"))

        // Static factory: Tensor.from_list(data, shape)
        .def_static(
            "from_list",
            [](std::vector<float> data, std::vector<int> shape) {
                int expected_size = compute_size(shape);

                if (static_cast<int>(data.size()) != expected_size) {
                    throw std::invalid_argument("Data length mismatch: expected " +
                                                std::to_string(expected_size) + ", got " +
                                                std::to_string(data.size()));
                }

                Tensor *t = tensor_create(shape.size(), shape.data());
                std::memcpy(t->data, data.data(), data.size() * sizeof(float));
                return TensorPtr(t);
            },
            py::arg("data"), py::arg("shape"))

        .def_static("ones_like", [](const Tensor *t) { return TensorPtr(tensor_ones_like(t)); })

        // Static factory: Tensor.random(shape, min, max)
        .def_static(
            "random",
            [](std::vector<int> shape, float min, float max) {
                Tensor *t = tensor_random(shape.size(), shape.data(), min, max);
                return TensorPtr(t);
            },
            py::arg("shape"), py::arg("min"), py::arg("max"))

        // Export: tensor.to_bytes()
        .def("to_bytes",
             [](const Tensor *t) {
                 return py::bytes(reinterpret_cast<const char *>(t->data), t->size * sizeof(float));
             })

        // Export: tensor.to_list()
        .def("to_list",
             [](const Tensor *t) { return std::vector<float>(t->data, t->data + t->size); })

        // Element access: tensor[i] or tensor[i, j, k]
        .def("__getitem__",
             [](const Tensor *t, py::object idx) {
                 // Single integer index (flat)
                 if (py::isinstance<py::int_>(idx)) {
                     int i = idx.cast<int>();
                     if (i < 0 || i >= t->size) {
                         throw std::out_of_range("Index out of bounds");
                     }
                     return t->data[i];
                 }

                 // Tuple of indices
                 auto indices = idx.cast<std::vector<int>>();
                 if (static_cast<int>(indices.size()) != t->ndim) {
                     throw std::invalid_argument("Wrong number of indices");
                 }

                 // Compute flat index from multidimensional indices
                 int flat_idx = 0;
                 for (int d = 0; d < t->ndim; d++) {
                     if (indices[d] < 0 || indices[d] >= t->shape[d]) {
                         throw std::out_of_range("Index out of bounds for dimension " +
                                                 std::to_string(d));
                     }
                     flat_idx += indices[d] * t->strides[d];
                 }
                 return t->data[flat_idx];
             })

        // Element assignment: tensor[i] = val or tensor[i, j, k] = val
        .def("__setitem__",
             [](Tensor *t, py::object idx, float val) {
                 if (py::isinstance<py::int_>(idx)) {
                     int i = idx.cast<int>();
                     if (i < 0 || i >= t->size) {
                         throw std::out_of_range("Index out of bounds");
                     }
                     t->data[i] = val;
                     return;
                 }

                 auto indices = idx.cast<std::vector<int>>();
                 if (static_cast<int>(indices.size()) != t->ndim) {
                     throw std::invalid_argument("Wrong number of indices");
                 }

                 int flat_idx = 0;
                 for (int d = 0; d < t->ndim; d++) {
                     if (indices[d] < 0 || indices[d] >= t->shape[d]) {
                         throw std::out_of_range("Index out of bounds for dimension " +
                                                 std::to_string(d));
                     }
                     flat_idx += indices[d] * t->strides[d];
                 }
                 t->data[flat_idx] = val;
             })
        .def("__len__", [](const Tensor *t) { return t->size; })

        // Nice repr for debugging
        .def("__repr__",
             [](const Tensor *t) {
                 std::string s = "Tensor(shape=[";
                 for (int i = 0; i < t->ndim; i++) {
                     if (i > 0) {
                         s += ", ";
                     }
                     s += std::to_string(t->shape[i]);
                 }
                 s += "], size=" + std::to_string(t->size) + ")";
                 return s;
             })

        // ==============================================================================
        // Tensor Ops
        // ==============================================================================

        .def("add",
             [](const Tensor *t, const Tensor *other) {
                 if (t->ndim == 2 && other->ndim == 1 && other->shape[0] == t->shape[1]) {
                     Tensor *dest = tensor_create(t->ndim, t->shape);
                     tensor_add_broadcast(dest, t, other);
                     return TensorPtr(dest);
                 }
                 Tensor *dest = tensor_create(t->ndim, t->shape);
                 tensor_add(dest, t, other);
                 return TensorPtr(dest);
             })
        .def("subtract",
             [](const Tensor *t, const Tensor *other) {
                 Tensor *dest = tensor_create(t->ndim, t->shape);
                 tensor_subtract(dest, t, other);
                 return TensorPtr(dest);
             })
        .def("elementwise_mul",
             [](const Tensor *t, const Tensor *other) {
                 Tensor *dest = tensor_create(t->ndim, t->shape);
                 tensor_elementwise_mul(dest, t, other);
                 return TensorPtr(dest);
             })
        .def("matmul",
             [](const Tensor *t, const Tensor *other) {
                 Tensor *dest = tensor_create(2, (int[]){t->shape[0], other->shape[1]});
                 tensor_matmul(dest, t, other);
                 return TensorPtr(dest);
             })

        .def("relu",
             [](const Tensor *t) {
                 Tensor *dest = tensor_create(t->ndim, t->shape);
                 tensor_relu(dest, t);
                 return TensorPtr(dest);
             })

        .def("transpose2d", [](const Tensor *t) { return TensorPtr(tensor_transpose2d(t)); })
        .def("scale",
             [](const Tensor *t, float scalar) {
                 Tensor *dest = tensor_create(t->ndim, t->shape);
                 tensor_scale(dest, t, scalar);
                 return TensorPtr(dest);
             })
        .def("clone", [](const Tensor *t) { return TensorPtr(tensor_clone(t)); })
        .def("fill", [](Tensor *t, float val) { tensor_fill(t, val); })
        .def("sum", [](const Tensor *t) { return tensor_sum(t); })
        .def(
            "sum_axis",
            [](const Tensor *t, int axis) {
                if (axis < 0 || axis >= t->ndim) {
                    throw std::invalid_argument("axis out of range");
                }

                std::vector<int> dest_shape;
                for (int i = 0; i < t->ndim; i++) {
                    if (i != axis) {
                        dest_shape.push_back(t->shape[i]);
                    }
                }

                if (dest_shape.empty()) {
                    dest_shape.push_back(1);
                }

                Tensor *dest = tensor_create(dest_shape.size(), dest_shape.data());
                tensor_sum_axis(dest, t, axis);
                return TensorPtr(dest);
            },
            py::arg("axis"))
        .def("argmax", [](const Tensor *t) { return tensor_argmax(t); })

        // ==============================================================================
        // Backward helpers
        // ==============================================================================
        .def(
            "relu_backward",
            [](const Tensor *grad_output, const Tensor *relu_output) {
                Tensor *mask = tensor_create(relu_output->ndim, relu_output->shape);
                tensor_relu_derivative(mask, relu_output);

                Tensor *dest = tensor_create(grad_output->ndim, grad_output->shape);
                tensor_elementwise_mul(dest, grad_output, mask);

                tensor_free(mask);
                return TensorPtr(dest);
            },
            py::arg("relu_output"))

        // ==============================================================================
        // Loss Methods
        // ==============================================================================
        .def(
            "softmax_cross_entropy",
            [](const Tensor *logits, const Tensor *target) {
                return tensor_softmax_cross_entropy(logits, target);
            },
            py::arg("target"))

        .def(
            "softmax_cross_entropy_backward",
            [](const Tensor *logits, const Tensor *target) {
                Tensor *result = tensor_clone(logits);
                tensor_softmax_cross_entropy_derivative(result, logits, target);
                return TensorPtr(result);
            },
            py::arg("target"));
}

// =============================================================================
// Module Definition
// =============================================================================

PYBIND11_MODULE(_nn_core, m) {
    m.doc() = "Neural network core library";

    bind_tensor(m);

    // Add more bindings as you implement them:
    // bind_layer(m);
    // bind_neural_net(m);
    // bind_gpu_tensor(m);
}
