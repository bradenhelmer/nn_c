/*
 * layer.c
 *
 * Generic layer implementations
 */
#include "layer.h"
#include <stdlib.h>

Layer *layer_create(LayerType type, void *layer) {
    Layer *L = (Layer *)(malloc(sizeof(Layer)));
    L->type = type;
    L->layer = layer;
    return L;
}

void layer_free(Layer *layer) {
    switch (layer->type) {
    case LAYER_LINEAR:
        linear_layer_free((LinearLayer *)layer->layer);
    }
}

void layer_forward(Layer *layer, Tensor *input);

void layer_backward(Layer *layer, Tensor *upstream_grad);

void layer_init_weights(Layer *layer);
