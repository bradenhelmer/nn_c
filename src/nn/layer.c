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
        break;
    case LAYER_ACTIVATION:
        activation_layer_free((ActivationLayer *)layer->layer);
        break;
    case LAYER_CONV_2D:
        conv2d_layer_free((Conv2DLayer *)layer->layer);
        break;
    case LAYER_MAX_POOL:
        maxpool_layer_free((MaxPoolLayer *)layer->layer);
        break;
    case LAYER_FLATTEN:
        flatten_layer_free((FlattenLayer *)layer->layer);
        break;
    }
    free(layer);
}

Tensor *layer_forward(Layer *layer, const Tensor *input) {
    switch (layer->type) {
    case LAYER_LINEAR:
        return linear_layer_forward((LinearLayer *)layer->layer, input);
    case LAYER_ACTIVATION:
        return activation_layer_forward((ActivationLayer *)layer->layer, input);
    case LAYER_CONV_2D:
        return conv_layer_forward_im2col((Conv2DLayer *)layer->layer, input);
    case LAYER_MAX_POOL:
        return maxpool_layer_forward((MaxPoolLayer *)layer->layer, input);
    case LAYER_FLATTEN:
        return flatten_layer_forward((FlattenLayer *)layer->layer, input);
    }
}

Tensor *layer_backward(Layer *layer, const Tensor *upstream_grad) {
    switch (layer->type) {
    case LAYER_LINEAR:
        return linear_layer_backward((LinearLayer *)layer->layer, upstream_grad);
    case LAYER_ACTIVATION:
        return activation_layer_backward((ActivationLayer *)layer->layer, upstream_grad);
    case LAYER_CONV_2D:
        return conv_layer_backward_im2col((Conv2DLayer *)layer->layer, upstream_grad);
    case LAYER_MAX_POOL:
        return maxpool_layer_backward((MaxPoolLayer *)layer->layer, upstream_grad);
    case LAYER_FLATTEN:
        return flatten_layer_backward((FlattenLayer *)layer->layer, upstream_grad);
    }
    return NULL;
}

Tensor *layer_get_output(Layer *layer) {
    switch (layer->type) {
    case LAYER_LINEAR:
        return ((LinearLayer *)layer->layer)->output;
    case LAYER_ACTIVATION:
        return ((ActivationLayer *)layer->layer)->output;
    case LAYER_CONV_2D:
        return ((Conv2DLayer *)layer->layer)->output;
    case LAYER_MAX_POOL:
        return ((MaxPoolLayer *)layer->layer)->output;
    case LAYER_FLATTEN:
        return ((FlattenLayer *)layer->layer)->output;
    }
    return NULL;
}

void layer_zero_gradients(Layer *layer) {
    switch (layer->type) {
    case LAYER_LINEAR: {
        LinearLayer *ll = (LinearLayer *)layer->layer;
        tensor_fill(ll->grad_weights, 0.f);
        tensor_fill(ll->grad_biases, 0.f);
        break;
    }
    case LAYER_CONV_2D: {
        Conv2DLayer *cl = (Conv2DLayer *)layer->layer;
        tensor_fill(cl->grad_weights, 0.f);
        tensor_fill(cl->grad_biases, 0.f);
        break;
    }
    case LAYER_ACTIVATION:
    case LAYER_MAX_POOL:
    case LAYER_FLATTEN:
        // No gradients to zero
        break;
    }
}

void layer_update_weights(Layer *layer, float learning_rate) {
    switch (layer->type) {
    case LAYER_LINEAR: {
        LinearLayer *ll = (LinearLayer *)layer->layer;
        for (int i = 0; i < ll->weights->size; ++i) {
            ll->weights->data[i] -= learning_rate * ll->grad_weights->data[i];
        }
        for (int i = 0; i < ll->biases->size; ++i) {
            ll->biases->data[i] -= learning_rate * ll->grad_biases->data[i];
        }
        break;
    }
    case LAYER_CONV_2D: {
        Conv2DLayer *cl = (Conv2DLayer *)layer->layer;
        for (int i = 0; i < cl->weights->size; ++i) {
            cl->weights->data[i] -= learning_rate * cl->grad_weights->data[i];
        }
        for (int i = 0; i < cl->biases->size; ++i) {
            cl->biases->data[i] -= learning_rate * cl->grad_biases->data[i];
        }
        break;
    }
    case LAYER_ACTIVATION:
    case LAYER_MAX_POOL:
    case LAYER_FLATTEN:
        // No weights to update
        break;
    }
}

void layer_scale_gradients(Layer *layer, float scale) {
    switch (layer->type) {
    case LAYER_LINEAR: {
        LinearLayer *ll = (LinearLayer *)layer->layer;
        tensor_scale(ll->grad_weights, ll->grad_weights, scale);
        tensor_scale(ll->grad_biases, ll->grad_biases, scale);
        break;
    }
    case LAYER_CONV_2D: {
        Conv2DLayer *cl = (Conv2DLayer *)layer->layer;
        tensor_scale(cl->grad_weights, cl->grad_weights, scale);
        tensor_scale(cl->grad_biases, cl->grad_biases, scale);
        break;
    }
    case LAYER_ACTIVATION:
    case LAYER_MAX_POOL:
    case LAYER_FLATTEN:
        // No gradients to scale
        break;
    }
}

void layer_add_l2_gradient(Layer *layer, float lambda) {
    switch (layer->type) {
    case LAYER_LINEAR: {
        LinearLayer *ll = (LinearLayer *)layer->layer;
        for (int i = 0; i < ll->weights->size; i++) {
            ll->grad_weights->data[i] += lambda * ll->weights->data[i];
        }
        break;
    }
    case LAYER_CONV_2D: {
        Conv2DLayer *cl = (Conv2DLayer *)layer->layer;
        for (int i = 0; i < cl->weights->size; i++) {
            cl->grad_weights->data[i] += lambda * cl->weights->data[i];
        }
        break;
    }
    case LAYER_ACTIVATION:
    case LAYER_MAX_POOL:
    case LAYER_FLATTEN:
        // No weights for L2 regularization
        break;
    }
}

LayerParameters layer_get_parameters(Layer *layer) {
    LayerParameters params = {.pairs = NULL, .num_pairs = 0};
    switch (layer->type) {
    case LAYER_LINEAR: {
        LinearLayer *ll = (LinearLayer *)layer->layer;
        params.num_pairs = 2;
        params.pairs = (ParameterPair *)malloc(sizeof(ParameterPair) * 2);
        params.pairs[0].param = ll->weights;
        params.pairs[0].grad_param = ll->grad_weights;
        params.pairs[1].param = ll->biases;
        params.pairs[1].grad_param = ll->grad_biases;
        break;
    }
    case LAYER_CONV_2D: {
        Conv2DLayer *cl = (Conv2DLayer *)layer->layer;
        params.num_pairs = 2;
        params.pairs = (ParameterPair *)malloc(sizeof(ParameterPair) * 2);
        params.pairs[0].param = cl->weights;
        params.pairs[0].grad_param = cl->grad_weights;
        params.pairs[1].param = cl->biases;
        params.pairs[1].grad_param = cl->grad_biases;
        break;
    }
    case LAYER_ACTIVATION:
    case LAYER_MAX_POOL:
    case LAYER_FLATTEN:
        params.num_pairs = 0;
        params.pairs = NULL;
        break;
    }
    return params;
}

void layer_parameters_free(LayerParameters *params) {
    if (params->pairs != NULL) {
        free(params->pairs);
        params->pairs = NULL;
    }
    params->num_pairs = 0;
}
