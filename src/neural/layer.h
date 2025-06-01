#ifndef LAYER_H
#define LAYER_H

#include <stddef.h>
#include "neuroplast.h"

typedef struct {
    size_t input_size;
    size_t output_size;
    int activation_type;
    float **weights;
    float *biases;
    float *outputs;
    float *deltas;
    NeuroPlastParams *np_params;
} Layer;

Layer *layer_create(size_t input_size, size_t output_size, int activation_type);
void layer_free(Layer *layer);
void layer_forward(Layer *layer, float *input);
void layer_backward(Layer *layer, float *input, float *delta, float learning_rate);

#endif