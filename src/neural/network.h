#ifndef NETWORK_H
#define NETWORK_H

#include <stddef.h>
#include "layer.h"

typedef struct {
    size_t num_layers;
    Layer **layers;
} NeuralNetwork;

NeuralNetwork *network_create(size_t n_layers, const size_t *layer_sizes, const char **activations);

void network_free(NeuralNetwork *net);
void network_forward(NeuralNetwork *net, float *input);
void network_backward(NeuralNetwork *net, float *input, float *target, float learning_rate, float class_weight);
float *network_output(NeuralNetwork *net);

#endif