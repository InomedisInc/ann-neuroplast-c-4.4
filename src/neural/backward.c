#include "backward.h"

void backward_pass(NeuralNetwork *net, float *input, float *target, float learning_rate) {
    network_backward(net, input, target, learning_rate, 1.0f);
}
