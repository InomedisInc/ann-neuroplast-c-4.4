#include "forward.h"

void forward_pass(NeuralNetwork *net, float *input) {
    network_forward(net, input);
}