#ifndef BACKWARD_H
#define BACKWARD_H

#include "network.h"

void backward_pass(NeuralNetwork *net, float *input, float *target, float learning_rate);

#endif