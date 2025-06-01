#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "neuroplast.h"  // Inclure pour NeuroPlastParams

// Types d'activation
#define ACTIVATION_RELU 0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_GELU 2
#define ACTIVATION_NEUROPLAST 3
#define ACTIVATION_LEAKY_RELU 4
#define ACTIVATION_ELU 5
#define ACTIVATION_MISH 6
#define ACTIVATION_SWISH 7
#define ACTIVATION_PRELU 8
#define ACTIVATION_TANH 9
#define ACTIVATION_LINEAR 10
#define ACTIVATION_UNKNOWN -1

// Typedef pour le type d'activation
typedef int activation_type_t;

// Fonctions d'activation
float relu(float x);
float leaky_relu(float x, float alpha);
float sigmoid(float x);
float gelu(float x);
float elu(float x, float alpha);
float mish(float x);
float swish(float x);
float prelu(float x, float alpha);
float neuroplast(float x, NeuroPlastParams *params);

// Conversion nom -> type
int get_activation_type(const char *name);

#endif