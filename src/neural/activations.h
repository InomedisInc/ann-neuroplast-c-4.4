#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

// Types d'activation
typedef enum {
    ACTIVATION_RELU = 0,
    ACTIVATION_SIGMOID = 1,
    ACTIVATION_GELU = 2,
    ACTIVATION_NEUROPLAST = 3,
    ACTIVATION_LEAKY_RELU = 4,
    ACTIVATION_ELU = 5,
    ACTIVATION_MISH = 6,
    ACTIVATION_SWISH = 7,
    ACTIVATION_PRELU = 8
} ActivationType;

// Fonctions d'activation
float relu(float x);
float sigmoid(float x);
float gelu(float x);
float neuroplast(float x, void *params);
float leaky_relu(float x, float alpha);
float elu(float x, float alpha);
float mish(float x);
float swish(float x);
float prelu(float x, float alpha);

#endif // ACTIVATIONS_H 