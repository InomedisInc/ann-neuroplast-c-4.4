#include "activation.h"
#include <string.h>
#include <math.h>

int get_activation_type(const char *name) {
    if (strcmp(name, "relu") == 0) return ACTIVATION_RELU;
    if (strcmp(name, "sigmoid") == 0) return ACTIVATION_SIGMOID;
    if (strcmp(name, "gelu") == 0) return ACTIVATION_GELU;
    if (strcmp(name, "neuroplast") == 0) return ACTIVATION_NEUROPLAST;
    if (strcmp(name, "leaky_relu") == 0) return ACTIVATION_LEAKY_RELU;
    if (strcmp(name, "elu") == 0) return ACTIVATION_ELU;
    if (strcmp(name, "mish") == 0) return ACTIVATION_MISH;
    if (strcmp(name, "swish") == 0) return ACTIVATION_SWISH;
    if (strcmp(name, "prelu") == 0) return ACTIVATION_PRELU;
    if (strcmp(name, "tanh") == 0) return ACTIVATION_TANH;
    if (strcmp(name, "linear") == 0) return ACTIVATION_LINEAR;
    return ACTIVATION_RELU; // Par défaut
}

float relu(float x) { return x > 0 ? x : 0; }
float leaky_relu(float x, float alpha) { return x > 0 ? x : alpha * x; }
float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
float gelu(float x) { return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x))); }
float elu(float x, float alpha) { return x > 0 ? x : alpha * (expf(x) - 1.0f); }
float mish(float x) { return x * tanhf(log1pf(expf(x))); }
float swish(float x) { return x * sigmoid(x); }
float prelu(float x, float alpha) { return x > 0 ? x : alpha * x; }
float neuroplast(float x, NeuroPlastParams *p) {
    // Utiliser les paramètres alpha, beta, gamma, delta définis dans neuroplast.h
    // alpha = slope, beta = shift, gamma = plateau_height, delta = plateau_width
    float slope = p->alpha;
    float shift = p->beta;
    float plateau_height = p->gamma;
    float plateau_width = p->delta;
    
    float sigmoid_part = 1.0f / (1.0f + expf(-slope * (x - shift)));
    float gaussian_plateau = plateau_height * expf(-((x - shift) * (x - shift)) / (plateau_width * plateau_width));
    return sigmoid_part * gaussian_plateau;
}