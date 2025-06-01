#include "math_utils.h"
#include <math.h>

// Fonction exponentielle rapide (approximation)
float fast_exp(float x) {
    return expf(x);
}

// Fonction sigmoid rapide
float fast_sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

// Fonction tangente hyperbolique rapide
float fast_tanh(float x) {
    return tanhf(x);
}

// Fonction ReLU rapide
float fast_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Fonction Leaky ReLU rapide
float fast_leaky_relu(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}

// Fonction GELU rapide (approximation simplifiée)
float fast_gelu(float x) {
    return 0.5f * x * (1.0f + fast_tanh(0.79788456f * (x + 0.044715f * x * x * x)));
}

// Fonction ELU rapide
float fast_elu(float x, float alpha) {
    return x > 0.0f ? x : alpha * (fast_exp(x) - 1.0f);
}

// Fonction Swish rapide
float fast_swish(float x) {
    return x * fast_sigmoid(x);
}

// Fonction Mish rapide
float fast_mish(float x) {
    return x * fast_tanh(logf(1.0f + fast_exp(x)));
}