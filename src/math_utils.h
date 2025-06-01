#ifndef MATH_UTILS_H
#define MATH_UTILS_H

// Fonctions mathématiques optimisées
float fast_exp(float x);
float fast_sigmoid(float x);
float fast_tanh(float x);
float fast_relu(float x);
float fast_leaky_relu(float x, float alpha);
float fast_gelu(float x);
float fast_elu(float x, float alpha);
float fast_swish(float x);
float fast_mish(float x);

#endif /* MATH_UTILS_H */