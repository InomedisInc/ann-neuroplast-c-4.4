#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct {
    size_t rows;
    size_t cols;
    float **data;
} Matrix;

// Création et suppression
Matrix *matrix_create(size_t rows, size_t cols);
void matrix_free(Matrix *mat);

// Opérations matricielles essentielles
void matrix_fill(Matrix *mat, float value);
void matrix_randomize(Matrix *mat, float min, float max);
Matrix *matrix_dot(Matrix *a, Matrix *b);
void matrix_add(Matrix *dest, Matrix *src);
void matrix_apply_function(Matrix *mat, float (*func)(float));

#endif /* MATRIX_H */