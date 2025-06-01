#include "matrix.h"
#include "memory.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// Créer une matrice vide
Matrix *matrix_create(size_t rows, size_t cols) {
    Matrix *mat = mem_alloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = mem_alloc(rows * sizeof(float *));
    for (size_t i = 0; i < rows; i++)
        mat->data[i] = mem_calloc(cols, sizeof(float));
    return mat;
}

// Libérer une matrice
void matrix_free(Matrix *mat) {
    if (mat) {
        for (size_t i = 0; i < mat->rows; i++)
            mem_free(mat->data[i]);
        mem_free(mat->data);
        mem_free(mat);
    }
}

// Remplir une matrice avec une valeur spécifique
void matrix_fill(Matrix *mat, float value) {
    for (size_t i = 0; i < mat->rows; i++)
        for (size_t j = 0; j < mat->cols; j++)
            mat->data[i][j] = value;
}

// Remplir une matrice avec des valeurs aléatoires
void matrix_randomize(Matrix *mat, float min, float max) {
    for (size_t i = 0; i < mat->rows; i++)
        for (size_t j = 0; j < mat->cols; j++)
            mat->data[i][j] = min + (max - min) * ((float)rand() / RAND_MAX);
}

// Produit matriciel
Matrix *matrix_dot(Matrix *a, Matrix *b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "Erreur : dimensions incompatibles pour le produit matriciel.\n");
        exit(EXIT_FAILURE);
    }

    Matrix *result = matrix_create(a->rows, b->cols);
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a->cols; k++)
                sum += a->data[i][k] * b->data[k][j];
            result->data[i][j] = sum;
        }
    }
    return result;
}

// Addition matricielle
void matrix_add(Matrix *dest, Matrix *src) {
    if (dest->rows != src->rows || dest->cols != src->cols) {
        fprintf(stderr, "Erreur : dimensions incompatibles pour l'addition matricielle.\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < dest->rows; i++)
        for (size_t j = 0; j < dest->cols; j++)
            dest->data[i][j] += src->data[i][j];
}

// Appliquer une fonction à chaque élément de la matrice
void matrix_apply_function(Matrix *mat, float (*func)(float)) {
    for (size_t i = 0; i < mat->rows; i++)
        for (size_t j = 0; j < mat->cols; j++)
            mat->data[i][j] = func(mat->data[i][j]);
}