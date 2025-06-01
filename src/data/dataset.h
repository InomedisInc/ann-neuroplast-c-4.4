#ifndef DATASET_H
#define DATASET_H

#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    float **inputs;
    float **outputs;
    size_t num_samples;
    size_t input_cols;
    size_t output_cols;
} Dataset;

// Crée un nouveau dataset avec la capacité et les dimensions spécifiées
Dataset *dataset_create(size_t capacity, size_t input_cols, size_t output_cols);

// Redimensionne un dataset existant à une nouvelle capacité
// Retourne true si le redimensionnement a réussi, false sinon
bool dataset_resize(Dataset *dataset, size_t new_capacity);

// Libère la mémoire d'un dataset
void dataset_free(Dataset *dataset);

#endif