#include "split.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

void split_dataset(const Dataset *src, float ratio, Dataset **train, Dataset **test) {
    size_t train_size = (size_t)(src->num_samples * ratio);
    size_t test_size = src->num_samples - train_size;
    
    // Création des datasets
    *train = dataset_create(train_size, src->input_cols, src->output_cols);
    *test = dataset_create(test_size, src->input_cols, src->output_cols);
    
    // Copie des données d'entraînement
    for (size_t i = 0; i < train_size; i++) {
        for (size_t j = 0; j < src->input_cols; j++) {
            (*train)->inputs[i][j] = src->inputs[i][j];
        }
        for (size_t j = 0; j < src->output_cols; j++) {
            (*train)->outputs[i][j] = src->outputs[i][j];
        }
    }
    
    // Copie des données de test
    for (size_t i = 0; i < test_size; i++) {
        for (size_t j = 0; j < src->input_cols; j++) {
            (*test)->inputs[i][j] = src->inputs[train_size + i][j];
        }
        for (size_t j = 0; j < src->output_cols; j++) {
            (*test)->outputs[i][j] = src->outputs[train_size + i][j];
        }
    }
    
    (*train)->num_samples = train_size;
    (*test)->num_samples = test_size;
}