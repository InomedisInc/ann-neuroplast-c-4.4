#include "dataset.h"
#include <stdlib.h>
#include <stdbool.h>

Dataset *dataset_create(size_t capacity, size_t input_cols, size_t output_cols) {
    Dataset *d = malloc(sizeof(Dataset));
    if (!d) return NULL;

    d->inputs = malloc(capacity * sizeof(float*));
    if (!d->inputs) {
        free(d);
        return NULL;
    }

    d->outputs = malloc(capacity * sizeof(float*));
    if (!d->outputs) {
        free(d->inputs);
        free(d);
        return NULL;
    }

    d->num_samples = 0;
    d->input_cols = input_cols;
    d->output_cols = output_cols;

    bool success = true;
    for (size_t i = 0; i < capacity && success; ++i) {
        d->inputs[i] = malloc(input_cols * sizeof(float));
        if (!d->inputs[i]) {
            success = false;
            break;
        }
        d->outputs[i] = malloc(output_cols * sizeof(float));
        if (!d->outputs[i]) {
            free(d->inputs[i]);
            success = false;
            break;
        }
    }

    if (!success) {
        for (size_t i = 0; i < capacity; ++i) {
            free(d->inputs[i]);
            free(d->outputs[i]);
        }
        free(d->inputs);
        free(d->outputs);
        free(d);
        return NULL;
    }

    return d;
}

bool dataset_resize(Dataset *d, size_t new_capacity) {
    if (!d || new_capacity < d->num_samples) return false;

    // Réalloue les tableaux de pointeurs
    float **new_inputs = realloc(d->inputs, new_capacity * sizeof(float*));
    if (!new_inputs) return false;
    d->inputs = new_inputs;

    float **new_outputs = realloc(d->outputs, new_capacity * sizeof(float*));
    if (!new_outputs) return false;
    d->outputs = new_outputs;

    // Alloue la mémoire pour les nouvelles lignes
    bool success = true;
    for (size_t i = d->num_samples; i < new_capacity && success; ++i) {
        d->inputs[i] = malloc(d->input_cols * sizeof(float));
        if (!d->inputs[i]) {
            success = false;
            break;
        }
        d->outputs[i] = malloc(d->output_cols * sizeof(float));
        if (!d->outputs[i]) {
            free(d->inputs[i]);
            success = false;
            break;
        }
    }

    // En cas d'échec, libère la mémoire allouée
    if (!success) {
        for (size_t i = d->num_samples; i < new_capacity; ++i) {
            free(d->inputs[i]);
            free(d->outputs[i]);
        }
        return false;
    }

    return true;
}

void dataset_free(Dataset *d) {
    if (!d) return;

    if (d->inputs) {
        for (size_t i = 0; i < d->num_samples; ++i) {
            free(d->inputs[i]);
        }
        free(d->inputs);
    }

    if (d->outputs) {
        for (size_t i = 0; i < d->num_samples; ++i) {
            free(d->outputs[i]);
        }
        free(d->outputs);
    }

    free(d);
}