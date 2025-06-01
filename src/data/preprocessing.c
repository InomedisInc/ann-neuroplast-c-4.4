#include "preprocessing.h"
#include <stdlib.h>
#include <time.h>

void normalize_dataset(Dataset *d, float new_min, float new_max) {
    for (size_t col = 0; col < d->input_cols; ++col) {
        float min = d->inputs[0][col], max = d->inputs[0][col];
        for (size_t i = 1; i < d->num_samples; ++i) {
            if (d->inputs[i][col] < min) min = d->inputs[i][col];
            if (d->inputs[i][col] > max) max = d->inputs[i][col];
        }
        float range = max - min;
        if (range == 0) range = 1.0f;
        for (size_t i = 0; i < d->num_samples; ++i)
            d->inputs[i][col] = new_min + (d->inputs[i][col] - min) * (new_max - new_min) / range;
    }
}

void shuffle_dataset(Dataset *d) {
    srand((unsigned int)time(NULL));
    for (size_t i = d->num_samples - 1; i > 0; --i) {
        size_t j = rand() % (i + 1);
        float *tmp_in = d->inputs[i]; d->inputs[i] = d->inputs[j]; d->inputs[j] = tmp_in;
        float *tmp_out = d->outputs[i]; d->outputs[i] = d->outputs[j]; d->outputs[j] = tmp_out;
    }
}