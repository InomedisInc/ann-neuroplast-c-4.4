#include "sgd.h"
#include <stdlib.h>

SGDState *sgd_init(size_t size, float lr) {
    SGDState *state = malloc(sizeof(SGDState));
    state->lr = lr;
    state->size = size;
    return state;
}

void sgd_update(SGDState *state, float *w, float *grad) {
    for (size_t i = 0; i < state->size; i++)
        w[i] -= state->lr * grad[i];
}

void sgd_free(SGDState *state) {
    free(state);
}