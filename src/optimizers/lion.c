#include "lion.h"
#include <stdlib.h>
#include <math.h>

// Source : https://arxiv.org/abs/2302.06675
LionState *lion_init(size_t size, float lr, float beta1, float beta2) {
    LionState *state = malloc(sizeof(LionState));
    state->m = calloc(size, sizeof(float));
    state->lr = lr; state->beta1 = beta1; state->beta2 = beta2; state->size = size;
    return state;
}

void lion_update(LionState *state, float *w, float *grad) {
    for (size_t i = 0; i < state->size; i++) {
        float update = state->beta1 * state->m[i] + (1 - state->beta1) * grad[i];
        w[i] -= state->lr * copysignf(1.0f, update);
        state->m[i] = state->beta2 * state->m[i] + (1 - state->beta2) * grad[i];
    }
}

void lion_free(LionState *state) {
    free(state->m); free(state);
}