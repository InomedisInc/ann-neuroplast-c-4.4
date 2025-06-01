#include "rmsprop.h"
#include <stdlib.h>
#include <math.h>

RMSPropState *rmsprop_init(size_t size, float lr, float beta, float epsilon) {
    RMSPropState *state = malloc(sizeof(RMSPropState));
    state->v = calloc(size, sizeof(float));
    state->lr = lr; state->beta = beta; state->epsilon = epsilon; state->size = size;
    return state;
}

void rmsprop_update(RMSPropState *state, float *w, float *grad) {
    for (size_t i = 0; i < state->size; i++) {
        state->v[i] = state->beta * state->v[i] + (1.0f - state->beta) * grad[i] * grad[i];
        w[i] -= state->lr * grad[i] / (sqrtf(state->v[i]) + state->epsilon);
    }
}

void rmsprop_free(RMSPropState *state) {
    free(state->v); free(state);
}