#include "adamax.h"
#include <stdlib.h>
#include <math.h>

AdamaxState *adamax_init(size_t size, float lr, float beta1, float beta2, float epsilon) {
    AdamaxState *state = malloc(sizeof(AdamaxState));
    state->m = calloc(size, sizeof(float));
    state->u = calloc(size, sizeof(float));
    state->beta1 = beta1; state->beta2 = beta2; state->lr = lr; state->epsilon = epsilon; state->size = size; state->t = 0;
    return state;
}

void adamax_update(AdamaxState *state, float *w, float *grad) {
    state->t++;
    float b1 = state->beta1, lr = state->lr, eps = state->epsilon;
    float b1t = powf(b1, state->t);

    for (size_t i = 0; i < state->size; i++) {
        state->m[i] = b1 * state->m[i] + (1.0f - b1) * grad[i];
        state->u[i] = fmaxf(state->beta2 * state->u[i], fabsf(grad[i]));
        w[i] -= (lr / (1.0f - b1t)) * (state->m[i] / (state->u[i] + eps));
    }
}

void adamax_free(AdamaxState *state) {
    free(state->m); free(state->u); free(state);
}