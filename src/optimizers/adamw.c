#include "adamw.h"
#include <stdlib.h>
#include <math.h>

AdamWState *adamw_init(size_t size, float lr, float beta1, float beta2, float epsilon, float weight_decay) {
    AdamWState *state = malloc(sizeof(AdamWState));
    state->m = calloc(size, sizeof(float));
    state->v = calloc(size, sizeof(float));
    state->beta1 = beta1; state->beta2 = beta2; state->epsilon = epsilon;
    state->lr = lr; state->weight_decay = weight_decay; state->size = size; state->t = 0;
    return state;
}

void adamw_update(AdamWState *state, float *w, float *grad) {
    state->t++;
    float b1 = state->beta1, b2 = state->beta2, lr = state->lr, eps = state->epsilon, wd = state->weight_decay;
    float b1t = powf(b1, state->t), b2t = powf(b2, state->t);
    for (size_t i = 0; i < state->size; i++) {
        state->m[i] = b1 * state->m[i] + (1.0f - b1) * grad[i];
        state->v[i] = b2 * state->v[i] + (1.0f - b2) * grad[i] * grad[i];
        float m_hat = state->m[i] / (1.0f - b1t);
        float v_hat = state->v[i] / (1.0f - b2t);
        w[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * w[i]);
    }
}

void adamw_free(AdamWState *state) {
    free(state->m); free(state->v); free(state);
}