#include "nadam.h"
#include <stdlib.h>
#include <math.h>

// Initialisation des états NAdam
NadamState *nadam_init(size_t size, float lr, float beta1, float beta2, float epsilon) {
    NadamState *state = malloc(sizeof(NadamState));
    state->m = calloc(size, sizeof(float));
    state->v = calloc(size, sizeof(float));
    state->beta1 = beta1;
    state->beta2 = beta2;
    state->epsilon = epsilon;
    state->lr = lr;
    state->size = size;
    state->t = 0;
    return state;
}

void nadam_update(NadamState *state, float *w, float *grad) {
    state->t += 1;
    float b1 = state->beta1;
    float b2 = state->beta2;
    float lr = state->lr;
    float eps = state->epsilon;
    float b1t = powf(b1, state->t);
    float b2t = powf(b2, state->t);

    for (size_t i = 0; i < state->size; i++) {
        // Mise à jour des moments
        state->m[i] = b1 * state->m[i] + (1.0f - b1) * grad[i];
        state->v[i] = b2 * state->v[i] + (1.0f - b2) * grad[i] * grad[i];

        // Moments biais-corrigés
        float m_hat = state->m[i] / (1.0f - b1t);
        float v_hat = state->v[i] / (1.0f - b2t);

        // Correction Nesterov (NAdam)
        float m_nesterov = b1 * m_hat + (1.0f - b1) * grad[i] / (1.0f - b1t);

        // Mise à jour du poids
        w[i] -= lr * m_nesterov / (sqrtf(v_hat) + eps);
    }
}

void nadam_free(NadamState *state) {
    if (state) {
        free(state->m);
        free(state->v);
        free(state);
    }
}