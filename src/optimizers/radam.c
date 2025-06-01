#include "radam.h"
#include <stdlib.h>
#include <math.h>

RAdamState *radam_init(size_t size, float lr, float beta1, float beta2, float epsilon) {
    RAdamState *state = malloc(sizeof(RAdamState));
    state->m = calloc(size, sizeof(float));
    state->v = calloc(size, sizeof(float));
    state->beta1 = beta1; state->beta2 = beta2; state->epsilon = epsilon;
    state->lr = lr; state->size = size; state->t = 0;
    return state;
}

void radam_update(RAdamState *state, float *w, float *grad) {
    state->t++;
    float b1 = state->beta1, b2 = state->beta2, lr = state->lr, eps = state->epsilon;
    float b1t = powf(b1, state->t), b2t = powf(b2, state->t);

    float rho_inf = 2.0f / (1.0f - b2) - 1.0f;
    float rho_t = rho_inf - 2.0f * state->t * b2t / (1.0f - b2t);

    for (size_t i = 0; i < state->size; i++) {
        state->m[i] = b1 * state->m[i] + (1.0f - b1) * grad[i];
        state->v[i] = b2 * state->v[i] + (1.0f - b2) * grad[i] * grad[i];

        float m_hat = state->m[i] / (1.0f - b1t);
        if (rho_t > 4) {
            float v_hat = state->v[i] / (1.0f - b2t);
            float r = sqrtf(((rho_t - 4) * (rho_t - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho_t));
            w[i] -= lr * r * m_hat / (sqrtf(v_hat) + eps);
        } else {
            w[i] -= lr * m_hat;
        }
    }
}

void radam_free(RAdamState *state) {
    free(state->m); free(state->v); free(state);
}