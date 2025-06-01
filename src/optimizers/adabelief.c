#include "adabelief.h"
#include <stdlib.h>
#include <math.h>

AdaBeliefState *adabelief_init(size_t size, float lr, float beta1, float beta2, float epsilon) {
    AdaBeliefState *state = malloc(sizeof(AdaBeliefState));
    state->m = calloc(size, sizeof(float));
    state->s = calloc(size, sizeof(float));
    state->beta1 = beta1; state->beta2 = beta2; state->epsilon = epsilon; state->lr = lr; state->size = size; state->t = 0;
    return state;
}

void adabelief_update(AdaBeliefState *state, float *w, float *grad) {
    state->t++;
    float b1 = state->beta1, b2 = state->beta2, lr = state->lr, eps = state->epsilon;
    float b1t = powf(b1, state->t), b2t = powf(b2, state->t);
    for (size_t i = 0; i < state->size; i++) {
        float g = grad[i];
        state->m[i] = b1 * state->m[i] + (1.0f - b1) * g;
        float diff = g - state->m[i];
        state->s[i] = b2 * state->s[i] + (1.0f - b2) * diff * diff;
        float m_hat = state->m[i] / (1.0f - b1t);
        float s_hat = state->s[i] / (1.0f - b2t);
        w[i] -= lr * m_hat / (sqrtf(s_hat) + eps);
    }
}

void adabelief_free(AdaBeliefState *state) {
    free(state->m); free(state->s); free(state);
}