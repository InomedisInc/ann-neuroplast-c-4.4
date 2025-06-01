#ifndef ADAMW_H
#define ADAMW_H

#include <stddef.h>
typedef struct {
    float *m, *v;
    float beta1, beta2, epsilon, lr, weight_decay;
    size_t size;
    int t;
} AdamWState;

AdamWState *adamw_init(size_t size, float lr, float beta1, float beta2, float epsilon, float weight_decay);
void adamw_update(AdamWState *state, float *w, float *grad);
void adamw_free(AdamWState *state);

#endif