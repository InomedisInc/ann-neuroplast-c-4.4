#ifndef LION_H
#define LION_H

#include <stddef.h>
typedef struct {
    float *m;
    float beta1, beta2, lr;
    size_t size;
} LionState;

LionState *lion_init(size_t size, float lr, float beta1, float beta2);
void lion_update(LionState *state, float *w, float *grad);
void lion_free(LionState *state);

#endif