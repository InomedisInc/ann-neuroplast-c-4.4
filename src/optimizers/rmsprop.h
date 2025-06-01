#ifndef RMSPROP_H
#define RMSPROP_H

#include <stddef.h>
typedef struct {
    float *v;
    float lr, beta, epsilon;
    size_t size;
} RMSPropState;

RMSPropState *rmsprop_init(size_t size, float lr, float beta, float epsilon);
void rmsprop_update(RMSPropState *state, float *w, float *grad);
void rmsprop_free(RMSPropState *state);

#endif