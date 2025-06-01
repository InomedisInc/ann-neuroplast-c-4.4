#ifndef RADAM_H
#define RADAM_H

#include <stddef.h>
typedef struct {
    float *m, *v;
    float beta1, beta2, epsilon, lr;
    size_t size;
    int t;
} RAdamState;

RAdamState *radam_init(size_t size, float lr, float beta1, float beta2, float epsilon);
void radam_update(RAdamState *state, float *w, float *grad);
void radam_free(RAdamState *state);

#endif