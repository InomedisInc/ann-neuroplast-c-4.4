#ifndef NADAM_H
#define NADAM_H

#include <stddef.h>
typedef struct {
    float *m, *v;
    float beta1, beta2, epsilon, lr;
    size_t size;
    int t;
} NadamState;

NadamState *nadam_init(size_t size, float lr, float beta1, float beta2, float epsilon);
void nadam_update(NadamState *state, float *w, float *grad);
void nadam_free(NadamState *state);

#endif