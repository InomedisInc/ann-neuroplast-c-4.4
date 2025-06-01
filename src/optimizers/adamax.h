#ifndef ADAMAX_H
#define ADAMAX_H

#include <stddef.h>
typedef struct {
    float *m, *u;
    float beta1, beta2, lr, epsilon;
    size_t size;
    int t;
} AdamaxState;

AdamaxState *adamax_init(size_t size, float lr, float beta1, float beta2, float epsilon);
void adamax_update(AdamaxState *state, float *w, float *grad);
void adamax_free(AdamaxState *state);

#endif