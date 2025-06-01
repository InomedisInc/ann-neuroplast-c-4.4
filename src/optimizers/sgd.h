#ifndef SGD_H
#define SGD_H

#include <stddef.h>
typedef struct {
    float lr;
    size_t size;
} SGDState;

SGDState *sgd_init(size_t size, float lr);
void sgd_update(SGDState *state, float *w, float *grad);
void sgd_free(SGDState *state);

#endif