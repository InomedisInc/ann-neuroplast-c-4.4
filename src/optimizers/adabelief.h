#ifndef ADABELIEF_H
#define ADABELIEF_H

#include <stddef.h>
typedef struct {
    float *m, *s;
    float beta1, beta2, epsilon, lr;
    size_t size;
    int t;
} AdaBeliefState;

AdaBeliefState *adabelief_init(size_t size, float lr, float beta1, float beta2, float epsilon);
void adabelief_update(AdaBeliefState *state, float *w, float *grad);
void adabelief_free(AdaBeliefState *state);

#endif