#ifndef ADAM_H
#define ADAM_H

#include <stddef.h>

typedef struct {
    float *m, *v;
    float beta1, beta2, epsilon, lr;
    size_t size;
    int t;
    
    // Nouveaux champs pour les améliorations
    float initial_lr;        // Taux d'apprentissage initial
    int warmup_steps;        // Étapes de warmup
    float decay_factor;      // Facteur de décroissance
    float grad_clip_norm;    // Norme pour gradient clipping
} AdamState;

AdamState *adam_init(size_t size, float lr, float beta1, float beta2, float epsilon);
void adam_update(AdamState *state, float *w, float *grad);
void adam_free(AdamState *state);

#endif