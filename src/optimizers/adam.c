#include "adam.h"
#include <stdlib.h>
#include <math.h>

AdamState *adam_init(size_t size, float lr, float beta1, float beta2, float epsilon) {
    AdamState *state = malloc(sizeof(AdamState));
    if (!state) return NULL;
    
    state->m = calloc(size, sizeof(float));
    state->v = calloc(size, sizeof(float));
    
    if (!state->m || !state->v) {
        free(state->m);
        free(state->v);
        free(state);
        return NULL;
    }
    
    state->beta1 = beta1;
    state->beta2 = beta2;
    state->epsilon = epsilon;
    state->lr = lr;
    state->size = size;
    state->t = 0;
    
    // Nouveaux paramètres pour l'amélioration
    state->initial_lr = lr;
    state->warmup_steps = 1000;
    state->decay_factor = 0.9999f;
    state->grad_clip_norm = 5.0f;
    
    return state;
}

void adam_update(AdamState *state, float *w, float *grad) {
    if (!state || !w || !grad) return;
    
    state->t++;
    
    // Calcul de la norme du gradient pour clipping
    float grad_norm = 0.0f;
    for (size_t i = 0; i < state->size; i++) {
        grad_norm += grad[i] * grad[i];
    }
    grad_norm = sqrtf(grad_norm);
    
    // Gradient clipping adaptatif
    float clip_factor = 1.0f;
    if (grad_norm > state->grad_clip_norm) {
        clip_factor = state->grad_clip_norm / grad_norm;
    }
    
    // Taux d'apprentissage adaptatif avec warmup et decay
    float adaptive_lr = state->lr;
    
    // Phase de warmup (augmentation progressive)
    if (state->t <= state->warmup_steps) {
        adaptive_lr = state->initial_lr * (float)state->t / state->warmup_steps;
    } else {
        // Décroissance exponentielle après warmup
        adaptive_lr = state->initial_lr * powf(state->decay_factor, state->t - state->warmup_steps);
    }
    
    // Limites pour éviter des taux d'apprentissage trop extrêmes
    adaptive_lr = fmaxf(adaptive_lr, state->initial_lr * 0.01f); // Au moins 1% du taux initial
    adaptive_lr = fminf(adaptive_lr, state->initial_lr * 2.0f);  // Au plus 200% du taux initial
    
    float b1 = state->beta1, b2 = state->beta2, eps = state->epsilon;
    
    // Correction de biais améliorée (rectifiée)
    float b1t = powf(b1, state->t);
    float b2t = powf(b2, state->t);
    
    // RAdam correction - Variance corrigée
    float rho_inf = 2.0f / (1.0f - b2) - 1.0f;
    float rho_t = rho_inf - 2.0f * state->t * b2t / (1.0f - b2t);
    
    for (size_t i = 0; i < state->size; i++) {
        // Application du gradient clipping
        float clipped_grad = grad[i] * clip_factor;
        
        // Mise à jour des moments avec amélioration numérique
        state->m[i] = b1 * state->m[i] + (1.0f - b1) * clipped_grad;
        state->v[i] = b2 * state->v[i] + (1.0f - b2) * clipped_grad * clipped_grad;
        
        // Correction de biais
        float m_hat = state->m[i] / (1.0f - b1t);
        
        float update;
        
        if (rho_t > 4.0f) {
            // RAdam update avec variance corrigée
            float v_hat = state->v[i] / (1.0f - b2t);
            float rect = sqrtf((rho_t - 4.0f) * (rho_t - 2.0f) * rho_inf / 
                              ((rho_inf - 4.0f) * (rho_inf - 2.0f) * rho_t));
            update = adaptive_lr * rect * m_hat / (sqrtf(v_hat) + eps);
        } else {
            // SGD avec momentum quand la variance n'est pas bien définie
            update = adaptive_lr * m_hat;
        }
        
        // Application de la mise à jour avec vérification de stabilité
        float weight_update = update;
        
        // Anti-explosion des poids
        if (fabsf(weight_update) > 1.0f) {
            weight_update = copysignf(1.0f, weight_update);
        }
        
        w[i] -= weight_update;
        
        // Contrainte pour éviter les poids extrêmes
        w[i] = fmaxf(-10.0f, fminf(10.0f, w[i]));
    }
    
    // Mise à jour adaptative des hyperparamètres basée sur la performance
    if (state->t % 100 == 0) {
        // Ajustement adaptatif de beta2 basé sur la variance des gradients
        if (grad_norm > 1.0f) {
            state->beta2 = fminf(0.999f, state->beta2 + 0.001f); // Augmenter pour plus de stabilité
        } else if (grad_norm < 0.1f) {
            state->beta2 = fmaxf(0.9f, state->beta2 - 0.001f); // Diminuer pour plus de réactivité
        }
    }
}

void adam_free(AdamState *state) {
    if (!state) return;
    free(state->m); 
    free(state->v); 
    free(state);
}