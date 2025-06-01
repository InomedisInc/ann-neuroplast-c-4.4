#include "neuroplast.h"
#include "activation.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

// Structure pour l'optimisation bayésienne des paramètres NeuroPlast
typedef struct {
    float *param_history;     // Historique des paramètres testés
    float *performance_history; // Historique des performances correspondantes
    int history_size;         // Taille de l'historique
    int current_size;         // Taille actuelle utilisée
} BayesianOptimizer;

// Structure pour l'optimisation par essaim de particules
typedef struct {
    float position[4];        // alpha, beta, gamma, delta
    float velocity[4];        // Vitesses pour chaque paramètre
    float best_position[4];   // Meilleure position personnelle
    float best_fitness;       // Meilleure performance personnelle
} Particle;

typedef struct {
    Particle *particles;      // Tableau de particules
    int num_particles;        // Nombre de particules
    float global_best_position[4]; // Meilleure position globale
    float global_best_fitness;      // Meilleure performance globale
    float inertia;            // Coefficient d'inertie
    float cognitive;          // Coefficient cognitif
    float social;             // Coefficient social
} SwarmOptimizer;

// Initialisation corrigée avec les bons noms de paramètres
void neuroplast_init_params(NeuroPlastParams *params, float alpha, float beta, float gamma, float delta) {
    params->alpha = alpha;
    params->beta = beta;
    params->gamma = gamma;
    params->delta = delta;
}

// Calcul de la dérivée de la fonction neuroplast
static float neuroplast_derivative(float x, NeuroPlastParams *params) {
    float slope = params->alpha;
    float shift = params->beta;
    float plateau_height = params->gamma;
    float plateau_width = params->delta;
    
    // Dérivée de la partie sigmoïde
    float sigmoid_val = 1.0f / (1.0f + expf(-slope * (x - shift)));
    float sigmoid_deriv = slope * sigmoid_val * (1.0f - sigmoid_val);
    
    // Dérivée de la partie gaussienne
    float gaussian_val = plateau_height * expf(-((x - shift) * (x - shift)) / (plateau_width * plateau_width));
    float gaussian_deriv = gaussian_val * (-2.0f * (x - shift)) / (plateau_width * plateau_width);
    
    // Dérivée totale (règle du produit)
    return sigmoid_deriv * gaussian_val + sigmoid_val * gaussian_deriv;
}

// Fonction d'évaluation pour l'optimisation des paramètres
static float evaluate_neuroplast_params(NeuroPlastParams *params, float *x, float *y, int n) {
    float total_error = 0.0f;
    int valid_samples = 0;
    
    for (int i = 0; i < n; i++) {
        float predicted = neuroplast(x[i], params);
        float error = (y[i] - predicted) * (y[i] - predicted); // MSE
        
        // Ajouter une pénalité pour les paramètres extrêmes
        float param_penalty = 0.0f;
        if (params->alpha < 0.1f || params->alpha > 10.0f) param_penalty += 0.1f;
        if (params->beta < -5.0f || params->beta > 5.0f) param_penalty += 0.1f;
        if (params->gamma < 0.01f || params->gamma > 2.0f) param_penalty += 0.1f;
        if (params->delta < 0.1f || params->delta > 10.0f) param_penalty += 0.1f;
        
        total_error += error + param_penalty;
        valid_samples++;
    }
    
    if (valid_samples == 0) return 1e6f; // Pénalité élevée si aucun échantillon valide
    
    return total_error / valid_samples;
}

// Optimisation bayésienne simplifiée
static void bayesian_optimize_params(NeuroPlastParams *params, float *x, float *y, int n) {
    const int max_iterations = 20;
    const int num_candidates = 10;
    
    float best_fitness = evaluate_neuroplast_params(params, x, y, n);
    NeuroPlastParams best_params = *params;
    
    // Échantillonnage initial aléatoire
    for (int iter = 0; iter < max_iterations; iter++) {
        for (int cand = 0; cand < num_candidates; cand++) {
            NeuroPlastParams candidate = *params;
            
            // Perturbation adaptative basée sur l'itération
            float perturbation_scale = 0.5f * expf(-0.1f * iter);
            
            candidate.alpha += ((float)rand() / RAND_MAX - 0.5f) * 2.0f * perturbation_scale;
            candidate.beta += ((float)rand() / RAND_MAX - 0.5f) * 2.0f * perturbation_scale;
            candidate.gamma += ((float)rand() / RAND_MAX - 0.5f) * 0.5f * perturbation_scale;
            candidate.delta += ((float)rand() / RAND_MAX - 0.5f) * 2.0f * perturbation_scale;
            
            // Contraintes sur les paramètres
            candidate.alpha = fmaxf(0.1f, fminf(10.0f, candidate.alpha));
            candidate.beta = fmaxf(-5.0f, fminf(5.0f, candidate.beta));
            candidate.gamma = fmaxf(0.01f, fminf(2.0f, candidate.gamma));
            candidate.delta = fmaxf(0.1f, fminf(10.0f, candidate.delta));
            
            float fitness = evaluate_neuroplast_params(&candidate, x, y, n);
            
            if (fitness < best_fitness) {
                best_fitness = fitness;
                best_params = candidate;
            }
        }
        
        // Mise à jour graduelle vers les meilleurs paramètres
        float alpha = 0.8f; // Facteur d'apprentissage
        params->alpha = alpha * best_params.alpha + (1.0f - alpha) * params->alpha;
        params->beta = alpha * best_params.beta + (1.0f - alpha) * params->beta;
        params->gamma = alpha * best_params.gamma + (1.0f - alpha) * params->gamma;
        params->delta = alpha * best_params.delta + (1.0f - alpha) * params->delta;
    }
    
    *params = best_params;
}

// Optimisation par essaim de particules (PSO)
static void pso_optimize_params(NeuroPlastParams *params, float *x, float *y, int n) {
    const int num_particles = 15;
    const int max_iterations = 30;
    
    SwarmOptimizer swarm;
    swarm.particles = malloc(num_particles * sizeof(Particle));
    swarm.num_particles = num_particles;
    swarm.inertia = 0.9f;
    swarm.cognitive = 2.0f;
    swarm.social = 2.0f;
    swarm.global_best_fitness = 1e6f;
    
    // Initialisation des particules
    for (int i = 0; i < num_particles; i++) {
        Particle *p = &swarm.particles[i];
        
        // Position aléatoire
        p->position[0] = 0.5f + ((float)rand() / RAND_MAX) * 3.0f; // alpha
        p->position[1] = -1.0f + ((float)rand() / RAND_MAX) * 2.0f; // beta
        p->position[2] = 0.1f + ((float)rand() / RAND_MAX) * 0.8f; // gamma
        p->position[3] = 1.0f + ((float)rand() / RAND_MAX) * 4.0f; // delta
        
        // Vitesse initiale
        for (int j = 0; j < 4; j++) {
            p->velocity[j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            p->best_position[j] = p->position[j];
        }
        
        // Évaluation initiale
        NeuroPlastParams test_params = {p->position[0], p->position[1], p->position[2], p->position[3]};
        p->best_fitness = evaluate_neuroplast_params(&test_params, x, y, n);
        
        // Mise à jour du meilleur global
        if (p->best_fitness < swarm.global_best_fitness) {
            swarm.global_best_fitness = p->best_fitness;
            for (int j = 0; j < 4; j++) {
                swarm.global_best_position[j] = p->position[j];
            }
        }
    }
    
    // Optimisation PSO
    for (int iter = 0; iter < max_iterations; iter++) {
        // Décroissance de l'inertie
        swarm.inertia = 0.9f - 0.4f * iter / max_iterations;
        
        for (int i = 0; i < num_particles; i++) {
            Particle *p = &swarm.particles[i];
            
            // Mise à jour de la vitesse
            for (int j = 0; j < 4; j++) {
                float r1 = (float)rand() / RAND_MAX;
                float r2 = (float)rand() / RAND_MAX;
                
                p->velocity[j] = swarm.inertia * p->velocity[j] +
                               swarm.cognitive * r1 * (p->best_position[j] - p->position[j]) +
                               swarm.social * r2 * (swarm.global_best_position[j] - p->position[j]);
            }
            
            // Mise à jour de la position
            for (int j = 0; j < 4; j++) {
                p->position[j] += p->velocity[j];
            }
            
            // Contraintes
            p->position[0] = fmaxf(0.1f, fminf(10.0f, p->position[0])); // alpha
            p->position[1] = fmaxf(-5.0f, fminf(5.0f, p->position[1])); // beta
            p->position[2] = fmaxf(0.01f, fminf(2.0f, p->position[2])); // gamma
            p->position[3] = fmaxf(0.1f, fminf(10.0f, p->position[3])); // delta
            
            // Évaluation
            NeuroPlastParams test_params = {p->position[0], p->position[1], p->position[2], p->position[3]};
            float fitness = evaluate_neuroplast_params(&test_params, x, y, n);
            
            // Mise à jour du meilleur personnel
            if (fitness < p->best_fitness) {
                p->best_fitness = fitness;
                for (int j = 0; j < 4; j++) {
                    p->best_position[j] = p->position[j];
                }
                
                // Mise à jour du meilleur global
                if (fitness < swarm.global_best_fitness) {
                    swarm.global_best_fitness = fitness;
                    for (int j = 0; j < 4; j++) {
                        swarm.global_best_position[j] = p->position[j];
                    }
                }
            }
        }
    }
    
    // Application des meilleurs paramètres trouvés
    params->alpha = swarm.global_best_position[0];
    params->beta = swarm.global_best_position[1];
    params->gamma = swarm.global_best_position[2];
    params->delta = swarm.global_best_position[3];
    
    free(swarm.particles);
}

// Optimisation adaptative par gradient
static void gradient_optimize_params(NeuroPlastParams *params, float *x, float *y, int n) {
    const int max_iterations = 50;
    const float learning_rate = 0.01f;
    const float epsilon = 1e-5f;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        float gradients[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // alpha, beta, gamma, delta
        
        // Calcul des gradients par différences finies
        for (int i = 0; i < n; i++) {
            float predicted = neuroplast(x[i], params);
            float error = predicted - y[i];
            
            // Gradient par rapport à alpha
            NeuroPlastParams temp = *params;
            temp.alpha += epsilon;
            float pred_alpha = neuroplast(x[i], &temp);
            gradients[0] += error * (pred_alpha - predicted) / epsilon;
            
            // Gradient par rapport à beta
            temp = *params;
            temp.beta += epsilon;
            float pred_beta = neuroplast(x[i], &temp);
            gradients[1] += error * (pred_beta - predicted) / epsilon;
            
            // Gradient par rapport à gamma
            temp = *params;
            temp.gamma += epsilon;
            float pred_gamma = neuroplast(x[i], &temp);
            gradients[2] += error * (pred_gamma - predicted) / epsilon;
            
            // Gradient par rapport à delta
            temp = *params;
            temp.delta += epsilon;
            float pred_delta = neuroplast(x[i], &temp);
            gradients[3] += error * (pred_delta - predicted) / epsilon;
        }
        
        // Normalisation des gradients
        for (int j = 0; j < 4; j++) {
            gradients[j] /= n;
        }
        
        // Mise à jour des paramètres avec clipping
        params->alpha -= learning_rate * gradients[0];
        params->beta -= learning_rate * gradients[1];
        params->gamma -= learning_rate * gradients[2];
        params->delta -= learning_rate * gradients[3];
        
        // Contraintes
        params->alpha = fmaxf(0.1f, fminf(10.0f, params->alpha));
        params->beta = fmaxf(-5.0f, fminf(5.0f, params->beta));
        params->gamma = fmaxf(0.01f, fminf(2.0f, params->gamma));
        params->delta = fmaxf(0.1f, fminf(10.0f, params->delta));
    }
}

// Fonction principale d'optimisation avec méthodes multiples
void neuroplast_optimize_params(NeuroPlastParams *params, float *x, float *y, int n) {
    if (!params || !x || !y || n <= 0) return;
    
    static int method_counter = 0;
    method_counter++;
    
    // Alterner entre différentes méthodes d'optimisation
    switch (method_counter % 3) {
        case 0:
            printf("Optimisation NeuroPlast: méthode bayésienne\n");
            bayesian_optimize_params(params, x, y, n);
            break;
        case 1:
            printf("Optimisation NeuroPlast: essaim de particules (PSO)\n");
            pso_optimize_params(params, x, y, n);
            break;
        case 2:
            printf("Optimisation NeuroPlast: descente de gradient\n");
            gradient_optimize_params(params, x, y, n);
            break;
    }
    
    // Validation finale des paramètres
    float final_error = evaluate_neuroplast_params(params, x, y, n);
    printf("Paramètres NeuroPlast optimisés: α=%.4f, β=%.4f, γ=%.4f, δ=%.4f (erreur=%.6f)\n",
           params->alpha, params->beta, params->gamma, params->delta, final_error);
}

// Fonction pour obtenir la dérivée de neuroplast (utile pour la rétropropagation)
float neuroplast_get_derivative(float x, NeuroPlastParams *params) {
    return neuroplast_derivative(x, params);
}