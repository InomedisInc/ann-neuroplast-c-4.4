#include "adaptive.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../progress_bar.h"

// Méthode d'entraînement adaptatif améliorée
void train_adaptive(Trainer *trainer, Dataset *dataset) {
    if (!trainer || !dataset || !trainer->net) {
        printf("ERREUR: Paramètres invalides dans train_adaptive\n");
        return;
    }
    
    // Préparer les variables adaptatives
    float base_lr = trainer->learning_rate;
    float lr_decay_factor = 0.95f;
    float patience_threshold = 0.001f;
    int patience_counter = 0;
    int max_patience = 10;
    
    float best_loss = INFINITY;
    float *velocity = calloc(dataset->num_samples, sizeof(float));
    
    if (!velocity) {
        printf("ERREUR: Impossible d'allouer la mémoire pour la vélocité\n");
        return;
    }
    
    // Nombre d'époques adaptatif - commence plus haut
    int max_epochs = (trainer->epochs < 30) ? 30 : trainer->epochs;
    
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        // Learning rate adaptatif
        float epoch_lr = base_lr * powf(lr_decay_factor, epoch);
        if (patience_counter > 3) {
            epoch_lr *= 0.5f; // Réduction plus agressive si pas d'amélioration
        }
        
        for (size_t i = 0; i < dataset->num_samples; ++i) {
            // Vérification de sécurité
            if (!trainer->net || !trainer->net->layers || !dataset->inputs[i] || !dataset->outputs[i]) {
                printf("ERREUR: Pointeurs invalides à l'échantillon %zu\n", i);
                return;
            }
            
            // Forward pass
            network_forward(trainer->net, dataset->inputs[i]);
            
            // Calcul de la perte et de la précision
            float *output = network_output(trainer->net);
            if (output) {
                float target = dataset->outputs[i][0];
                float prediction = output[0];
                
                // Binary cross-entropy loss
                prediction = fmaxf(1e-7f, fminf(1.0f - 1e-7f, prediction));
                float loss = -(target * logf(prediction) + (1 - target) * logf(1 - prediction));
                epoch_loss += loss;
                
                // Backward pass
                network_backward(trainer->net, dataset->inputs[i], dataset->outputs[i], epoch_lr, 1.0f);
            }
        }
        
        // Calcul des métriques de l'époque
        epoch_loss /= dataset->num_samples;
        
        // Early stopping adaptatif
        if (epoch_loss < best_loss - patience_threshold) {
            best_loss = epoch_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
        }
        
        if (patience_counter >= max_patience && epoch > 10) {
            break;
        }
        
        // Convergence check
        if (epoch_loss < 0.02f && epoch > 8) {
            break;
        }
    }
    
    free(velocity);
}