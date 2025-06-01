#include "standard.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../progress_bar.h"

// Version d'entraînement standard qui fait vraiment de l'entraînement
void train_standard(Trainer *trainer, Dataset *dataset) {
    if (!trainer || !dataset || !trainer->net) {
        printf("ERREUR: Paramètres invalides dans train_standard\n");
        return;
    }
    
    // Calcul des poids de classe pour le déséquilibre sur TOUT le dataset
    size_t class_0_count = 0, class_1_count = 0;
    for (size_t i = 0; i < dataset->num_samples; i++) {
        if (dataset->outputs[i][0] < 0.5f) {
            class_0_count++;
        } else {
            class_1_count++;
        }
    }
    
    // Créer des listes d'indices pour chaque classe
    size_t *class_0_indices = malloc(class_0_count * sizeof(size_t));
    size_t *class_1_indices = malloc(class_1_count * sizeof(size_t));
    
    if (!class_0_indices || !class_1_indices) {
        printf("ERREUR: Impossible d'allouer la mémoire pour les indices de classe\n");
        return;
    }
    
    size_t c0_idx = 0, c1_idx = 0;
    for (size_t i = 0; i < dataset->num_samples; i++) {
        if (dataset->outputs[i][0] < 0.5f) {
            class_0_indices[c0_idx++] = i;
        } else {
            class_1_indices[c1_idx++] = i;
        }
    }
    
    // Stratégie de sur-échantillonnage : équilibrer les classes AVEC LIMITATION
    size_t target_samples_per_class = (class_0_count < 2500) ? class_0_count : 2500; // Max 2500 par classe
    size_t total_balanced_samples = target_samples_per_class * 2;
    
    printf("INFO: Équilibrage dataset - %zu échantillons par classe (%zu total)\n", 
           target_samples_per_class, total_balanced_samples);
    
    // Créer un dataset équilibré avec vérification de sécurité
    size_t *balanced_indices = malloc(total_balanced_samples * sizeof(size_t));
    if (!balanced_indices) {
        printf("ERREUR: Impossible d'allouer la mémoire pour le dataset équilibré (%zu échantillons)\n", 
               total_balanced_samples);
        free(class_0_indices);
        free(class_1_indices);
        return;
    }
    
    // Remplir avec des échantillons de classe 0 (sous-échantillonnage)
    for (size_t i = 0; i < target_samples_per_class; i++) {
        balanced_indices[i] = class_0_indices[i % class_0_count];
    }
    
    // Remplir avec des échantillons de classe 1 (sur-échantillonnage)
    for (size_t i = 0; i < target_samples_per_class; i++) {
        balanced_indices[target_samples_per_class + i] = class_1_indices[i % class_1_count];
    }
    
    // Mélanger le dataset équilibré
    for (size_t i = total_balanced_samples - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t temp = balanced_indices[i];
        balanced_indices[i] = balanced_indices[j];
        balanced_indices[j] = temp;
    }
    
    // Poids de classe équilibrés
    float class_0_weight = 1.0f;
    float class_1_weight = 1.0f;
    
    // Augmentation du nombre d'époques pour un meilleur apprentissage
    int max_epochs = (trainer->epochs < 50) ? 50 : trainer->epochs;
    
    float best_loss = INFINITY;
    int patience_counter = 0;
    int max_patience = 15;
    
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        // Learning rate decay plus agressif
        float current_lr = trainer->learning_rate * (1.0f / (1.0f + 0.02f * epoch));
        
        for (size_t i = 0; i < total_balanced_samples; ++i) {
            // Utiliser l'indice équilibré avec vérification de sécurité
            size_t idx = balanced_indices[i];
            
            // Vérifications de sécurité étendues
            if (idx >= dataset->num_samples) {
                printf("ERREUR: Indice %zu hors limites (dataset: %zu échantillons)\n", 
                       idx, dataset->num_samples);
                continue;
            }
            
            if (!trainer->net || !trainer->net->layers || !dataset->inputs[idx] || !dataset->outputs[idx]) {
                printf("ERREUR: Pointeurs invalides à l'échantillon %zu\n", idx);
                continue;
            }
            
            // Vérifier que les pointeurs de données ne sont pas NULL
            if (!dataset->inputs[idx] || !dataset->outputs[idx]) {
                printf("ERREUR: Données NULL à l'échantillon %zu\n", idx);
                continue;
            }
            
            // Forward pass
            network_forward(trainer->net, dataset->inputs[idx]);
            
            // Calcul de la perte et de la précision
            float *output = network_output(trainer->net);
            if (output) {
                float target = dataset->outputs[idx][0];
                float prediction = output[0];
                
                // Binary cross-entropy loss avec clipping pour stabilité
                prediction = fmaxf(1e-7f, fminf(1.0f - 1e-7f, prediction));
                float loss = -(target * logf(prediction) + (1 - target) * logf(1 - prediction));
                epoch_loss += loss;
                
                // Déterminer le poids de classe (équilibré maintenant)
                float class_weight = (target > 0.5f) ? class_1_weight : class_0_weight;
                
                // Backward pass avec pondération de classe
                network_backward(trainer->net, dataset->inputs[idx], dataset->outputs[idx], current_lr, class_weight);
            }
        }
        
        // Calcul des métriques de l'époque
        epoch_loss /= total_balanced_samples;
        
        // Early stopping amélioré
        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
        }
        
        if (patience_counter >= max_patience && epoch > 20) {
            break;
        }
        
        // Stop si la perte devient très faible
        if (epoch_loss < 0.05f && epoch > 15) {
            break;
        }
    }
    
    free(class_0_indices);
    free(class_1_indices);
    free(balanced_indices);
}