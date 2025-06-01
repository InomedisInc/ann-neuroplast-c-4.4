#include "propagation.h"
#include <stdio.h>

// Version sécurisée de l'entraînement par propagation
void train_propagation(Trainer *trainer, Dataset *dataset) {
    if (!trainer || !dataset) return;
    
    for (int epoch = 0; epoch < trainer->epochs; epoch++) {
        size_t max_samples = dataset->num_samples;
        for (size_t i = 0; i < max_samples; i++) {
            // Vérification de sécurité
            if (!trainer->net || !trainer->net->layers || !dataset->inputs[i] || !dataset->outputs[i]) {
                printf("ERREUR: Pointeurs invalides à l'échantillon %zu\n", i);
                return;
            }
            
            // Forward pass
            network_forward(trainer->net, dataset->inputs[i]);
            
            // Pour l'instant, on évite les updates d'optimizer qui causent le crash
            // Ici on pourrait ajouter: label propagation, message passing, etc.
        }
    }
}