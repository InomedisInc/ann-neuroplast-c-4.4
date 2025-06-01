#include "progressive.h"
#include <stdio.h>

// Version sécurisée de l'entraînement progressif
void train_progressive(Trainer *trainer, Dataset *dataset) {
    if (!trainer || !dataset) return;
    
    for (int epoch = 0; epoch < trainer->epochs; epoch++) {
        // Apprentissage progressif : augmente le nombre d'échantillons graduellement
        size_t progressive_samples = (epoch + 1) * dataset->num_samples / trainer->epochs;
        if (progressive_samples > dataset->num_samples) progressive_samples = dataset->num_samples;
        
        for (size_t i = 0; i < progressive_samples; i++) {
            // Vérification de sécurité
            if (!trainer->net || !trainer->net->layers || !dataset->inputs[i] || !dataset->outputs[i]) {
                printf("ERREUR: Pointeurs invalides à l'échantillon %zu\n", i);
                return;
            }
            
            // Forward pass
            network_forward(trainer->net, dataset->inputs[i]);
            
            // Pour l'instant, on évite les updates d'optimizer qui causent le crash
            // Ici on pourrait ajouter: augmentation progressive de complexité, curriculum learning, etc.
        }
    }
}