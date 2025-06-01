#include "all_methods.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../config.h"
#include "../data/data_loader.h"
#include "../data/dataset.h"
#include "../reporting/experiment_results.h"

// Prototype de fonction interne
static void run_single_test(const char *neuroplast_method,
                            const char *optimizer,
                            const char *activation,
                            int seed);

bool compare_all_methods(const char *config_file, 
                         const char *optim_config,
                         char **neuroplast_methods, int num_neuroplast_methods,
                         char **optimizers, int num_optimizers,
                         char **activations, int num_activations,
                         int seed) {
    printf("\nDémarrage des comparaisons complètes...\n");
    
    // Vérification des paramètres
    if (!config_file) {
        printf("Erreur: config_file est NULL\n");
        return false;
    }
    
    if (!neuroplast_methods || num_neuroplast_methods <= 0) {
        printf("Erreur: neuroplast_methods invalide\n");
        return false;
    }
    
    if (!optimizers || num_optimizers <= 0) {
        printf("Erreur: optimizers invalide\n");
        return false;
    }
    
    if (!activations || num_activations <= 0) {
        printf("Erreur: activations invalide\n");
        return false;
    }

    for (int i = 0; i < num_neuroplast_methods; i++) {
        for (int j = 0; j < num_optimizers; j++) {
            for (int k = 0; k < num_activations; k++) {
                printf("\n[Exp] Méthode: %s | Optimizer: %s | Activation: %s\n",
                       neuroplast_methods[i], optimizers[j], activations[k]);
                
                // Désactivé temporairement pour déboguer
                // run_single_test(neuroplast_methods[i], optimizers[j], activations[k], seed);
                printf("Test désactivé pour déboguer\n");
            }
        }
    }

    printf("\nToutes les comparaisons sont terminées.\n");
    return true;
}

// Implémentation simplifiée d'une fonction de test unitaire
static void run_single_test(const char *neuroplast_method,
                            const char *optimizer,
                            const char *activation,
                            int seed) {
    printf("Entraînement avec méthode %s, optimiseur %s, activation %s, seed %d...\n",
           neuroplast_method, optimizer, activation, seed);

    // Simulation du processus (à remplacer par ta logique réelle)
    printf("Résultats fictifs : Accuracy: 0.95 | F1-score: 0.94 | AUC: 0.96\n");
}