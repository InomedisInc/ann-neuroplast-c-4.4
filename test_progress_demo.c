#include <stdio.h>
#include <unistd.h>
#include "src/progress_bar.h"
#include "src/colored_output.h"

int main() {
    // Initialiser le système de barres de progression avec zones séparées
    progress_init_dual_zone("Test des Barres de Progression Améliorées", 3, 5, 10);
    
    // Créer les barres de progression
    int general_bar = progress_global_add(PROGRESS_GENERAL, "Combinaisons", 3, 25);
    int trials_bar = progress_global_add(PROGRESS_TRIALS, "Essais", 5, 25);
    int epochs_bar = progress_global_add(PROGRESS_EPOCHS, "Époques", 10, 20);
    
    // Simuler l'entraînement avec 3 combinaisons
    for (int combo = 0; combo < 3; combo++) {
        // Afficher l'en-tête de la combinaison
        char method[32], optimizer[32], activation[32];
        snprintf(method, sizeof(method), "neuroplast");
        snprintf(optimizer, sizeof(optimizer), combo == 0 ? "adamw" : combo == 1 ? "adam" : "radam");
        snprintf(activation, sizeof(activation), combo == 0 ? "relu" : combo == 1 ? "gelu" : "mish");
        
        progress_display_combination_header(combo + 1, 3, method, optimizer, activation);
        
        // Afficher les informations du réseau
        progress_display_network_info("Input(2)→256→128→Output(1)", "XOR Dataset (4 samples)", 0.001f, NULL);
        
        // Mettre à jour la barre générale
        progress_global_update(general_bar, combo, 0.0f, 0.0f, 0.0f);
        
        // Simuler 5 essais
        for (int trial = 0; trial < 5; trial++) {
            // Mettre à jour la barre des essais
            progress_global_update(trials_bar, trial, 0.0f, 0.0f, 0.0f);
            
            // Simuler 10 époques
            for (int epoch = 0; epoch < 10; epoch++) {
                // Simuler des métriques d'entraînement
                float loss = 1.0f - (epoch * 0.1f) - (trial * 0.02f) - (combo * 0.01f);
                float accuracy = 0.5f + (epoch * 0.05f) + (trial * 0.01f) + (combo * 0.005f);
                float precision = accuracy + 0.02f;
                float recall = accuracy - 0.01f;
                float f1_score = 2 * (precision * recall) / (precision + recall);
                
                if (loss < 0.0f) loss = 0.001f;
                if (accuracy > 1.0f) accuracy = 1.0f;
                if (precision > 1.0f) precision = 1.0f;
                if (recall > 1.0f) recall = 1.0f;
                if (f1_score > 1.0f) f1_score = 1.0f;
                
                // Mettre à jour la barre des époques
                progress_global_update(epochs_bar, epoch, loss, accuracy, 0.001f);
                
                // Afficher les informations de l'époque
                progress_display_epoch_info(epoch, 10, loss, accuracy, precision, recall, f1_score);
                
                // Pause pour voir l'animation
                usleep(200000); // 200ms
            }
            
            // Afficher le résumé de l'essai
            float best_accuracy = 0.9f + (trial * 0.02f);
            float best_f1 = best_accuracy + 0.01f;
            progress_display_trial_summary(trial, 5, best_accuracy, best_f1, 8);
            
            usleep(300000); // 300ms
        }
        
        // Afficher le résumé de la combinaison
        float avg_f1 = 0.85f + (combo * 0.05f);
        float best_f1 = avg_f1 + 0.1f;
        progress_display_combination_summary(avg_f1, best_f1, 4, 5);
        
        // Préparer pour la prochaine combinaison
        if (combo < 2) {
            progress_prepare_next_combination();
            usleep(500000); // 500ms
        }
    }
    
    // Finaliser les barres
    progress_global_update(general_bar, 3, 0.0f, 0.0f, 0.0f);
    progress_global_update(trials_bar, 5, 0.0f, 0.0f, 0.0f);
    progress_global_update(epochs_bar, 10, 0.001f, 0.95f, 0.001f);
    
    // Nettoyer
    progress_global_clear();
    
    printf("\n\n🎉 Démonstration terminée ! Les barres de progression sont maintenant corrigées.\n");
    printf("✅ Plus de superposition de texte\n");
    printf("✅ Zones d'affichage bien séparées\n");
    printf("✅ Couleurs et émojis améliorés\n");
    printf("✅ Positionnement fixe et stable\n\n");
    
    return 0;
} 