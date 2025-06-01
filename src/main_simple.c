#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "neural/network_simple.h"
#include "neural/activation.h"
#include "colored_output.h"

// Structure pour stocker toutes les métriques
typedef struct {
    float accuracy;
    float precision;
    float recall;
    float f1_score;
    float loss;
} SimpleMetrics;

// Fonction pour calculer les métriques de base
SimpleMetrics compute_metrics(NeuralNetwork *network, float inputs[][2], float targets[][1], int num_samples) {
    SimpleMetrics metrics = {0};
    
    int correct = 0;
    int TP = 0, TN = 0, FP = 0, FN = 0;
    float total_loss = 0.0f;
    
    for (int i = 0; i < num_samples; i++) {
        network_forward_simple(network, inputs[i]);
        float *output = network_output_simple(network);
        
        if (output) {
            float predicted = output[0];
            float target = targets[i][0];
            
            // Protection contre log(0) pour la loss
            predicted = fmaxf(1e-7f, fminf(1.0f - 1e-7f, predicted));
            float loss = -(target * logf(predicted) + (1.0f - target) * logf(1.0f - predicted));
            total_loss += loss;
            
            // Classification
            int pred_class = (predicted > 0.5f) ? 1 : 0;
            int true_class = (target > 0.5f) ? 1 : 0;
            
            if (pred_class == true_class) correct++;
            
            // Confusion matrix
            if (true_class == 1 && pred_class == 1) TP++;
            else if (true_class == 0 && pred_class == 0) TN++;
            else if (true_class == 0 && pred_class == 1) FP++;
            else if (true_class == 1 && pred_class == 0) FN++;
        }
    }
    
    metrics.accuracy = (float)correct / num_samples;
    metrics.loss = total_loss / num_samples;
    metrics.precision = (TP + FP > 0) ? (float)TP / (TP + FP) : 0.0f;
    metrics.recall = (TP + FN > 0) ? (float)TP / (TP + FN) : 0.0f;
    metrics.f1_score = (metrics.precision + metrics.recall > 0) ? 
                       2.0f * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall) : 0.0f;
    
    return metrics;
}

void print_banner() {
    printf("\n");
    printf(COLOR_CYAN "======================================================\n");
    printf("🧠 NEUROPLAST-ANN - Architecture Simplifiée Robuste\n");
    printf("======================================================\n" COLOR_RESET);
    printf("  Inspirée des meilleures pratiques de Scikit-Learn  \n\n");
}

int main() {
    print_banner();
    
    // Seed fixe pour reproductibilité
    srand(42);
    
    // Données XOR (problème classique non-linéaire)
    float inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f}, 
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    
    float targets[4][1] = {
        {0.0f}, // 0 XOR 0 = 0
        {1.0f}, // 0 XOR 1 = 1
        {1.0f}, // 1 XOR 0 = 1
        {0.0f}  // 1 XOR 1 = 0
    };
    
    printf(COLOR_YELLOW "🎯 Problème: XOR (classification binaire non-linéaire)\n" COLOR_RESET);
    printf("Dataset:\n");
    for (int i = 0; i < 4; i++) {
        printf("  [%.1f, %.1f] → %.1f\n", inputs[i][0], inputs[i][1], targets[i][0]);
    }
    
    // ARCHITECTURE SCIKIT-LEARN STYLE
    printf(COLOR_GREEN "\n🏗️ Architecture: Style Scikit-Learn MLPClassifier\n" COLOR_RESET);
    size_t layer_sizes[] = {2, 100, 1}; // 2 → 100 → 1 (architecture par défaut sklearn)
    const char *activations[] = {"relu", "sigmoid"}; // ReLU + Sigmoid (standard)
    
    NeuralNetwork *network = network_create_simple(3, layer_sizes, activations);
    if (!network) {
        printf(COLOR_RED "❌ Erreur: impossible de créer le réseau\n" COLOR_RESET);
        return 1;
    }
    
    printf("✅ Réseau créé: 2 → 100 → 1 (ReLU → Sigmoid)\n");
    printf("✅ Optimiseur: SGD avec momentum 0.9\n");
    printf("✅ Initialisation: Xavier/Glorot\n");
    
    // PARAMÈTRES D'ENTRAÎNEMENT
    printf(COLOR_BLUE "\n🚀 Entraînement...\n" COLOR_RESET);
    int max_epochs = 10000;
    float learning_rate = 0.1f; // LR plus élevé pour convergence rapide
    float best_accuracy = 0.0f;
    int patience = 1000;
    int stagnation = 0;
    
    printf("Max epochs: %d\n", max_epochs);
    printf("Learning rate: %.3f\n", learning_rate);
    printf("Early stopping: patience = %d\n", patience);
    
    // BOUCLE D'ENTRAÎNEMENT
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        // Entraînement sur batch complet
        for (int i = 0; i < 4; i++) {
            network_forward_simple(network, inputs[i]);
            network_backward_simple(network, inputs[i], targets[i], learning_rate);
        }
        
        // Évaluation périodique
        if (epoch % 100 == 0) {
            SimpleMetrics metrics = compute_metrics(network, inputs, targets, 4);
            
            printf("Époque %4d - Loss: %.6f - Accuracy: %.1f%% - F1: %.3f\n", 
                   epoch, metrics.loss, metrics.accuracy * 100, metrics.f1_score);
            
            // Early stopping
            if (metrics.accuracy > best_accuracy) {
                best_accuracy = metrics.accuracy;
                stagnation = 0;
            } else {
                stagnation++;
            }
            
            // Convergence parfaite
            if (metrics.accuracy >= 1.0f) {
                printf(COLOR_GREEN "✅ Convergence parfaite atteinte à l'époque %d!\n" COLOR_RESET, epoch);
                break;
            }
            
            // Arrêt par stagnation
            if (stagnation >= patience / 100) {
                printf(COLOR_YELLOW "⚠️ Arrêt par stagnation à l'époque %d\n" COLOR_RESET, epoch);
                break;
            }
        }
    }
    
    // ÉVALUATION FINALE
    printf(COLOR_MAGENTA "\n=== RÉSULTATS FINAUX ===\n" COLOR_RESET);
    SimpleMetrics final_metrics = compute_metrics(network, inputs, targets, 4);
    
    printf("📊 Métriques finales:\n");
    printf("  Accuracy:  %.4f (%.2f%%)\n", final_metrics.accuracy, final_metrics.accuracy * 100);
    printf("  Precision: %.4f (%.2f%%)\n", final_metrics.precision, final_metrics.precision * 100);
    printf("  Recall:    %.4f (%.2f%%)\n", final_metrics.recall, final_metrics.recall * 100);
    printf("  F1-Score:  %.4f (%.2f%%)\n", final_metrics.f1_score, final_metrics.f1_score * 100);
    printf("  Final Loss: %.6f\n", final_metrics.loss);
    
    printf("\n🔍 Prédictions détaillées:\n");
    for (int i = 0; i < 4; i++) {
        network_forward_simple(network, inputs[i]);
        float *output = network_output_simple(network);
        
        if (output) {
            float predicted = output[0];
            int pred_class = (predicted > 0.5f) ? 1 : 0;
            int true_class = (targets[i][0] > 0.5f) ? 1 : 0;
            const char *status = (pred_class == true_class) ? COLOR_GREEN "✅" COLOR_RESET : COLOR_RED "❌" COLOR_RESET;
            
            printf("  [%.1f, %.1f] → %.4f (%d) | Attendu: %d %s\n",
                   inputs[i][0], inputs[i][1], predicted, pred_class, true_class, status);
        }
    }
    
    // ÉVALUATION DE LA PERFORMANCE
    if (final_metrics.accuracy >= 1.0f) {
        printf(COLOR_GREEN "\n🏆 EXCELLENTE PERFORMANCE! (100%% accuracy)\n" COLOR_RESET);
        printf("🎉 Architecture validée avec succès!\n");
    } else if (final_metrics.accuracy >= 0.9f) {
        printf(COLOR_BLUE "\n👍 Très bonne performance (≥90%% accuracy)\n" COLOR_RESET);
    } else if (final_metrics.accuracy >= 0.75f) {
        printf(COLOR_YELLOW "\n⚠️ Performance modérée (≥75%% accuracy)\n" COLOR_RESET);
        printf("Recommandation: Ajuster les hyperparamètres\n");
    } else {
        printf(COLOR_RED "\n❌ Performance insuffisante (<75%% accuracy)\n" COLOR_RESET);
        printf("Problème identifié dans l'architecture\n");
    }
    
    // COMPARAISON AVEC SCIKIT-LEARN
    printf(COLOR_CYAN "\n=== COMPARAISON AVEC SCIKIT-LEARN ===\n" COLOR_RESET);
    printf("Accuracy obtenue: %.1f%%\n", final_metrics.accuracy * 100);
    printf("Scikit-Learn (typique): 80-100%%\n");
    
    if (final_metrics.accuracy >= 0.8f) {
        printf(COLOR_GREEN "✅ Performance comparable à Scikit-Learn!\n" COLOR_RESET);
    } else {
        printf(COLOR_RED "❌ Performance inférieure à Scikit-Learn\n" COLOR_RESET);
        printf("Suggestions d'amélioration:\n");
        printf("  - Augmenter le learning rate\n");
        printf("  - Ajouter plus de neurones\n");
        printf("  - Changer l'architecture\n");
    }
    
    // Nettoyage
    network_free_simple(network);
    
    printf(COLOR_CYAN "\n🎯 Test terminé avec succès!\n" COLOR_RESET);
    return 0;
} 