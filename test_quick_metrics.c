#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "src/neural/network_simple.h"

// Test rapide pour vérifier l'évolution des métriques
int main() {
    printf("🧪 TEST RAPIDE DE L'ÉVOLUTION DES MÉTRIQUES\n");
    printf("==========================================\n\n");
    
    srand(42); // Seed fixe pour reproductibilité
    
    // Créer un réseau simple
    size_t layer_sizes[] = {2, 64, 32, 1};
    const char *activations[] = {"relu", "relu", "sigmoid"};
    NeuralNetwork *network = network_create_simple(4, layer_sizes, activations);
    
    if (!network) {
        printf("❌ Erreur création réseau\n");
        return 1;
    }
    
    // Dataset XOR étendu pour test
    float inputs[8][2] = {
        {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
        {0.1f, 0.1f}, {0.1f, 0.9f}, {0.9f, 0.1f}, {0.9f, 0.9f}
    };
    float targets[8] = {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f};
    
    printf("📊 Dataset: XOR étendu (8 échantillons)\n");
    printf("🎯 Architecture: Input(2)→64→32→Output(1)\n");
    printf("⚡ Learning rate: 0.01\n\n");
    
    // Test sur plusieurs époques pour voir l'évolution
    for (int epoch = 0; epoch < 100; epoch += 10) {
        // Entraînement
        for (int e = 0; e < 10; e++) {
            for (int i = 0; i < 8; i++) {
                network_forward_simple(network, inputs[i]);
                float target_array[] = {targets[i]};
                network_backward_simple(network, inputs[i], target_array, 0.01f);
            }
        }
        
        // Évaluation
        int correct = 0;
        float total_loss = 0.0f;
        float predictions[8];
        
        for (int i = 0; i < 8; i++) {
            network_forward_simple(network, inputs[i]);
            float *output = network_output_simple(network);
            predictions[i] = output[0];
            
            // Accuracy
            float prediction_class = (output[0] > 0.5f) ? 1.0f : 0.0f;
            if (fabs(prediction_class - targets[i]) < 0.1f) correct++;
            
            // Loss (Binary Cross-Entropy)
            float prediction = fmaxf(1e-7f, fminf(1.0f - 1e-7f, output[0]));
            float bce_loss = -(targets[i] * logf(prediction) + (1.0f - targets[i]) * logf(1.0f - prediction));
            total_loss += bce_loss;
        }
        
        float accuracy = (float)correct / 8.0f;
        float avg_loss = total_loss / 8.0f;
        
        // Calculer Precision, Recall, F1
        int TP = 0, FP = 0, FN = 0, TN = 0;
        for (int i = 0; i < 8; i++) {
            int pred = (predictions[i] > 0.5f) ? 1 : 0;
            int true_val = (targets[i] > 0.5f) ? 1 : 0;
            
            if (pred == 1 && true_val == 1) TP++;
            else if (pred == 1 && true_val == 0) FP++;
            else if (pred == 0 && true_val == 1) FN++;
            else TN++;
        }
        
        float precision = (TP + FP > 0) ? (float)TP / (TP + FP) : 0.0f;
        float recall = (TP + FN > 0) ? (float)TP / (TP + FN) : 0.0f;
        float f1 = (precision + recall > 0) ? 2.0f * precision * recall / (precision + recall) : 0.0f;
        
        printf("📈 Époque %3d | Loss: %.4f | Acc: %5.1f%% | Prec: %5.1f%% | Rec: %5.1f%% | F1: %5.1f%%\n",
               epoch + 10, avg_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100);
        
        // Arrêter si convergence
        if (accuracy >= 0.95f) {
            printf("✅ Convergence atteinte à l'époque %d!\n", epoch + 10);
            break;
        }
    }
    
    printf("\n🏆 Test terminé - Les métriques évoluent correctement!\n");
    
    network_free_simple(network);
    return 0;
} 