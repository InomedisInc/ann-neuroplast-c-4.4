#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "src/neural/network_simple.h"
#include "src/data/dataset.h"
#include "src/data/split.h"
#include "src/evaluation/metrics.h"

// Test équitable des activations avec paramètres identiques
int main() {
    printf("⚖️ TEST ÉQUITABLE DES ACTIVATIONS - PARAMÈTRES IDENTIQUES\n");
    printf("=========================================================\n\n");
    
    const char *activations[] = {
        "relu", "sigmoid", "tanh", "gelu", "leaky_relu", 
        "elu", "mish", "swish", "prelu", "neuroplast"
    };
    int num_activations = sizeof(activations) / sizeof(activations[0]);
    
    // Créer un dataset synthétique mais réaliste
    printf("🔧 Création d'un dataset synthétique réaliste...\n");
    
    Dataset *full_dataset = malloc(sizeof(Dataset));
    full_dataset->num_samples = 1000;
    full_dataset->input_cols = 8;
    full_dataset->output_cols = 1;
    
    // Allocation
    full_dataset->inputs = malloc(1000 * sizeof(float*));
    full_dataset->outputs = malloc(1000 * sizeof(float*));
    for (int i = 0; i < 1000; i++) {
        full_dataset->inputs[i] = malloc(8 * sizeof(float));
        full_dataset->outputs[i] = malloc(1 * sizeof(float));
    }
    
    // Génération de données non-linéaires réalistes
    srand(42); // Seed fixe pour reproductibilité
    for (int i = 0; i < 1000; i++) {
        // 8 features avec corrélations complexes
        for (int j = 0; j < 8; j++) {
            full_dataset->inputs[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // [-1,1]
        }
        
        // Target basé sur une combinaison non-linéaire des features
        float f1 = full_dataset->inputs[i][0];
        float f2 = full_dataset->inputs[i][1];
        float f3 = full_dataset->inputs[i][2];
        float f4 = full_dataset->inputs[i][3];
        
        float complex_value = f1*f1 + sinf(f2) + f3*f4 + 0.5f*f1*f2;
        full_dataset->outputs[i][0] = (complex_value > 0) ? 1.0f : 0.0f;
    }
    
    // Division train/test
    Dataset *train_set = NULL, *test_set = NULL;
    split_dataset(full_dataset, 0.8f, &train_set, &test_set);
    
    printf("✅ Dataset créé: %zu train, %zu test\n", train_set->num_samples, test_set->num_samples);
    printf("📊 Répartition des classes:\n");
    
    int train_pos = 0, test_pos = 0;
    for (size_t i = 0; i < train_set->num_samples; i++) {
        if (train_set->outputs[i][0] > 0.5f) train_pos++;
    }
    for (size_t i = 0; i < test_set->num_samples; i++) {
        if (test_set->outputs[i][0] > 0.5f) test_pos++;
    }
    
    printf("   Train: %d positifs / %zu total (%.1f%%)\n", 
           train_pos, train_set->num_samples, 100.0f * train_pos / train_set->num_samples);
    printf("   Test: %d positifs / %zu total (%.1f%%)\n\n", 
           test_pos, test_set->num_samples, 100.0f * test_pos / test_set->num_samples);
    
    // Test de chaque activation avec PARAMÈTRES IDENTIQUES
    printf("🧪 TEST DES ACTIVATIONS - CONDITIONS IDENTIQUES\n");
    printf("===============================================\n\n");
    
    typedef struct {
        char name[32];
        float final_accuracy;
        float final_f1;
        float final_loss;
        int convergence_epoch;
    } ActivationResult;
    
    ActivationResult results[10];
    
    for (int a = 0; a < num_activations; a++) {
        printf("🎯 Test activation %d/%d: %s\n", a+1, num_activations, activations[a]);
        strcpy(results[a].name, activations[a]);
        
        // ARCHITECTURE IDENTIQUE POUR TOUS
        size_t layer_sizes[] = {8, 64, 32, 1};  // Architecture modeste et fixe
        const char *test_activations[] = {activations[a], activations[a], "sigmoid"};
        
        NeuralNetwork *network = network_create_simple(4, layer_sizes, test_activations);
        if (!network) {
            printf("❌ Erreur création réseau pour %s\n", activations[a]);
            results[a].final_accuracy = 0.0f;
            results[a].final_f1 = 0.0f;
            results[a].final_loss = 999.0f;
            results[a].convergence_epoch = -1;
            continue;
        }
        
        // PARAMÈTRES D'ENTRAÎNEMENT IDENTIQUES POUR TOUS
        float learning_rate = 0.005f;  // LR fixe, pas d'ajustement
        int max_epochs = 100;           // Nombre d'époques fixe
        float best_f1 = 0.0f;
        int convergence_epoch = -1;
        
        printf("   Architecture: 8→64→32→1, LR=%.4f, %d époques max\n", learning_rate, max_epochs);
        
        // Entraînement sans ajustements spéciaux
        for (int epoch = 0; epoch < max_epochs; epoch++) {
            // Une seule passe par époque (pas de multi-pass)
            for (size_t i = 0; i < train_set->num_samples; i++) {
                network_forward_simple(network, train_set->inputs[i]);
                network_backward_simple(network, train_set->inputs[i], train_set->outputs[i], learning_rate);
            }
            
            // Évaluation toutes les 10 époques
            if (epoch % 10 == 0 || epoch == max_epochs - 1) {
                // Test sur le dataset de test
                int correct = 0;
                float total_loss = 0.0f;
                
                for (size_t i = 0; i < test_set->num_samples; i++) {
                    network_forward_simple(network, test_set->inputs[i]);
                    float *output = network_output_simple(network);
                    
                    float prediction = (output[0] > 0.5f) ? 1.0f : 0.0f;
                    float target = test_set->outputs[i][0];
                    
                    if (fabsf(prediction - target) < 0.1f) correct++;
                    
                    float error = output[0] - target;
                    total_loss += error * error;
                }
                
                float accuracy = (float)correct / test_set->num_samples;
                float avg_loss = total_loss / test_set->num_samples;
                
                // Calcul F1-Score simplifié
                int tp = 0, fp = 0, fn = 0;
                for (size_t i = 0; i < test_set->num_samples; i++) {
                    network_forward_simple(network, test_set->inputs[i]);
                    float *output = network_output_simple(network);
                    
                    int predicted = (output[0] > 0.5f) ? 1 : 0;
                    int actual = (test_set->outputs[i][0] > 0.5f) ? 1 : 0;
                    
                    if (predicted == 1 && actual == 1) tp++;
                    else if (predicted == 1 && actual == 0) fp++;
                    else if (predicted == 0 && actual == 1) fn++;
                }
                
                float precision = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.0f;
                float recall = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.0f;
                float f1 = (precision + recall > 0) ? 2.0f * precision * recall / (precision + recall) : 0.0f;
                
                if (f1 > best_f1) {
                    best_f1 = f1;
                    if (f1 > 0.7f && convergence_epoch == -1) {
                        convergence_epoch = epoch;
                    }
                }
                
                if (epoch % 20 == 0) {
                    printf("     Époque %d: Acc=%.3f, F1=%.3f, Loss=%.4f\n", 
                           epoch, accuracy, f1, avg_loss);
                }
                
                // Sauvegarder les métriques finales
                if (epoch == max_epochs - 1) {
                    results[a].final_accuracy = accuracy;
                    results[a].final_f1 = f1;
                    results[a].final_loss = avg_loss;
                    results[a].convergence_epoch = convergence_epoch;
                }
            }
        }
        
        printf("   ✅ Final: Acc=%.3f, F1=%.3f, Loss=%.4f", 
               results[a].final_accuracy, results[a].final_f1, results[a].final_loss);
        if (results[a].convergence_epoch >= 0) {
            printf(", Convergence: époque %d\n", results[a].convergence_epoch);
        } else {
            printf(", Pas de convergence\n");
        }
        printf("\n");
        
        network_free_simple(network);
    }
    
    // Analyse des résultats
    printf("📊 ANALYSE COMPARATIVE DES ACTIVATIONS\n");
    printf("=====================================\n\n");
    
    printf("Rang | Activation   | Accuracy | F1-Score | Loss    | Convergence\n");
    printf("-----|--------------|----------|----------|---------|------------\n");
    
    // Trier par F1-Score
    for (int i = 0; i < num_activations - 1; i++) {
        for (int j = i + 1; j < num_activations; j++) {
            if (results[j].final_f1 > results[i].final_f1) {
                ActivationResult temp = results[i];
                results[i] = results[j];
                results[j] = temp;
            }
        }
    }
    
    for (int i = 0; i < num_activations; i++) {
        printf("%4d | %-12s | %8.3f | %8.3f | %7.4f | ", 
               i+1, results[i].name, results[i].final_accuracy, results[i].final_f1, results[i].final_loss);
        if (results[i].convergence_epoch >= 0) {
            printf("Époque %d\n", results[i].convergence_epoch);
        } else {
            printf("Aucune\n");
        }
    }
    
    // Statistiques de variation
    float max_f1 = results[0].final_f1;
    float min_f1 = results[num_activations-1].final_f1;
    float variation_f1 = max_f1 - min_f1;
    
    float max_acc = -1.0f, min_acc = 2.0f;
    for (int i = 0; i < num_activations; i++) {
        if (results[i].final_accuracy > max_acc) max_acc = results[i].final_accuracy;
        if (results[i].final_accuracy < min_acc) min_acc = results[i].final_accuracy;
    }
    float variation_acc = max_acc - min_acc;
    
    printf("\n🔍 STATISTIQUES DE VARIATION :\n");
    printf("==============================\n");
    printf("📊 F1-Score:\n");
    printf("   Meilleur: %.3f (%s)\n", max_f1, results[0].name);
    printf("   Pire: %.3f (%s)\n", min_f1, results[num_activations-1].name);
    printf("   Variation: %.3f (%.1f%%)\n", variation_f1, 100.0f * variation_f1 / max_f1);
    
    printf("📊 Accuracy:\n");
    printf("   Meilleur: %.3f\n", max_acc);
    printf("   Pire: %.3f\n", min_acc);
    printf("   Variation: %.3f (%.1f%%)\n", variation_acc, 100.0f * variation_acc / max_acc);
    
    printf("\n🎯 CONCLUSIONS :\n");
    printf("===============\n");
    
    if (variation_f1 < 0.05f) {
        printf("❌ PROBLÈME CONFIRMÉ: Variation très faible (%.1f%%) entre activations\n", 
               100.0f * variation_f1 / max_f1);
        printf("   → Possibles causes:\n");
        printf("     • Dataset trop simple ou trop petit\n");
        printf("     • Architecture inappropriée\n");
        printf("     • Paramètres d'entraînement non optimaux\n");
        printf("     • Problème dans l'implémentation des activations\n");
    } else if (variation_f1 < 0.1f) {
        printf("⚠️ Variation modérée (%.1f%%) - Acceptable mais pourrait être améliorée\n", 
               100.0f * variation_f1 / max_f1);
    } else {
        printf("✅ Variation significative (%.1f%%) - Les activations se différencient bien\n", 
               100.0f * variation_f1 / max_f1);
    }
    
    // Nettoyage
    dataset_free(full_dataset);
    dataset_free(train_set);
    dataset_free(test_set);
    
    return (variation_f1 < 0.05f) ? 1 : 0;
} 