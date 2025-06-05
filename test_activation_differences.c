#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "src/neural/network_simple.h"
#include "src/neural/activation.h"

// Test pour vérifier que les activations produisent des résultats différents
int main() {
    printf("🔍 TEST DE VÉRIFICATION DES DIFFÉRENCES ENTRE ACTIVATIONS\n");
    printf("=========================================================\n\n");
    
    const char *activations[] = {
        "relu", "sigmoid", "tanh", "gelu", "leaky_relu", 
        "elu", "mish", "swish", "prelu", "neuroplast"
    };
    int num_activations = sizeof(activations) / sizeof(activations[0]);
    
    // Dataset de test simple mais représentatif
    float test_inputs[5][2] = {
        {-2.0f, 1.5f},  // Valeurs négatives/positives mélangées
        {0.0f, 0.0f},   // Valeurs nulles
        {1.0f, -1.0f},  // Valeurs opposées
        {3.0f, 2.0f},   // Valeurs positives grandes
        {-0.5f, 0.7f}   // Valeurs mixtes petites
    };
    float test_targets[5] = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f};
    
    printf("📊 Dataset de test :\n");
    for (int i = 0; i < 5; i++) {
        printf("   Input[%d]: [%.1f, %.1f] → Target: %.1f\n", 
               i, test_inputs[i][0], test_inputs[i][1], test_targets[i]);
    }
    printf("\n");
    
    // Tester chaque activation individuellement
    float results[10][5]; // [activation][test_case] = résultat
    
    for (int a = 0; a < num_activations; a++) {
        printf("🧪 Test activation: %s\n", activations[a]);
        
        // Créer un réseau simple: 2→4→1 avec l'activation testée
        size_t layer_sizes[] = {2, 4, 1};
        const char *test_activations[] = {activations[a], "sigmoid"};
        
        NeuralNetwork *network = network_create_simple(3, layer_sizes, test_activations);
        if (!network) {
            printf("❌ Erreur création réseau pour %s\n", activations[a]);
            continue;
        }
        
        // Entraînement rapide (20 époques)
        for (int epoch = 0; epoch < 20; epoch++) {
            for (int sample = 0; sample < 5; sample++) {
                network_forward_simple(network, test_inputs[sample]);
                float target_array[] = {test_targets[sample]};
                network_backward_simple(network, test_inputs[sample], target_array, 0.01f);
            }
        }
        
        // Enregistrer les résultats finaux
        printf("   Résultats après 20 époques :\n");
        float total_output = 0.0f;
        for (int sample = 0; sample < 5; sample++) {
            network_forward_simple(network, test_inputs[sample]);
            float *output = network_output_simple(network);
            results[a][sample] = output[0];
            total_output += output[0];
            printf("     Input[%d]: %.4f (target: %.1f)\n", sample, output[0], test_targets[sample]);
        }
        
        // Calculer une "signature" de l'activation
        float avg_output = total_output / 5.0f;
        float variance = 0.0f;
        for (int sample = 0; sample < 5; sample++) {
            float diff = results[a][sample] - avg_output;
            variance += diff * diff;
        }
        variance /= 5.0f;
        
        printf("   📊 Signature: Moyenne=%.4f, Variance=%.4f\n\n", avg_output, variance);
        
        network_free_simple(network);
    }
    
    // Analyser les différences entre activations
    printf("🔍 ANALYSE DES DIFFÉRENCES ENTRE ACTIVATIONS\n");
    printf("===========================================\n\n");
    
    printf("📊 Tableau des résultats finaux :\n");
    printf("Activation    | Test0   Test1   Test2   Test3   Test4   | Variance\n");
    printf("------------- | ------ ------ ------ ------ ------ | --------\n");
    
    for (int a = 0; a < num_activations; a++) {
        printf("%-12s  | ", activations[a]);
        
        float total = 0.0f;
        for (int sample = 0; sample < 5; sample++) {
            printf("%6.3f ", results[a][sample]);
            total += results[a][sample];
        }
        
        float avg = total / 5.0f;
        float variance = 0.0f;
        for (int sample = 0; sample < 5; sample++) {
            float diff = results[a][sample] - avg;
            variance += diff * diff;
        }
        variance /= 5.0f;
        
        printf("| %8.4f\n", variance);
    }
    
    // Vérifier si certaines activations donnent des résultats trop similaires
    printf("\n🚨 DÉTECTION DE PROBLÈMES POTENTIELS :\n");
    printf("=====================================\n");
    
    int suspicious_pairs = 0;
    for (int a1 = 0; a1 < num_activations; a1++) {
        for (int a2 = a1 + 1; a2 < num_activations; a2++) {
            float total_diff = 0.0f;
            for (int sample = 0; sample < 5; sample++) {
                float diff = fabsf(results[a1][sample] - results[a2][sample]);
                total_diff += diff;
            }
            float avg_diff = total_diff / 5.0f;
            
            if (avg_diff < 0.05f) {  // Différence moyenne < 5%
                printf("⚠️ %s et %s sont très similaires (diff moyenne: %.4f)\n", 
                       activations[a1], activations[a2], avg_diff);
                suspicious_pairs++;
            }
        }
    }
    
    if (suspicious_pairs == 0) {
        printf("✅ Toutes les activations produisent des résultats différenciés\n");
    } else {
        printf("❌ %d paires d'activations sont suspectes (trop similaires)\n", suspicious_pairs);
    }
    
    printf("\n🎯 CONCLUSIONS :\n");
    printf("===============\n");
    
    if (suspicious_pairs > 5) {
        printf("❌ PROBLÈME MAJEUR : Les activations ne sont pas bien différenciées\n");
        printf("   → Vérifier l'implémentation dans src/neural/activation.c\n");
        printf("   → Vérifier l'application dans src/neural/network_simple.c\n");
        return 1;
    } else if (suspicious_pairs > 0) {
        printf("⚠️ PROBLÈME MINEUR : Quelques activations sont trop similaires\n");
        printf("   → Possiblement dû aux poids initiaux ou à l'architecture simple\n");
        return 0;
    } else {
        printf("✅ AUCUN PROBLÈME : Les activations sont bien différenciées\n");
        printf("   → Le problème de variation faible vient d'ailleurs\n");
        return 0;
    }
} 