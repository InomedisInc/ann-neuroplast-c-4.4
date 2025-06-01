/*
 * Exemple d'intégration de model_saver dans main.c
 * 
 * Ce fichier montre comment intégrer la librairie model_saver
 * dans votre programme principal neuroplast-ann
 */

#include <stdio.h>
#include "model_saver.h"
#include "../neural/network.h"
#include "../training/trainer.h"
#include "../data/dataset.h"
#include "../evaluation/metrics.h"

// Variables globales pour la sauvegarde des modèles
static ModelSaver *global_model_saver = NULL;

// Initialiser le système de sauvegarde des modèles
int init_model_saver(const char *save_directory) {
    if (global_model_saver) {
        printf("⚠️  ModelSaver déjà initialisé\n");
        return 0;
    }
    
    global_model_saver = model_saver_create(save_directory);
    if (!global_model_saver) {
        printf("❌ Erreur: Impossible d'initialiser ModelSaver\n");
        return -1;
    }
    
    printf("✅ ModelSaver initialisé: %s\n", save_directory);
    return 0;
}

// Évaluer et potentiellement sauvegarder un modèle
int evaluate_and_save_model(NeuralNetwork *network, Trainer *trainer, 
                           Dataset *train_data, Dataset *val_data, int epoch) {
    if (!global_model_saver || !network || !trainer) {
        return -1;
    }
    
    // Évaluer le modèle sur les données d'entraînement
    float train_accuracy = trainer_validate(trainer, train_data);
    float train_loss = 0.0f; // À calculer selon votre implémentation
    
    // Évaluer le modèle sur les données de validation
    float val_accuracy = trainer_validate(trainer, val_data);
    float val_loss = 0.0f; // À calculer selon votre implémentation
    
    // Calculer les pertes (exemple simplifié)
    // Vous devrez adapter cela à votre implémentation de calcul de perte
    train_loss = 1.0f - train_accuracy; // Approximation simple
    val_loss = 1.0f - val_accuracy;     // Approximation simple
    
    // Ajouter le modèle candidat
    int result = model_saver_add_candidate(global_model_saver, network, trainer,
                                         train_accuracy, train_loss,
                                         val_accuracy, val_loss, epoch);
    
    if (result == 1) {
        printf("🏆 Époque %d: Modèle ajouté au top 10! (Acc: %.3f, Val: %.3f)\n", 
               epoch, train_accuracy, val_accuracy);
    } else if (result == 0) {
        printf("📊 Époque %d: Modèle pas dans le top 10 (Acc: %.3f, Val: %.3f)\n", 
               epoch, train_accuracy, val_accuracy);
    }
    
    return result;
}

// Sauvegarder tous les modèles à la fin de l'entraînement
int finalize_model_saving(SaveFormat format) {
    if (!global_model_saver) {
        printf("⚠️  ModelSaver non initialisé\n");
        return -1;
    }
    
    printf("\n🎯 === FINALISATION DE L'ENTRAÎNEMENT ===\n");
    
    // Afficher le classement final
    model_saver_print_rankings(global_model_saver);
    
    // Sauvegarder tous les modèles
    printf("\n💾 Sauvegarde des modèles...\n");
    int saved_count = model_saver_save_all(global_model_saver, format);
    printf("✅ %d fichiers sauvegardés\n", saved_count);
    
    // Exporter l'interface Python
    char python_file[512];
    snprintf(python_file, sizeof(python_file), "%s/model_loader.py", 
             global_model_saver->save_directory);
    
    if (model_saver_export_python_interface(global_model_saver, python_file) == 0) {
        printf("🐍 Interface Python exportée: %s\n", python_file);
    }
    
    return saved_count;
}

// Nettoyer le système de sauvegarde
void cleanup_model_saver(void) {
    if (global_model_saver) {
        model_saver_free(global_model_saver);
        global_model_saver = NULL;
        printf("🧹 ModelSaver nettoyé\n");
    }
}

// Charger le meilleur modèle sauvegardé
NeuralNetwork *load_best_model(const char *save_directory, ModelMetadata *metadata) {
    char best_model_path[512];
    snprintf(best_model_path, sizeof(best_model_path), "%s/model_1.h5", save_directory);
    
    NeuralNetwork *network = model_saver_load_model(best_model_path, metadata);
    if (network && metadata) {
        printf("🔄 Meilleur modèle chargé: %s (Acc: %.3f)\n", 
               metadata->model_name, metadata->accuracy);
    }
    
    return network;
}

/*
 * EXEMPLE D'INTÉGRATION DANS VOTRE MAIN.C :
 * 
 * int main(int argc, char *argv[]) {
 *     // ... votre code d'initialisation existant ...
 *     
 *     // Initialiser le système de sauvegarde
 *     if (init_model_saver("./best_models") != 0) {
 *         return 1;
 *     }
 *     
 *     // Votre boucle d'entraînement existante
 *     for (int epoch = 0; epoch < max_epochs; epoch++) {
 *         // ... votre code d'entraînement existant ...
 *         
 *         // Évaluer et potentiellement sauvegarder le modèle
 *         evaluate_and_save_model(network, trainer, train_data, val_data, epoch);
 *         
 *         // ... reste de votre code d'époque ...
 *     }
 *     
 *     // Finaliser la sauvegarde
 *     finalize_model_saving(FORMAT_BOTH);
 *     
 *     // ... votre code de nettoyage existant ...
 *     
 *     // Nettoyer le système de sauvegarde
 *     cleanup_model_saver();
 *     
 *     return 0;
 * }
 */ 