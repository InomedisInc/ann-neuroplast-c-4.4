#ifndef INTEGRATION_MAIN_H
#define INTEGRATION_MAIN_H

#include "model_saver.h"
#include "../neural/network.h"
#include "../training/trainer.h"
#include "../data/dataset.h"

/*
 * Fonctions d'intégration de model_saver dans le programme principal
 * 
 * Ces fonctions permettent d'intégrer facilement la sauvegarde automatique
 * des 10 meilleurs modèles dans votre boucle d'entraînement existante.
 */

// Fonctions d'intégration du Model Saver

// Initialiser le système de sauvegarde avec nom de dataset spécifique
int init_model_saver_with_dataset(const char *base_directory, const char *dataset_name);

// Initialiser le système de sauvegarde (version legacy)
int init_model_saver(const char *save_directory);

// Évaluer et potentiellement sauvegarder un modèle
int evaluate_and_save_model(NeuralNetwork *network, Trainer *trainer, 
                           Dataset *train_data, Dataset *val_data, int epoch);

// Sauvegarder tous les modèles à la fin de l'entraînement
int finalize_model_saving(SaveFormat format);

// Nettoyer le système de sauvegarde
void cleanup_model_saver(void);

// Charger le meilleur modèle sauvegardé
NeuralNetwork *load_best_model(const char *save_directory, ModelMetadata *metadata);

#endif // INTEGRATION_MAIN_H 