#ifndef RICH_CONFIG_H
#define RICH_CONFIG_H

#include <stddef.h>

#define MAX_METHODS 32
#define MAX_PARAMS 16
#define MAX_NAME 64
#define MAX_DATASETS 16

typedef struct {
    char key[MAX_NAME];
    float value;
} Param;

typedef struct {
    char name[MAX_NAME];
    char optimization_method[MAX_NAME];
    char optimized_with[MAX_NAME];
    Param params[MAX_PARAMS];
    int num_params;
} Activation;

typedef struct {
    char name[MAX_NAME];
    Param params[MAX_PARAMS];
    int num_params;
} OptimizerDef;

typedef struct {
    char name[MAX_NAME];
} MetricDef;

typedef struct {
    char name[MAX_NAME];
} NeuroplastMethod;

typedef struct {
    // Configuration du dataset
    char dataset[256];
    char base_dataset[256];
    char datasets[MAX_DATASETS][256];
    int num_datasets;
    size_t input_cols;
    size_t output_cols;
    char dataset_yaml[256];

    // Configuration pour le traitement d'images
    char image_train_dir[256];     // Répertoire d'entraînement (obligatoire)
    char image_test_dir[256];      // Répertoire de test (obligatoire)
    char image_val_dir[256];       // Répertoire de validation (optionnel)
    int image_width;               // Largeur des images
    int image_height;              // Hauteur des images
    int image_channels;            // Nombre de canaux (1=grayscale, 3=RGB)
    int is_image_dataset;          // 0 = données tabulaires, 1 = images
    float train_test_split;        // Ratio de division train/test (pour données tabulaires)

    // Configuration de l'entraînement
    int batch_size;
    int max_epochs;
    float learning_rate;
    
    // Configuration de l'early stopping
    int early_stopping;  // 0 = désactivé, 1 = activé
    int patience;        // nombre d'époques sans amélioration avant arrêt
    
    // Configuration de l'optimisation adaptative
    int optimized_parameters;  // 0 = configuration statique, 1 = optimiseur temps réel
    
    // Méthodes et paramètres
    NeuroplastMethod neuroplast_methods[MAX_METHODS];
    int num_neuroplast_methods;
    
    Activation activations[MAX_METHODS];
    int num_activations;
    
    OptimizerDef optimizers[MAX_METHODS];
    int num_optimizers;
    
    MetricDef metrics[MAX_METHODS];
    int num_metrics;
} RichConfig;

int parse_yaml_rich_config(const char *filename, RichConfig *cfg);

void merge_rich_configs(RichConfig *dest, const RichConfig *src);

#endif