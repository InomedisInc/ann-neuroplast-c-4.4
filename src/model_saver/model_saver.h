#ifndef MODEL_SAVER_H
#define MODEL_SAVER_H

#include <stddef.h>
#include <time.h>
#include "../neural/network.h"
#include "../training/trainer.h"

// Structure pour stocker les métadonnées d'un modèle
typedef struct {
    float accuracy;
    float loss;
    float validation_accuracy;
    float validation_loss;
    int epoch;
    time_t timestamp;
    char model_name[64];
    char optimizer_name[32];
    char strategy_name[32];
    float learning_rate;
    int batch_size;
    size_t num_layers;
    size_t *layer_sizes;
    char **activation_names;
} ModelMetadata;

// Structure pour un modèle sauvegardé
typedef struct {
    ModelMetadata metadata;
    NeuralNetwork *network;
    float score; // Score composite pour le classement
} SavedModel;

// Structure pour gérer les 10 meilleurs modèles
typedef struct {
    SavedModel models[10];
    int count;
    char save_directory[256];
    int next_model_id;
} ModelSaver;

// Énumération pour les formats de sauvegarde
typedef enum {
    FORMAT_PTH,
    FORMAT_H5,
    FORMAT_BOTH
} SaveFormat;

// Fonctions principales
ModelSaver *model_saver_create(const char *save_directory);
void model_saver_free(ModelSaver *saver);

// Ajouter un modèle candidat
int model_saver_add_candidate(ModelSaver *saver, 
                             NeuralNetwork *network,
                             Trainer *trainer,
                             float accuracy,
                             float loss,
                             float validation_accuracy,
                             float validation_loss,
                             int epoch);

// Sauvegarder tous les modèles
int model_saver_save_all(ModelSaver *saver, SaveFormat format);

// Charger un modèle spécifique
NeuralNetwork *model_saver_load_model(const char *filepath, ModelMetadata *metadata);

// Utilitaires
void model_saver_print_rankings(ModelSaver *saver);
float model_saver_calculate_score(float accuracy, float loss, float val_accuracy, float val_loss);
int model_saver_export_python_interface(ModelSaver *saver, const char *output_file);

// Fonctions internes de sérialisation
int model_saver_save_pth(const SavedModel *model, const char *filepath);
int model_saver_save_h5(const SavedModel *model, const char *filepath);
NeuralNetwork *model_saver_load_pth(const char *filepath, ModelMetadata *metadata);
NeuralNetwork *model_saver_load_h5(const char *filepath, ModelMetadata *metadata);

#endif // MODEL_SAVER_H 