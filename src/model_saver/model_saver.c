#include "model_saver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include "../memory.h"

// Créer un ModelSaver
ModelSaver *model_saver_create(const char *save_directory) {
    ModelSaver *saver = malloc(sizeof(ModelSaver));
    if (!saver) return NULL;
    
    saver->count = 0;
    saver->next_model_id = 1;
    strncpy(saver->save_directory, save_directory, sizeof(saver->save_directory) - 1);
    saver->save_directory[sizeof(saver->save_directory) - 1] = '\0';
    
    // Créer le répertoire s'il n'existe pas
    mkdir(save_directory, 0755);
    
    // Initialiser les modèles
    for (int i = 0; i < 10; i++) {
        saver->models[i].network = NULL;
        saver->models[i].score = -1.0f;
        saver->models[i].metadata.layer_sizes = NULL;
        saver->models[i].metadata.activation_names = NULL;
    }
    
    return saver;
}

// Libérer la mémoire du ModelSaver
void model_saver_free(ModelSaver *saver) {
    if (!saver) return;
    
    for (int i = 0; i < saver->count; i++) {
        if (saver->models[i].network) {
            network_free(saver->models[i].network);
            saver->models[i].network = NULL;
        }
        if (saver->models[i].metadata.layer_sizes) {
            free(saver->models[i].metadata.layer_sizes);
            saver->models[i].metadata.layer_sizes = NULL;
        }
        if (saver->models[i].metadata.activation_names) {
            for (size_t j = 0; j < saver->models[i].metadata.num_layers; j++) {
                if (saver->models[i].metadata.activation_names[j]) {
                    free(saver->models[i].metadata.activation_names[j]);
                }
            }
            free(saver->models[i].metadata.activation_names);
            saver->models[i].metadata.activation_names = NULL;
        }
    }
    
    free(saver);
}

// Calculer le score composite d'un modèle
float model_saver_calculate_score(float accuracy, float loss, float val_accuracy, float val_loss) {
    // Score composite : moyenne pondérée de précision et inverse de la perte
    // Plus le score est élevé, meilleur est le modèle
    float score = (accuracy * 0.4f) + (val_accuracy * 0.4f) + 
                  ((1.0f / (1.0f + loss)) * 0.1f) + 
                  ((1.0f / (1.0f + val_loss)) * 0.1f);
    return score;
} 