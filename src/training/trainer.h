#ifndef TRAINER_H
#define TRAINER_H

#include "../neural/network.h"
#include "../data/dataset.h"
#include "../optimizers/optimizer.h"

// Pointeur de fonction pour mise à jour optimiseur
typedef void (*OptimizerUpdateFn)(void *state, float *weights, float *gradients);

// On fait la déclaration du type Trainer ici AVANT le typedef de la fonction :
typedef struct Trainer Trainer;

// Pointeur de fonction pour stratégie d'entraînement
typedef void (*TrainingStrategyFn)(Trainer *trainer, Dataset *dataset);



// Structure principale Trainer
struct Trainer {
    NeuralNetwork *net;
    void *optimizer_state;
    OptimizerUpdateFn optimizer_update;
    TrainingStrategyFn train_strategy;
    float learning_rate;
    int epochs;
    int batch_size;
    char strategy_name[32];
    char optimizer_name[32];
    int progress_bar_id;  // ID de la barre de progression pour cet entraînement
};

// Création générique du trainer selon optimizer et méthode
Trainer *trainer_create(NeuralNetwork *net,
                        const char *optimizer,
                        float lr,
                        int epochs,
                        int batch_size,
                        TrainingStrategyFn train_strategy,
                        void *optimizer_state,
                        OptimizerUpdateFn optimizer_update);



// API générique pour entraîner et valider
void trainer_train(Trainer *trainer, Dataset *dataset);
float trainer_validate(Trainer *trainer, Dataset *dataset);

// Libération
void trainer_free(Trainer *trainer);


// Helpers pour mapping string->optimizer
void *trainer_create_optimizer_state(const char *optimizer, size_t num_weights, float lr);
OptimizerUpdateFn trainer_get_optimizer_update(const char *optimizer);

#endif