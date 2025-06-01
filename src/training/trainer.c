#include "trainer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Inclure tous les headers des optimiseurs
#include "../optimizers/sgd.h"
#include "../optimizers/adam.h"
#include "../optimizers/adamw.h"
#include "../optimizers/rmsprop.h"
#include "../optimizers/lion.h"
#include "../optimizers/adabelief.h"
#include "../optimizers/radam.h"
#include "../optimizers/adamax.h"
#include "../optimizers/nadam.h"

// Fonctions adaptateurs pour chaque optimiseur
void sgd_update_adapter(void *state, float *w, float *g)       { sgd_update((SGDState*)state, w, g); }
void adam_update_adapter(void *state, float *w, float *g)      { adam_update((AdamState*)state, w, g); }
void adamw_update_adapter(void *state, float *w, float *g)     { adamw_update((AdamWState*)state, w, g); }
void rmsprop_update_adapter(void *state, float *w, float *g)   { rmsprop_update((RMSPropState*)state, w, g); }
void lion_update_adapter(void *state, float *w, float *g)      { lion_update((LionState*)state, w, g); }
void adabelief_update_adapter(void *state, float *w, float *g) { adabelief_update((AdaBeliefState*)state, w, g); }
void radam_update_adapter(void *state, float *w, float *g)     { radam_update((RAdamState*)state, w, g); }
void adamax_update_adapter(void *state, float *w, float *g)    { adamax_update((AdamaxState*)state, w, g); }
void nadam_update_adapter(void *state, float *w, float *g)     { nadam_update((NadamState*)state, w, g); }

// Création dynamique de l'état de l'optimiseur
void *trainer_create_optimizer_state(const char *optimizer, size_t num_weights, float lr) {
    if (strcmp(optimizer, "sgd") == 0)       return sgd_init(num_weights, lr);
    if (strcmp(optimizer, "adam") == 0)      return adam_init(num_weights, lr, 0.9f, 0.999f, 1e-8f);
    if (strcmp(optimizer, "adamw") == 0)     return adamw_init(num_weights, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
    if (strcmp(optimizer, "rmsprop") == 0)   return rmsprop_init(num_weights, lr, 0.99f, 1e-8f);
    if (strcmp(optimizer, "lion") == 0)      return lion_init(num_weights, lr, 0.9f, 0.99f);
    if (strcmp(optimizer, "adabelief") == 0) return adabelief_init(num_weights, lr, 0.9f, 0.999f, 1e-8f);
    if (strcmp(optimizer, "radam") == 0)     return radam_init(num_weights, lr, 0.9f, 0.999f, 1e-8f);
    if (strcmp(optimizer, "adamax") == 0)    return adamax_init(num_weights, lr, 0.9f, 0.999f, 1e-8f);
    if (strcmp(optimizer, "nadam") == 0)     return nadam_init(num_weights, lr, 0.9f, 0.999f, 1e-8f);
    return NULL;
}

// Mapping des fonctions de mise à jour
OptimizerUpdateFn trainer_get_optimizer_update(const char *optimizer) {
    if (strcmp(optimizer, "sgd") == 0)       return sgd_update_adapter;
    if (strcmp(optimizer, "adam") == 0)      return adam_update_adapter;
    if (strcmp(optimizer, "adamw") == 0)     return adamw_update_adapter;
    if (strcmp(optimizer, "rmsprop") == 0)   return rmsprop_update_adapter;
    if (strcmp(optimizer, "lion") == 0)      return lion_update_adapter;
    if (strcmp(optimizer, "adabelief") == 0) return adabelief_update_adapter;
    if (strcmp(optimizer, "radam") == 0)     return radam_update_adapter;
    if (strcmp(optimizer, "adamax") == 0)    return adamax_update_adapter;
    if (strcmp(optimizer, "nadam") == 0)     return nadam_update_adapter;
    return NULL;
}

// Création du trainer
Trainer *trainer_create(NeuralNetwork *net,
                        const char *optimizer,
                        float lr,
                        int epochs,
                        int batch_size,
                        TrainingStrategyFn train_strategy,
                        void *optimizer_state,
                        OptimizerUpdateFn optimizer_update) {
    Trainer *t = malloc(sizeof(Trainer));
    t->net = net;
    t->learning_rate = lr;
    t->epochs = epochs;
    t->batch_size = batch_size;
    t->train_strategy = train_strategy;
    t->optimizer_state = optimizer_state;
    t->optimizer_update = optimizer_update;
    t->progress_bar_id = -1; // Pas de barre de progression par défaut
    strncpy(t->optimizer_name, optimizer, sizeof(t->optimizer_name)-1);
    strncpy(t->strategy_name, "custom", sizeof(t->strategy_name)-1);
    return t;
}

void trainer_free(Trainer *t) {
    if (t) {
        // Libérer l'état de l'optimiseur si il existe
        if (t->optimizer_state) {
            free(t->optimizer_state);
        }
        free(t);
    }
}

void trainer_train(Trainer *t, Dataset *d) {
    if (t->train_strategy)
        t->train_strategy(t, d);
}

float trainer_validate(Trainer *t, Dataset *d) {
    float acc = 0.0f;
    for (size_t i = 0; i < d->num_samples; ++i) {
        network_forward(t->net, d->inputs[i]);
        float *out = network_output(t->net);
        if (out[0] == d->outputs[i][0]) acc += 1.0f; // Pour du binaire
    }
    return acc / d->num_samples;
}