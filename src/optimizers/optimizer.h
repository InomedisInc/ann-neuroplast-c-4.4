#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "sgd.h"
#include "adam.h"
#include "adamw.h"
#include "rmsprop.h"
#include "lion.h"
#include "adabelief.h"
#include "radam.h"
#include "adamax.h"
#include "nadam.h"
#include "../rich_config.h"

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM,
    OPTIMIZER_ADAMW,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_LION,
    OPTIMIZER_ADABELIEF,
    OPTIMIZER_RADAM,
    OPTIMIZER_ADAMAX,
    OPTIMIZER_NADAM
} OptimizerType;

// Structure générique pour un optimiseur (pour extension future)
typedef struct {
    OptimizerType type;
    float learning_rate;
    // Ajouter ici d’autres champs pour les moments, etc.
} Optimizer;

Optimizer *optimizer_create(const char *name, Param *params, int num_params, float lr);
void optimizer_free(Optimizer *opt);


#endif /* OPTIMIZER_H */