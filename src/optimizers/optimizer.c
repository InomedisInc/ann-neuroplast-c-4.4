#include "optimizer.h"
#include <stdlib.h>
#include <string.h>

Optimizer *optimizer_create(const char *name, Param *params, int num_params, float lr) {
    Optimizer *opt = malloc(sizeof(Optimizer));
    opt->learning_rate = lr;
    
    if (strcmp(name, "sgd") == 0) opt->type = OPTIMIZER_SGD;
    else if (strcmp(name, "adam") == 0) opt->type = OPTIMIZER_ADAM;
    else if (strcmp(name, "adamw") == 0) opt->type = OPTIMIZER_ADAMW;
    else if (strcmp(name, "rmsprop") == 0) opt->type = OPTIMIZER_RMSPROP;
    else if (strcmp(name, "lion") == 0) opt->type = OPTIMIZER_LION;
    else if (strcmp(name, "adabelief") == 0) opt->type = OPTIMIZER_ADABELIEF;
    else if (strcmp(name, "radam") == 0) opt->type = OPTIMIZER_RADAM;
    else if (strcmp(name, "adamax") == 0) opt->type = OPTIMIZER_ADAMAX;
    else if (strcmp(name, "nadam") == 0) opt->type = OPTIMIZER_NADAM;
    else opt->type = OPTIMIZER_SGD; // Par défaut
    
    // TODO: Utiliser params et num_params pour configurer l'optimiseur
    (void)params;
    (void)num_params;
    
    return opt;
}

void optimizer_free(Optimizer *opt) {
    free(opt);
}