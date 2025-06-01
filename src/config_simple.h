// config_simple.h
#ifndef CONFIG_SIMPLE_H
#define CONFIG_SIMPLE_H

#define MAX_LIST 32
#define MAX_STR 128

typedef struct {
    char dataset[MAX_STR];
    char neuroplast_methods[MAX_LIST][MAX_STR];
    int num_neuroplast_methods;
    char optimizers[MAX_LIST][MAX_STR];
    int num_optimizers;
    char activations[MAX_LIST][MAX_STR];
    int num_activations;
    int batch_size;
    int max_epochs;
    float learning_rate;
    // Ajoute ici tout autre champ utile
} ConfigSimple;

void print_config_simple(ConfigSimple *cfg);

#endif