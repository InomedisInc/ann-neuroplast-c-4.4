#ifndef CONFIG_H
#define CONFIG_H

#include <stdbool.h>

typedef struct {
    char *config_file;
    char *mode;
    char *optim_config;
    
    char **neuroplast_methods;
    int num_neuroplast_methods;

    char **optimizers;
    int num_optimizers;

    char **activations;
    int num_activations;

    int seed;
} Config;

// Fonctions pour libérer la mémoire
void free_config(Config *config);
void free_list(char **list, int count);

#endif /* CONFIG_H */