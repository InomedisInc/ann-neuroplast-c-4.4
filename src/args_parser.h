#ifndef ARGS_PARSER_H
#define ARGS_PARSER_H

#include <stdbool.h>
#include "rich_config.h"

#define MAX_METHODS_CLI 32
#define MAX_METHOD_NAME 64

typedef enum {
    MODE_DEFAULT,
    MODE_COMPARE_ALL_METHODS,
    MODE_SINGLE_EXPERIMENT,
    MODE_HYPERPARAMETER_SEARCH,
    MODE_NORMAL = MODE_DEFAULT,
    MODE_TEST_HEART_DISEASE,
    MODE_TEST_ENHANCED,
    MODE_TEST_ROBUST,
    MODE_TEST_OPTIMIZED_METRICS,
    MODE_TEST_ALL_ACTIVATIONS,
    MODE_TEST_ALL_OPTIMIZERS,
    MODE_TEST_NEUROPLAST_METHODS,
    MODE_TEST_COMPLETE_COMBINATIONS,
    MODE_TEST_BENCHMARK_FULL,
    MODE_TEST_ALL
} RunMode;

typedef struct {
    char config_path[256];
    char optim_config_path[256];
    RunMode mode;
    unsigned int seed;
    
    // Méthodes spécifiées en ligne de commande
    char neuroplast_methods[MAX_METHODS_CLI][MAX_METHOD_NAME];
    int num_neuroplast_methods;
    
    char optimizers[MAX_METHODS_CLI][MAX_METHOD_NAME];
    int num_optimizers;
    
    char activations[MAX_METHODS_CLI][MAX_METHOD_NAME];
    int num_activations;
    
    bool has_config;
    bool has_optim_config;
    bool has_mode;
    bool has_seed;
    bool has_neuroplast_methods;
    bool has_optimizers;
    bool has_activations;
} CommandLineArgs;

// Initialise la structure avec des valeurs par défaut
void args_init(CommandLineArgs *args);

// Parse les arguments de la ligne de commande
bool args_parse(int argc, char **argv, CommandLineArgs *args);

// Fusionne les arguments CLI avec la config YAML
void args_merge_config(const CommandLineArgs *args, RichConfig *cfg);

// Affiche l'aide
void print_usage(void);

// Convertit une chaîne en mode d'exécution
RunMode string_to_mode(const char *mode_str);

// Convertit un mode d'exécution en chaîne
const char *mode_to_string(RunMode mode);

#endif /* ARGS_PARSER_H */