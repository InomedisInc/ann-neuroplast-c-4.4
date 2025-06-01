#include "args_parser.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

// Structure interne pour les options de ligne de commande
typedef struct {
    const char *name;
    const char *description;
    bool requires_value;
} Option;

static const Option OPTIONS[] = {
    {"--config", "Chemin vers le fichier de configuration principal", true},
    {"--mode", "Mode d'exécution (compare_all_methods, single_experiment, hyperparameter_search)", true},
    {"--optim_config", "Chemin vers le fichier de configuration d'optimisation", true},
    {"--neuroplast_methods", "Liste des méthodes neuroplastiques à utiliser", true},
    {"--optimizers", "Liste des optimiseurs à utiliser", true},
    {"--activations", "Liste des fonctions d'activation à utiliser", true},
    {"--seed", "Graine pour la génération de nombres aléatoires", true},
    {"--help", "Affiche ce message d'aide", false},
    {"-h", "Affiche ce message d'aide", false}
};

// Initialisation des arguments par défaut
void args_init(CommandLineArgs *args) {
    if (!args) return;  // Protection contre les pointeurs NULL
    
    memset(args, 0, sizeof(CommandLineArgs));
    strncpy(args->config_path, "config/default.yml", sizeof(args->config_path)-1);
    args->config_path[sizeof(args->config_path)-1] = '\0';
    args->mode = MODE_DEFAULT;
    args->seed = 42;
    args->has_config = false;
    args->has_optim_config = false;
    args->has_mode = false;
    args->has_seed = false;
    args->has_neuroplast_methods = false;
    args->has_optimizers = false;
    args->has_activations = false;
}

// Fonction utilitaire pour parser une liste d'éléments séparés par des espaces
static bool parse_space_separated_list(const char *input, char output[][MAX_METHOD_NAME], int *count) {
    if (!input || !output || !count) return false;  // Protection contre les pointeurs NULL
    
    *count = 0;
    char *str = strdup(input);
    if (!str) return false;  // Protection contre l'échec de l'allocation
    
    char *token = strtok(str, " ");
    while (token && *count < MAX_METHODS_CLI) {
        strncpy(output[*count], token, MAX_METHOD_NAME-1);
        output[*count][MAX_METHOD_NAME-1] = '\0';
        (*count)++;
        token = strtok(NULL, " ");
    }
    
    free(str);
    return true;
}

RunMode string_to_mode(const char *mode_str) {
    if (!mode_str) return MODE_DEFAULT;  // Protection contre les pointeurs NULL
    
    if (strcmp(mode_str, "compare_all_methods") == 0) return MODE_COMPARE_ALL_METHODS;
    if (strcmp(mode_str, "single_experiment") == 0) return MODE_SINGLE_EXPERIMENT;
    if (strcmp(mode_str, "hyperparameter_search") == 0) return MODE_HYPERPARAMETER_SEARCH;
    return MODE_DEFAULT;
}

const char *mode_to_string(RunMode mode) {
    switch (mode) {
        case MODE_COMPARE_ALL_METHODS: return "compare_all_methods";
        case MODE_SINGLE_EXPERIMENT: return "single_experiment";
        case MODE_HYPERPARAMETER_SEARCH: return "hyperparameter_search";
        default: return "default";
    }
}

void print_usage(void) {
    printf("Usage: neuroplast-ann [options]\n\n");
    printf("Options disponibles:\n");
    
    for (size_t i = 0; i < sizeof(OPTIONS) / sizeof(OPTIONS[0]); i++) {
        printf("  %-20s %s%s\n", 
               OPTIONS[i].name, 
               OPTIONS[i].description,
               OPTIONS[i].requires_value ? " (requiert une valeur)" : "");
    }
    
    printf("\nExemple:\n");
    printf("  neuroplast-ann --config config/comprehensive_comparison.yml \\\n");
    printf("                 --mode compare_all_methods \\\n");
    printf("                 --optim_config config/heart_attack.yml \\\n");
    printf("                 --neuroplast_methods \"standard adaptive advanced bayesian progressive swarm propagation\" \\\n");
    printf("                 --optimizers \"adamw adam sgd rmsprop lion adabelief radam adamax nadam\" \\\n");
    printf("                 --activations \"NeuroPlast ReLU LeakyReLU GELU Sigmoid ELU Mish Swish PReLU\" \\\n");
    printf("                 --seed 42\n");
}

// Fonction utilitaire pour trouver une option
static const Option *find_option(const char *arg) {
    for (size_t i = 0; i < sizeof(OPTIONS) / sizeof(OPTIONS[0]); i++) {
        if (strcmp(OPTIONS[i].name, arg) == 0) {
            return &OPTIONS[i];
        }
    }
    return NULL;
}

bool args_parse(int argc, char **argv, CommandLineArgs *args) {
    if (!args || !argv) return false;  // Protection contre les pointeurs NULL
    
    args_init(args);
    
    for (int i = 1; i < argc; i++) {
        const Option *opt = find_option(argv[i]);
        
        if (!opt) {
            printf("Option inconnue: %s\n", argv[i]);
            print_usage();
            return false;
        }
        
        if (!opt->requires_value) {
            if (strcmp(opt->name, "--help") == 0 || strcmp(opt->name, "-h") == 0) {
                print_usage();
                return false;
            }
            continue;
        }
        
        if (i + 1 >= argc) {
            printf("Erreur: valeur manquante pour l'option %s\n", argv[i]);
            print_usage();
            return false;
        }
        
        const char *value = argv[++i];
        
        if (strcmp(opt->name, "--config") == 0) {
            strncpy(args->config_path, value, sizeof(args->config_path)-1);
            args->config_path[sizeof(args->config_path)-1] = '\0';
            args->has_config = true;
        }
        else if (strcmp(opt->name, "--mode") == 0) {
            args->mode = string_to_mode(value);
            args->has_mode = true;
        }
        else if (strcmp(opt->name, "--optim_config") == 0) {
            strncpy(args->optim_config_path, value, sizeof(args->optim_config_path)-1);
            args->optim_config_path[sizeof(args->optim_config_path)-1] = '\0';
            args->has_optim_config = true;
        }
        else if (strcmp(opt->name, "--neuroplast_methods") == 0) {
            if (!parse_space_separated_list(value, args->neuroplast_methods, &args->num_neuroplast_methods)) {
                printf("Erreur: impossible de parser la liste des méthodes neuroplastiques\n");
                return false;
            }
            args->has_neuroplast_methods = true;
        }
        else if (strcmp(opt->name, "--optimizers") == 0) {
            if (!parse_space_separated_list(value, args->optimizers, &args->num_optimizers)) {
                printf("Erreur: impossible de parser la liste des optimiseurs\n");
                return false;
            }
            args->has_optimizers = true;
        }
        else if (strcmp(opt->name, "--activations") == 0) {
            if (!parse_space_separated_list(value, args->activations, &args->num_activations)) {
                printf("Erreur: impossible de parser la liste des activations\n");
                return false;
            }
            args->has_activations = true;
        }
        else if (strcmp(opt->name, "--seed") == 0) {
            char *endptr;
            long seed = strtol(value, &endptr, 10);
            if (*endptr != '\0' || seed < 0) {
                printf("Erreur: la graine doit être un nombre entier positif\n");
                return false;
            }
            args->seed = (unsigned int)seed;
            args->has_seed = true;
        }
    }
    
    return true;
}

void args_merge_config(const CommandLineArgs *args, RichConfig *cfg) {
    // Si des méthodes neuroplastiques sont spécifiées en ligne de commande
    if (args->has_neuroplast_methods) {
        cfg->num_neuroplast_methods = args->num_neuroplast_methods;
        for (int i = 0; i < args->num_neuroplast_methods; i++) {
            strncpy(cfg->neuroplast_methods[i].name, args->neuroplast_methods[i], MAX_NAME-1);
            cfg->neuroplast_methods[i].name[MAX_NAME-1] = '\0';
        }
    }
    
    // Si des optimiseurs sont spécifiés en ligne de commande
    if (args->has_optimizers) {
        cfg->num_optimizers = args->num_optimizers;
        for (int i = 0; i < args->num_optimizers; i++) {
            strncpy(cfg->optimizers[i].name, args->optimizers[i], MAX_NAME-1);
            cfg->optimizers[i].name[MAX_NAME-1] = '\0';
            cfg->optimizers[i].num_params = 0;  // Réinitialise les paramètres
        }
    }
    
    // Si des activations sont spécifiées en ligne de commande
    if (args->has_activations) {
        cfg->num_activations = args->num_activations;
        for (int i = 0; i < args->num_activations; i++) {
            strncpy(cfg->activations[i].name, args->activations[i], MAX_NAME-1);
            cfg->activations[i].name[MAX_NAME-1] = '\0';
            cfg->activations[i].optimization_method[0] = '\0';
            cfg->activations[i].optimized_with[0] = '\0';
            cfg->activations[i].num_params = 0;  // Réinitialise les paramètres
        }
    }
}