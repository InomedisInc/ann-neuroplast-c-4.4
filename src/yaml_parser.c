#include "config_simple.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Utilitaire pour nettoyer début/fin d'une chaîne
void trim(char *str) {
    int len = strlen(str);
    int start = 0, end = len-1;
    while (isspace(str[start])) start++;
    while (end > start && isspace(str[end])) end--;
    if (start > 0) memmove(str, str+start, end-start+1);
    str[end-start+1] = '\0';
}

int parse_yaml(const char *filename, ConfigSimple *cfg) {
    FILE *f = fopen(filename, "r");
    if (!f) return 0;
    char line[256];
    int list_mode = 0;
    char *current_list = NULL;

    while (fgets(line, sizeof(line), f)) {
        // Supprimer les espaces en début et fin
        char *trimmed = line;
        while (*trimmed == ' ' || *trimmed == '\t') trimmed++;
        
        // Ignorer les commentaires et lignes vides
        if (*trimmed == '#' || *trimmed == '\n' || *trimmed == '\0') continue;

        if (strchr(trimmed, ':')) {
            // Nouvelle clé
            char *colon = strchr(trimmed, ':');
            *colon = '\0';
            char key[MAX_STR], value[MAX_STR];
            strncpy(key, trimmed, MAX_STR-1);
            trim(key);
            strcpy(value, colon+1);
            trim(value);

            // Gestion des listes
            if (strlen(value) == 0) {
                // On passe en mode liste
                list_mode = 1;
                if (strcmp(key, "neuroplast_methods") == 0) current_list = "neuroplast_methods";
                else if (strcmp(key, "optimizers") == 0) current_list = "optimizers";
                else if (strcmp(key, "activations") == 0) current_list = "activations";
                else current_list = NULL;
                continue;
            }

            // Gestion des clés simples
            if (strcmp(key, "dataset") == 0)
                strncpy(cfg->dataset, value, MAX_STR-1);
            else if (strcmp(key, "batch_size") == 0)
                cfg->batch_size = atoi(value);
            else if (strcmp(key, "max_epochs") == 0)
                cfg->max_epochs = atoi(value);
            else if (strcmp(key, "learning_rate") == 0)
                cfg->learning_rate = atof(value);

            list_mode = 0;
        } else if (list_mode && *trimmed == '-' && current_list) {
            // Élément d'une liste
            char *val = trimmed + 1;
            trim(val);
            if (strcmp(current_list, "neuroplast_methods") == 0)
                strncpy(cfg->neuroplast_methods[cfg->num_neuroplast_methods++], val, MAX_STR-1);
            else if (strcmp(current_list, "optimizers") == 0)
                strncpy(cfg->optimizers[cfg->num_optimizers++], val, MAX_STR-1);
            else if (strcmp(current_list, "activations") == 0)
                strncpy(cfg->activations[cfg->num_activations++], val, MAX_STR-1);
        }
    }
    fclose(f);
    return 1;
}

// Affichage pour debug/test
void print_config_simple(ConfigSimple *cfg) {
    printf("Dataset : %s\n", cfg->dataset);
    printf("Neuroplast methods :");
    for (int i = 0; i < cfg->num_neuroplast_methods; i++)
        printf(" %s", cfg->neuroplast_methods[i]);
    printf("\nOptimizers :");
    for (int i = 0; i < cfg->num_optimizers; i++)
        printf(" %s", cfg->optimizers[i]);
    printf("\nActivations :");
    for (int i = 0; i < cfg->num_activations; i++)
        printf(" %s", cfg->activations[i]);
    printf("\nBatch size : %d\n", cfg->batch_size);
    printf("Max epochs : %d\n", cfg->max_epochs);
    printf("Learning rate : %f\n", cfg->learning_rate);
}