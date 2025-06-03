#include "yaml_parser_rich.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE 1024

// Fonction utilitaire pour nettoyer les valeurs (enlever les guillemets)
static void clean_value(char *value) {
    if (!value || !*value) return;
    
    // Enlève les espaces en début et fin
    char *start = value;
    while (*start && isspace((unsigned char)*start)) start++;
    
    if (start != value) {
        memmove(value, start, strlen(start) + 1);
    }
    
    size_t len = strlen(value);
    if (len == 0) return;
    
    char *end = value + len - 1;
    while (end > value && isspace((unsigned char)*end)) {
        *end = '\0';
        end--;
    }
    
    // Enlève les guillemets au début s'il y en a
    if (value[0] == '"') {
        memmove(value, value + 1, strlen(value));
    }
    
    // Enlève les guillemets à la fin s'il y en a
    len = strlen(value);
    if (len > 0 && value[len - 1] == '"') {
        value[len - 1] = '\0';
    }
}

// Fonction pour lire un fichier dataset référencé dans le fichier principal
static int read_dataset_dimensions(const char *dataset_file, RichConfig *cfg) {
    char full_path[512];
    
    // Si le chemin est déjà absolu
    if (dataset_file[0] == '/') {
        strncpy(full_path, dataset_file, sizeof(full_path) - 1);
    } else {
        // Sinon, on considère que c'est relatif au répertoire courant
        snprintf(full_path, sizeof(full_path), "%s", dataset_file);
    }
    
    FILE *file = fopen(full_path, "r");
    if (!file) {
        printf("Erreur: Impossible d'ouvrir le fichier dataset %s\n", full_path);
        return 0;
    }
    
    char line[MAX_LINE];
    while (fgets(line, sizeof(line), file)) {
        // Retire le retour à la ligne
        line[strcspn(line, "\n")] = 0;
        
        // Ignore les lignes vides et commentaires
        if (line[0] == 0 || line[0] == '#') continue;
        
        // Cherche les dimensions
        if (strstr(line, "input_cols:")) {
            char *value = strchr(line, ':') + 1;
            while (*value && isspace(*value)) value++;
            cfg->input_cols = atoi(value);
        }
        else if (strstr(line, "output_cols:")) {
            char *value = strchr(line, ':') + 1;
            while (*value && isspace(*value)) value++;
            cfg->output_cols = atoi(value);
        }
        else if (strstr(line, "batch_size:")) {
            char *value = strchr(line, ':') + 1;
            while (*value && isspace(*value)) value++;
            cfg->batch_size = atoi(value);
        }
        else if (strstr(line, "dataset:")) {
            char *value = strchr(line, ':') + 1;
            while (*value && isspace(*value)) value++;
            strncpy(cfg->dataset, value, sizeof(cfg->dataset) - 1);
            cfg->dataset[sizeof(cfg->dataset) - 1] = '\0';
            clean_value(cfg->dataset);
        }
        else if (strstr(line, "max_epochs:")) {
            char *value = strchr(line, ':') + 1;
            while (*value && isspace(*value)) value++;
            cfg->max_epochs = atoi(value);
        }
        else if (strstr(line, "learning_rate:")) {
            char *value = strchr(line, ':') + 1;
            while (*value && isspace(*value)) value++;
            cfg->learning_rate = atof(value);
        }
        else if (strstr(line, "early_stopping:")) {
            char *value = strchr(line, ':') + 1;
            while (*value && isspace(*value)) value++;
            // Parsing booléen : true/false, 1/0, yes/no
            if (strstr(value, "true") || strstr(value, "yes") || strstr(value, "1")) {
                cfg->early_stopping = 1;
            } else {
                cfg->early_stopping = 0;
            }
        }
        else if (strstr(line, "patience:")) {
            char *value = strchr(line, ':') + 1;
            while (*value && isspace(*value)) value++;
            cfg->patience = atoi(value);
        }
    }
    
    fclose(file);
    
    return (cfg->input_cols > 0 && cfg->output_cols > 0);
}

int parse_yaml_rich(const char *filename, RichConfig *cfg) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Erreur: Impossible d'ouvrir le fichier %s\n", filename);
        return 0;
    }
    
    // Initialisation des valeurs par défaut
    memset(cfg, 0, sizeof(RichConfig));
    cfg->batch_size = 32;
    cfg->max_epochs = 100;
    cfg->learning_rate = 0.001f;
    cfg->early_stopping = 1;
    cfg->patience = 20;
    cfg->optimized_parameters = 0;
    cfg->input_cols = 10;
    cfg->output_cols = 1;
    cfg->image_width = 128;
    cfg->image_height = 128;
    cfg->image_channels = 3;
    
    char line[MAX_LINE];
    
    // Parser simple ligne par ligne
    while (fgets(line, sizeof(line), file)) {
        // Retire le retour à la ligne
        line[strcspn(line, "\n")] = 0;
        
        // Ignore les lignes vides et commentaires
        if (line[0] == 0 || line[0] == '#') continue;
        
        // Parser les valeurs simples uniquement
        if (strchr(line, ':') && line[0] != '-') {
            char *colon = strchr(line, ':');
            if (colon) {
                *colon = '\0';
                char key[256], value[256];
                strncpy(key, line, sizeof(key) - 1);
                key[sizeof(key) - 1] = '\0';
                strncpy(value, colon + 1, sizeof(value) - 1);
                value[sizeof(value) - 1] = '\0';
                
                // Nettoyer les espaces
                char *k = key;
                while (*k && isspace(*k)) k++;
                char *v = value;
                while (*v && isspace(*v)) v++;
                
                // Supprimer les espaces en fin
                char *end = k + strlen(k) - 1;
                while (end > k && isspace(*end)) *end-- = '\0';
                end = v + strlen(v) - 1;
                while (end > v && isspace(*end)) *end-- = '\0';
                
                // Parser les valeurs selon la clé
                if (strcmp(k, "max_epochs") == 0) {
                    cfg->max_epochs = atoi(v);
                }
                else if (strcmp(k, "batch_size") == 0) {
                    cfg->batch_size = atoi(v);
                }
                else if (strcmp(k, "learning_rate") == 0) {
                    cfg->learning_rate = atof(v);
                }
                else if (strcmp(k, "patience") == 0) {
                    cfg->patience = atoi(v);
                }
                else if (strcmp(k, "train_test_split") == 0) {
                    cfg->train_test_split = atof(v);
                }
                else if (strcmp(k, "early_stopping") == 0) {
                    cfg->early_stopping = (strstr(v, "true") || strstr(v, "yes") || strstr(v, "1")) ? 1 : 0;
                }
                else if (strcmp(k, "optimized_parameters") == 0) {
                    cfg->optimized_parameters = (strstr(v, "true") || strstr(v, "yes") || strstr(v, "1")) ? 1 : 0;
                }
                else if (strcmp(k, "is_image_dataset") == 0) {
                    cfg->is_image_dataset = (strstr(v, "true") || strstr(v, "yes") || strstr(v, "1")) ? 1 : 0;
                }
                else if (strcmp(k, "image_train_dir") == 0) {
                    strncpy(cfg->image_train_dir, v, sizeof(cfg->image_train_dir) - 1);
                    cfg->image_train_dir[sizeof(cfg->image_train_dir) - 1] = '\0';
                    clean_value(cfg->image_train_dir);
                }
                else if (strcmp(k, "image_test_dir") == 0) {
                    strncpy(cfg->image_test_dir, v, sizeof(cfg->image_test_dir) - 1);
                    cfg->image_test_dir[sizeof(cfg->image_test_dir) - 1] = '\0';
                    clean_value(cfg->image_test_dir);
                }
                else if (strcmp(k, "image_val_dir") == 0) {
                    strncpy(cfg->image_val_dir, v, sizeof(cfg->image_val_dir) - 1);
                    cfg->image_val_dir[sizeof(cfg->image_val_dir) - 1] = '\0';
                    clean_value(cfg->image_val_dir);
                }
                else if (strcmp(k, "image_width") == 0) {
                    cfg->image_width = atoi(v);
                }
                else if (strcmp(k, "image_height") == 0) {
                    cfg->image_height = atoi(v);
                }
                else if (strcmp(k, "image_channels") == 0) {
                    cfg->image_channels = atoi(v);
                }
                else if (strcmp(k, "input_cols") == 0) {
                    cfg->input_cols = atoi(v);
                }
                else if (strcmp(k, "output_cols") == 0) {
                    cfg->output_cols = atoi(v);
                }
                else if (strcmp(k, "dataset") == 0) {
                    strncpy(cfg->dataset, v, sizeof(cfg->dataset) - 1);
                    cfg->dataset[sizeof(cfg->dataset) - 1] = '\0';
                    clean_value(cfg->dataset);
                }
                else if (strcmp(k, "dataset_name") == 0) {
                    strncpy(cfg->dataset_name, v, sizeof(cfg->dataset_name) - 1);
                    cfg->dataset_name[sizeof(cfg->dataset_name) - 1] = '\0';
                    clean_value(cfg->dataset_name);
                }
                
                *colon = ':';  // Restaurer le caractère original
            }
        }
    }
    
    fclose(file);
    
    // Ajouter les méthodes par défaut si nécessaire
    if (cfg->num_neuroplast_methods == 0) {
        cfg->num_neuroplast_methods = 1;
        strcpy(cfg->neuroplast_methods[0].name, "standard");
    }
    
    if (cfg->num_activations == 0) {
        cfg->num_activations = 1;
        strcpy(cfg->activations[0].name, "relu");
    }
    
    if (cfg->num_optimizers == 0) {
        cfg->num_optimizers = 1;
        strcpy(cfg->optimizers[0].name, "adam");
    }
    
    return 1;
}