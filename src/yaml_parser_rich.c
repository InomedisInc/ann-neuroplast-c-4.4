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
    cfg->batch_size = 32;  // Valeurs par défaut
    cfg->max_epochs = 100; // Base 100 époques comme demandé
    cfg->learning_rate = 0.001f;
    cfg->early_stopping = 1;  // Early stopping activé par défaut
    cfg->patience = 20;       // Patience par défaut de 20 époques
    cfg->optimized_parameters = 0;  // Configuration statique par défaut
    
    char line[MAX_LINE];
    
    // Première passe: trouver le base_dataset
    while (fgets(line, sizeof(line), file)) {
        // Retire le retour à la ligne
        line[strcspn(line, "\n")] = 0;
        
        // Ignore les lignes vides et commentaires
        if (line[0] == 0 || line[0] == '#') continue;
        
        // Cherche base_dataset
        if (strstr(line, "base_dataset:")) {
            char *value = strchr(line, ':') + 1;
            while (*value && isspace(*value)) value++;
            strncpy(cfg->base_dataset, value, sizeof(cfg->base_dataset) - 1);
            cfg->base_dataset[sizeof(cfg->base_dataset) - 1] = '\0';
            clean_value(cfg->base_dataset);
        }
        
        // Cherche dataset
        if (strstr(line, "dataset:")) {
            char *value = strchr(line, ':') + 1;
            while (*value && isspace(*value)) value++;
            strncpy(cfg->dataset, value, sizeof(cfg->dataset) - 1);
            cfg->dataset[sizeof(cfg->dataset) - 1] = '\0';
            clean_value(cfg->dataset);
        }
    }
    
    // Si on a trouvé un fichier base_dataset, le charger pour obtenir les dimensions
    if (cfg->base_dataset[0] != '\0') {
        if (!read_dataset_dimensions(cfg->base_dataset, cfg)) {
            printf("Avertissement: Impossible de lire les dimensions depuis %s\n", cfg->base_dataset);
        }
    }
    
    // Revenir au début du fichier pour une deuxième passe
    rewind(file);
    
    // Section courante lors du parsing
    enum {
        SECTION_NONE,
        SECTION_NEUROPLAST_METHODS,
        SECTION_ACTIVATIONS,
        SECTION_OPTIMIZERS,
        SECTION_METRICS,
        SECTION_TRAINING
    } current_section = SECTION_NONE;
    
    // Index courants pour les sections
    int method_idx = -1;
    int activation_idx = -1;
    int optimizer_idx = -1;
    int metric_idx = -1;
    
    // Deuxième passe: parser toutes les données
    while (fgets(line, sizeof(line), file)) {
        // Retire le retour à la ligne
        line[strcspn(line, "\n")] = 0;
        
        // Ignore les lignes vides et commentaires
        if (line[0] == 0 || line[0] == '#') continue;
        
        // Détecte les sections
        if (strstr(line, "neuroplast_methods:")) {
            current_section = SECTION_NEUROPLAST_METHODS;
            continue;
        }
        else if (strstr(line, "activations:")) {
            current_section = SECTION_ACTIVATIONS;
            continue;
        }
        else if (strstr(line, "optimizers:")) {
            current_section = SECTION_OPTIMIZERS;
            continue;
        }
        else if (strstr(line, "metrics:")) {
            current_section = SECTION_METRICS;
            continue;
        }
        else if (strstr(line, "training:")) {
            current_section = SECTION_TRAINING;
            continue;
        }
        
        // Détecte les entrées de liste avec un tiret
        if (line[0] == '-') {
            char *name_start = strchr(line, '-') + 1;
            
            // Skip les espaces après le tiret
            while (*name_start && isspace(*name_start)) name_start++;
            
            if (*name_start == 0) continue; // Ligne vide
            
            // Si on a "- name:" ou juste "- valeur"
            if (strstr(name_start, "name:")) {
                char *name_value = strchr(name_start, ':') + 1;
                while (*name_value && isspace(*name_value)) name_value++;
                
                switch (current_section) {
                    case SECTION_NEUROPLAST_METHODS:
                        if (cfg->num_neuroplast_methods < MAX_METHODS) {
                            method_idx = cfg->num_neuroplast_methods++;
                            strncpy(cfg->neuroplast_methods[method_idx].name, name_value, MAX_NAME - 1);
                            cfg->neuroplast_methods[method_idx].name[MAX_NAME - 1] = '\0';
                        }
                        break;
                    case SECTION_ACTIVATIONS:
                        if (cfg->num_activations < MAX_METHODS) {
                            activation_idx = cfg->num_activations++;
                            strncpy(cfg->activations[activation_idx].name, name_value, MAX_NAME - 1);
                            cfg->activations[activation_idx].name[MAX_NAME - 1] = '\0';
                        }
                        break;
                    case SECTION_OPTIMIZERS:
                        if (cfg->num_optimizers < MAX_METHODS) {
                            optimizer_idx = cfg->num_optimizers++;
                            strncpy(cfg->optimizers[optimizer_idx].name, name_value, MAX_NAME - 1);
                            cfg->optimizers[optimizer_idx].name[MAX_NAME - 1] = '\0';
                        }
                        break;
                    case SECTION_METRICS:
                        if (cfg->num_metrics < MAX_METHODS) {
                            metric_idx = cfg->num_metrics++;
                            strncpy(cfg->metrics[metric_idx].name, name_value, MAX_NAME - 1);
                            cfg->metrics[metric_idx].name[MAX_NAME - 1] = '\0';
                        }
                        break;
                    default:
                        break;
                }
            }
            // Si c'est juste "- valeur" sans "name:"
            else {
                switch (current_section) {
                    case SECTION_NEUROPLAST_METHODS:
                        if (cfg->num_neuroplast_methods < MAX_METHODS) {
                            method_idx = cfg->num_neuroplast_methods++;
                            strncpy(cfg->neuroplast_methods[method_idx].name, name_start, MAX_NAME - 1);
                            cfg->neuroplast_methods[method_idx].name[MAX_NAME - 1] = '\0';
                        }
                        break;
                    case SECTION_ACTIVATIONS:
                        if (cfg->num_activations < MAX_METHODS) {
                            activation_idx = cfg->num_activations++;
                            strncpy(cfg->activations[activation_idx].name, name_start, MAX_NAME - 1);
                            cfg->activations[activation_idx].name[MAX_NAME - 1] = '\0';
                        }
                        break;
                    case SECTION_OPTIMIZERS:
                        if (cfg->num_optimizers < MAX_METHODS) {
                            optimizer_idx = cfg->num_optimizers++;
                            strncpy(cfg->optimizers[optimizer_idx].name, name_start, MAX_NAME - 1);
                            cfg->optimizers[optimizer_idx].name[MAX_NAME - 1] = '\0';
                        }
                        break;
                    case SECTION_METRICS:
                        if (cfg->num_metrics < MAX_METHODS) {
                            metric_idx = cfg->num_metrics++;
                            strncpy(cfg->metrics[metric_idx].name, name_start, MAX_NAME - 1);
                            cfg->metrics[metric_idx].name[MAX_NAME - 1] = '\0';
                        }
                        break;
                    default:
                        break;
                }
            }
            
            continue;
        }
        
        // Parser les valeurs simples (non-listes) pour toutes les sections
        if (strchr(line, ':') && line[0] != '-') {
            char *colon = strchr(line, ':');
            if (colon) {
                *colon = '\0';
                char key[MAX_NAME], value[MAX_NAME];
                strncpy(key, line, MAX_NAME - 1);
                key[MAX_NAME - 1] = '\0';
                strcpy(value, colon + 1);
                
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
                    // Parsing booléen : true/false, 1/0, yes/no
                    if (strstr(v, "true") || strstr(v, "yes") || strstr(v, "1")) {
                        cfg->early_stopping = 1;
                    } else {
                        cfg->early_stopping = 0;
                    }
                }
                else if (strcmp(k, "optimized_parameters") == 0) {
                    // Parsing booléen : true/false, 1/0, yes/no
                    if (strstr(v, "true") || strstr(v, "yes") || strstr(v, "1")) {
                        cfg->optimized_parameters = 1;
                    } else {
                        cfg->optimized_parameters = 0;
                    }
                }
                else if (strcmp(k, "is_image_dataset") == 0) {
                    // Parsing booléen : true/false, 1/0, yes/no
                    if (strstr(v, "true") || strstr(v, "yes") || strstr(v, "1")) {
                        cfg->is_image_dataset = 1;
                    } else {
                        cfg->is_image_dataset = 0;
                    }
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
                
                *colon = ':';  // Restaurer le caractère original
            }
            continue;
        }
        
        // Cas spécial: si on n'a trouvé aucun configurateur, on ajoute les méthodes par défaut
        if (cfg->num_neuroplast_methods == 0) {
            cfg->num_neuroplast_methods = 1;
            strcpy(cfg->neuroplast_methods[0].name, "standard");
        }
        
        if (cfg->num_activations == 0) {
            cfg->num_activations = 1;
            strcpy(cfg->activations[0].name, "sigmoid");
        }
        
        if (cfg->num_optimizers == 0) {
            cfg->num_optimizers = 1;
            strcpy(cfg->optimizers[0].name, "adam");
        }
    }
    
    fclose(file);
    
    // Si aucune dimension n'a été trouvée, utiliser des valeurs par défaut
    if (cfg->input_cols == 0) cfg->input_cols = 10;  // Valeur par défaut
    if (cfg->output_cols == 0) cfg->output_cols = 1; // Valeur par défaut binaire
    
    return 1;
}