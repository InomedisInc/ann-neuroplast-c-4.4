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
        else if (strstr(line, "debug_mode:")) {
            char *value = strchr(line, ':') + 1;
            while (*value && isspace(*value)) value++;
            // Parsing booléen : true/false, 1/0, yes/no
            if (strstr(value, "true") || strstr(value, "yes") || strstr(value, "1")) {
                cfg->debug_mode = 1;
            } else {
                cfg->debug_mode = 0;
            }
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
    cfg->debug_mode = 0;               // Messages debug masqués par défaut
    cfg->input_cols = 10;
    cfg->output_cols = 1;
    cfg->image_width = 128;
    cfg->image_height = 128;
    cfg->image_channels = 3;
    
    char line[MAX_LINE];
    char current_list_type[64] = {0};  // Type de liste en cours de lecture
    
    // Parser ligne par ligne avec support des listes
    while (fgets(line, sizeof(line), file)) {
        // Retire le retour à la ligne
        line[strcspn(line, "\n")] = 0;
        
        // Ignore les lignes vides et commentaires
        if (line[0] == 0 || line[0] == '#') continue;
        
        // Détection d'un élément de liste (commence par -)
        if (line[0] == ' ' && strstr(line, "- ")) {
            char *item_start = strstr(line, "- ") + 2;
            while (*item_start && isspace(*item_start)) item_start++;
            
            // Supprimer les espaces en fin
            char *end = item_start + strlen(item_start) - 1;
            while (end > item_start && isspace(*end)) *end-- = '\0';
            
            // Ajouter l'élément selon le type de liste
            if (strcmp(current_list_type, "neuroplast_methods") == 0 && cfg->num_neuroplast_methods < MAX_METHODS) {
                strncpy(cfg->neuroplast_methods[cfg->num_neuroplast_methods].name, item_start, MAX_NAME - 1);
                cfg->neuroplast_methods[cfg->num_neuroplast_methods].name[MAX_NAME - 1] = '\0';
                cfg->num_neuroplast_methods++;
            }
            else if (strcmp(current_list_type, "activations") == 0 && cfg->num_activations < MAX_METHODS) {
                strncpy(cfg->activations[cfg->num_activations].name, item_start, MAX_NAME - 1);
                cfg->activations[cfg->num_activations].name[MAX_NAME - 1] = '\0';
                cfg->activations[cfg->num_activations].optimization_method[0] = '\0';
                cfg->activations[cfg->num_activations].optimized_with[0] = '\0';
                cfg->activations[cfg->num_activations].num_params = 0;
                cfg->num_activations++;
            }
            else if (strcmp(current_list_type, "optimizers") == 0 && cfg->num_optimizers < MAX_METHODS) {
                strncpy(cfg->optimizers[cfg->num_optimizers].name, item_start, MAX_NAME - 1);
                cfg->optimizers[cfg->num_optimizers].name[MAX_NAME - 1] = '\0';
                cfg->optimizers[cfg->num_optimizers].num_params = 0;
                cfg->num_optimizers++;
            }
            else if (strcmp(current_list_type, "metrics") == 0 && cfg->num_metrics < MAX_METHODS) {
                strncpy(cfg->metrics[cfg->num_metrics].name, item_start, MAX_NAME - 1);
                cfg->metrics[cfg->num_metrics].name[MAX_NAME - 1] = '\0';
                cfg->num_metrics++;
            }
            continue;
        }
        
        // Parser les valeurs simples et détecter les en-têtes de listes
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
                
                // Détecter les en-têtes de listes
                if (strcmp(k, "neuroplast_methods") == 0) {
                    strcpy(current_list_type, "neuroplast_methods");
                    cfg->num_neuroplast_methods = 0;  // Reset pour relire
                }
                else if (strcmp(k, "activations") == 0) {
                    strcpy(current_list_type, "activations");
                    cfg->num_activations = 0;  // Reset pour relire
                }
                else if (strcmp(k, "optimizers") == 0) {
                    strcpy(current_list_type, "optimizers");
                    cfg->num_optimizers = 0;  // Reset pour relire
                }
                else if (strcmp(k, "metrics") == 0) {
                    strcpy(current_list_type, "metrics");
                    cfg->num_metrics = 0;  // Reset pour relire
                }
                // Parser les valeurs selon la clé
                else if (strcmp(k, "max_epochs") == 0) {
                    cfg->max_epochs = atoi(v);
                    current_list_type[0] = '\0';  // Sortir du mode liste
                }
                else if (strcmp(k, "batch_size") == 0) {
                    cfg->batch_size = atoi(v);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "learning_rate") == 0) {
                    cfg->learning_rate = atof(v);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "patience") == 0) {
                    cfg->patience = atoi(v);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "train_test_split") == 0) {
                    cfg->train_test_split = atof(v);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "early_stopping") == 0) {
                    cfg->early_stopping = (strstr(v, "true") || strstr(v, "yes") || strstr(v, "1")) ? 1 : 0;
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "optimized_parameters") == 0) {
                    cfg->optimized_parameters = (strstr(v, "true") || strstr(v, "yes") || strstr(v, "1")) ? 1 : 0;
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "debug_mode") == 0) {
                    cfg->debug_mode = (strstr(v, "true") || strstr(v, "yes") || strstr(v, "1")) ? 1 : 0;
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "is_image_dataset") == 0) {
                    cfg->is_image_dataset = (strstr(v, "true") || strstr(v, "yes") || strstr(v, "1")) ? 1 : 0;
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "image_train_dir") == 0) {
                    strncpy(cfg->image_train_dir, v, sizeof(cfg->image_train_dir) - 1);
                    cfg->image_train_dir[sizeof(cfg->image_train_dir) - 1] = '\0';
                    clean_value(cfg->image_train_dir);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "image_test_dir") == 0) {
                    strncpy(cfg->image_test_dir, v, sizeof(cfg->image_test_dir) - 1);
                    cfg->image_test_dir[sizeof(cfg->image_test_dir) - 1] = '\0';
                    clean_value(cfg->image_test_dir);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "image_val_dir") == 0) {
                    strncpy(cfg->image_val_dir, v, sizeof(cfg->image_val_dir) - 1);
                    cfg->image_val_dir[sizeof(cfg->image_val_dir) - 1] = '\0';
                    clean_value(cfg->image_val_dir);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "image_width") == 0) {
                    cfg->image_width = atoi(v);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "image_height") == 0) {
                    cfg->image_height = atoi(v);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "image_channels") == 0) {
                    cfg->image_channels = atoi(v);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "input_cols") == 0) {
                    cfg->input_cols = atoi(v);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "output_cols") == 0) {
                    cfg->output_cols = atoi(v);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "dataset") == 0) {
                    strncpy(cfg->dataset, v, sizeof(cfg->dataset) - 1);
                    cfg->dataset[sizeof(cfg->dataset) - 1] = '\0';
                    clean_value(cfg->dataset);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "dataset_name") == 0) {
                    strncpy(cfg->dataset_name, v, sizeof(cfg->dataset_name) - 1);
                    cfg->dataset_name[sizeof(cfg->dataset_name) - 1] = '\0';
                    clean_value(cfg->dataset_name);
                    current_list_type[0] = '\0';
                }
                // 🆕 NOUVEAUX CHAMPS POUR L'ANALYSE DYNAMIQUE DES DATASETS
                else if (strcmp(k, "input_fields") == 0) {
                    strncpy(cfg->input_fields, v, sizeof(cfg->input_fields) - 1);
                    cfg->input_fields[sizeof(cfg->input_fields) - 1] = '\0';
                    clean_value(cfg->input_fields);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "output_fields") == 0) {
                    strncpy(cfg->output_fields, v, sizeof(cfg->output_fields) - 1);
                    cfg->output_fields[sizeof(cfg->output_fields) - 1] = '\0';
                    clean_value(cfg->output_fields);
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "auto_normalize") == 0) {
                    cfg->auto_normalize = (strstr(v, "true") || strstr(v, "yes") || strstr(v, "1")) ? 1 : 0;
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "auto_categorize") == 0) {
                    cfg->auto_categorize = (strstr(v, "true") || strstr(v, "yes") || strstr(v, "1")) ? 1 : 0;
                    current_list_type[0] = '\0';
                }
                else if (strcmp(k, "field_detection") == 0) {
                    strncpy(cfg->field_detection, v, sizeof(cfg->field_detection) - 1);
                    cfg->field_detection[sizeof(cfg->field_detection) - 1] = '\0';
                    clean_value(cfg->field_detection);
                    current_list_type[0] = '\0';
                }
                else {
                    current_list_type[0] = '\0';  // Clé non reconnue, sortir du mode liste
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