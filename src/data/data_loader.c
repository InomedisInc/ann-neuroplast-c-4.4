#include "data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "dataset.h"
#include "image_loader.h"
#include "../yaml_parser_rich.h"
#include "../colored_output.h"

Dataset *load_csv_data(const char *filepath, size_t input_cols, size_t output_cols) {
    if (!filepath) {
        printf("Erreur: chemin de fichier invalide (NULL)\n");
        return NULL;
    }

    if (input_cols == 0 || output_cols == 0) {
        printf("Erreur: dimensions invalides (input_cols=%zu, output_cols=%zu)\n", 
               input_cols, output_cols);
        return NULL;
    }

    FILE *file = fopen(filepath, "r");
    if (!file) {
        printf("Erreur: impossible d'ouvrir le fichier '%s'\n", filepath);
        return NULL;
    }

    char buffer[4096];
    size_t row_capacity = 128, rows = 0;
    Dataset *d = dataset_create(row_capacity, input_cols, output_cols);
    if (!d) {
        printf("Erreur: impossible de créer le dataset (mémoire insuffisante)\n");
        fclose(file);
        return NULL;
    }

    // Ignore la première ligne (en-tête) et l'analyser pour déterminer le format
    if (fgets(buffer, sizeof(buffer), file) == NULL) {
        printf("Erreur: impossible de lire la première ligne du fichier '%s'\n", filepath);
        dataset_free(d);
        fclose(file);
        return NULL;
    }
    
    // Analyser l'en-tête pour déterminer le format du CSV
    int target_first = 0;
    char *header = strdup(buffer);
    if (header) {
        // Analyse si la cible est en premier ou à la fin
        char *first_col = strtok(header, ",");
        if (first_col && strstr(first_col, "Heart")) {
            target_first = 1;
            char info_msg[256];
            snprintf(info_msg, sizeof(info_msg), "Détection automatique - colonne cible au début du CSV");
            print_dataset_info(info_msg);
        }
        free(header);
    }

    size_t line_number = 1;  // On a déjà lu la ligne 1
    while (fgets(buffer, sizeof(buffer), file)) {
        line_number++;
        
        // Ignore les lignes vides ou commentées
        if (buffer[0] == '\n' || buffer[0] == '#') continue;

        if (rows >= row_capacity) {
            row_capacity *= 2;
            if (!dataset_resize(d, row_capacity)) {
                printf("Erreur: impossible de redimensionner le dataset (mémoire insuffisante)\n");
                dataset_free(d);
                fclose(file);
                return NULL;
            }
        }

        char *token = strtok(buffer, ",");
        size_t col = 0;
        float target_val = 0.0f;  // Stocker la valeur cible si elle est au début
        
        // Si la cible est au début, on la stocke avant
        if (target_first && token) {
            char *start = token;
            while (*start && isspace(*start)) start++;
            char *end = start + strlen(start) - 1;
            while (end > start && isspace(*end)) end--;
            *(end + 1) = '\0';
            
            char *endptr;
            target_val = strtof(start, &endptr);
            if (*endptr != '\0') {
                printf("Erreur: valeur non numérique '%s' à la ligne %zu, colonne 1 (cible)\n", 
                       start, line_number);
                dataset_free(d);
                fclose(file);
                return NULL;
            }
            
            token = strtok(NULL, ",");  // Passer à la colonne suivante
        }

        // Lire les input_cols colonnes d'entrée
        for (size_t i = 0; i < input_cols && token; i++) {
            char *start = token;
            while (*start && isspace(*start)) start++;
            char *end = start + strlen(start) - 1;
            while (end > start && isspace(*end)) end--;
            *(end + 1) = '\0';
            
            char *endptr;
            float v = strtof(start, &endptr);
            if (*endptr != '\0') {
                printf("Erreur: valeur non numérique '%s' à la ligne %zu, colonne %zu\n", 
                       start, line_number, i + 1 + (target_first ? 1 : 0));
                dataset_free(d);
                fclose(file);
                return NULL;
            }
            
            d->inputs[rows][i] = v;
            token = strtok(NULL, ",");
            col++;
        }
        
        // Si la cible est au début, on l'a déjà lue, sinon on lit les colonnes de sortie
        if (target_first) {
            if (output_cols == 1) {
                d->outputs[rows][0] = target_val;
            } else {
                // Pour classification binaire avec 2 sorties (one-hot)
                d->outputs[rows][0] = (target_val < 0.5f) ? 1.0f : 0.0f;
                d->outputs[rows][1] = (target_val < 0.5f) ? 0.0f : 1.0f;
            }
        } else {
            // Lire les output_cols colonnes de sortie
            for (size_t i = 0; i < output_cols && token; i++) {
                char *start = token;
                while (*start && isspace(*start)) start++;
                char *end = start + strlen(start) - 1;
                while (end > start && isspace(*end)) end--;
                *(end + 1) = '\0';
                
                char *endptr;
                float v = strtof(start, &endptr);
                if (*endptr != '\0') {
                    printf("Erreur: valeur non numérique '%s' à la ligne %zu, colonne %zu\n", 
                           start, line_number, col + 1);
                    dataset_free(d);
                    fclose(file);
                    return NULL;
                }
                
                d->outputs[rows][i] = v;
                token = strtok(NULL, ",");
                col++;
            }
        }
        
        // Vérifie s'il y a des colonnes en trop
        // Pour target_first=1: on a déjà lu 1 (target) + input_cols features = input_cols+1 total
        // Pour target_first=0: on a lu input_cols features + output_cols targets = input_cols+output_cols total
        if (token) {
            printf("Erreur: trop de colonnes à la ligne %zu (attendu %zu colonnes)\n", 
                   line_number, target_first ? input_cols + 1 : input_cols + output_cols);
            dataset_free(d);
            fclose(file);
            return NULL;
        }

        ++rows;
    }

    if (rows == 0) {
        printf("Erreur: le fichier '%s' ne contient aucune donnée valide\n", filepath);
        dataset_free(d);
        fclose(file);
        return NULL;
    }

    d->num_samples = rows;
    fclose(file);
    char final_success_msg[256];
    snprintf(final_success_msg, sizeof(final_success_msg), 
            "Dataset chargé avec succès : %zu échantillons, %zu entrées, %zu sorties", 
            d->num_samples, d->input_cols, d->output_cols);
    print_dataset_success(final_success_msg);
    return d;
}

Dataset *merge_datasets(const Dataset *d1, const Dataset *d2) {
    if (!d1 || !d2) {
        printf("Erreur: datasets invalides (NULL)\n");
        return NULL;
    }
    
    // Vérifie que les dimensions correspondent
    if (d1->input_cols != d2->input_cols || d1->output_cols != d2->output_cols) {
        printf("Erreur: dimensions incompatibles entre les datasets "
               "(d1: %zux%zu, d2: %zux%zu)\n",
               d1->input_cols, d1->output_cols,
               d2->input_cols, d2->output_cols);
        return NULL;
    }
    
    // Crée un nouveau dataset avec la capacité combinée
    Dataset *merged = dataset_create(d1->num_samples + d2->num_samples,
                                   d1->input_cols, d1->output_cols);
    if (!merged) {
        printf("Erreur: impossible de créer le dataset fusionné (mémoire insuffisante)\n");
        return NULL;
    }
    
    // Copie les données du premier dataset
    for (size_t i = 0; i < d1->num_samples; i++) {
        for (size_t j = 0; j < d1->input_cols; j++) {
            merged->inputs[i][j] = d1->inputs[i][j];
        }
        for (size_t j = 0; j < d1->output_cols; j++) {
            merged->outputs[i][j] = d1->outputs[i][j];
        }
    }
    
    // Copie les données du second dataset
    for (size_t i = 0; i < d2->num_samples; i++) {
        for (size_t j = 0; j < d2->input_cols; j++) {
            merged->inputs[d1->num_samples + i][j] = d2->inputs[i][j];
        }
        for (size_t j = 0; j < d2->output_cols; j++) {
            merged->outputs[d1->num_samples + i][j] = d2->outputs[i][j];
        }
    }
    
    merged->num_samples = d1->num_samples + d2->num_samples;
    return merged;
}

Dataset *load_dataset_from_yaml(const char *yaml_path) {
    if (!yaml_path) return NULL;
    
    RichConfig cfg = {0};
    if (!parse_yaml_rich(yaml_path, &cfg)) {
        return NULL;
    }
    
    if (cfg.input_cols == 0 || cfg.output_cols == 0) {
        printf("Erreur: dimensions non spécifiées dans le fichier dataset (input_cols=%zu, output_cols=%zu)\n", 
               cfg.input_cols, cfg.output_cols);
        return NULL;
    }

    // Pour une classification binaire, output_cols peut être 2 dans la config (2 classes)
    // mais dans le CSV il n'y a souvent qu'une seule colonne cible
    size_t csv_output_cols = cfg.output_cols;
    if (cfg.output_cols == 2 && strstr(yaml_path, "heart_attack.yml")) {
        printf("Ajustement pour classification binaire : output_cols=1 (au lieu de 2)\n");
        csv_output_cols = 1;
    }

    // Charge le dataset principal
    Dataset *current = NULL;
    if (cfg.dataset[0] != '\0') {
        printf("Chargement du dataset principal '%s'...\n", cfg.dataset);
        current = load_csv_data(cfg.dataset, cfg.input_cols, csv_output_cols);
        if (!current) {
            printf("Erreur: impossible de charger le dataset '%s'\n", cfg.dataset);
            return NULL;
        }
    }

    // Charge et fusionne les datasets additionnels
    for (int i = 0; i < cfg.num_datasets; i++) {
        printf("Chargement du dataset additionnel '%s'...\n", cfg.datasets[i]);
        Dataset *additional = load_dataset_from_yaml(cfg.datasets[i]);
        if (!additional) {
            if (current) dataset_free(current);
            printf("Erreur: impossible de charger le dataset additionnel '%s'\n", cfg.datasets[i]);
            return NULL;
        }

        if (current) {
            printf("Fusion avec le dataset additionnel...\n");
            Dataset *merged = merge_datasets(current, additional);
            dataset_free(current);
            dataset_free(additional);
            if (!merged) {
                printf("Erreur: impossible de fusionner avec le dataset additionnel '%s'\n", 
                       cfg.datasets[i]);
                return NULL;
            }
            current = merged;
        } else {
            current = additional;
        }
    }

    if (!current) {
        printf("Erreur: aucun dataset n'a été chargé\n");
        return NULL;
    }

    printf("Dataset chargé avec succès : %zu échantillons, %zu entrées, %zu sorties\n",
           current->num_samples, current->input_cols, current->output_cols);
    return current;
}

Dataset *load_dataset_from_config(const RichConfig *config) {
    if (!config) {
        printf("Erreur: configuration invalide (NULL)\n");
        return NULL;
    }

    // Vérifier si c'est un dataset d'images ou tabulaire
    if (config->is_image_dataset) {
        printf("=== Mode Dataset d'Images ===\n");
        
        // Vérifier que les paramètres d'images sont définis
        if (config->image_width <= 0 || config->image_height <= 0 || config->image_channels <= 0) {
            printf("Erreur: dimensions d'images invalides (width=%d, height=%d, channels=%d)\n",
                   config->image_width, config->image_height, config->image_channels);
            return NULL;
        }

        // Vérifier qu'au moins train et test sont définis
        if (config->image_train_dir[0] == '\0' || config->image_test_dir[0] == '\0') {
            printf("Erreur: répertoires train et test obligatoires pour les images\n");
            printf("  - image_train_dir: '%s'\n", config->image_train_dir);
            printf("  - image_test_dir: '%s'\n", config->image_test_dir);
            return NULL;
        }

        printf("Configuration d'images:\n");
        printf("  - Dimensions: %dx%dx%d\n", config->image_width, config->image_height, config->image_channels);
        printf("  - Train: %s\n", config->image_train_dir);
        printf("  - Test: %s\n", config->image_test_dir);
        if (config->image_val_dir[0] != '\0') {
            printf("  - Validation: %s\n", config->image_val_dir);
        } else {
            printf("  - Validation: non défini (optionnel)\n");
        }

        return load_image_dataset_from_config(config);
    } else {
        printf("=== Mode Dataset Tabulaire ===\n");
        
        // Vérifier que les paramètres tabulaires sont définis
        if (config->input_cols == 0 || config->output_cols == 0) {
            printf("Erreur: dimensions tabulaires invalides (input_cols=%zu, output_cols=%zu)\n",
                   config->input_cols, config->output_cols);
            return NULL;
        }

        // Vérifier qu'un dataset est défini
        if (config->dataset[0] == '\0') {
            printf("Erreur: aucun fichier dataset défini pour les données tabulaires\n");
            return NULL;
        }

        printf("Configuration tabulaire:\n");
        printf("  - Dimensions: %zu entrées, %zu sorties\n", config->input_cols, config->output_cols);
        printf("  - Dataset: %s\n", config->dataset);
        if (config->train_test_split > 0) {
            printf("  - Train/Test split: %.2f\n", config->train_test_split);
        }

        // Utiliser la fonction existante pour les données tabulaires
        Dataset *current = NULL;
        if (config->dataset[0] != '\0') {
            printf("Chargement du dataset principal '%s'...\n", config->dataset);
            
            // Pour une classification binaire, output_cols peut être 2 dans la config (2 classes)
            // mais dans le CSV il n'y a souvent qu'une seule colonne cible
            size_t csv_output_cols = config->output_cols;
            if (config->output_cols == 2) {
                printf("Ajustement pour classification binaire : output_cols=1 (au lieu de 2)\n");
                csv_output_cols = 1;
            }

            current = load_csv_data(config->dataset, config->input_cols, csv_output_cols);
            if (!current) {
                printf("Erreur: impossible de charger le dataset '%s'\n", config->dataset);
                return NULL;
            }
        }

        // Charge et fusionne les datasets additionnels si définis
        for (int i = 0; i < config->num_datasets; i++) {
            printf("Chargement du dataset additionnel '%s'...\n", config->datasets[i]);
            Dataset *additional = load_csv_data(config->datasets[i], config->input_cols, config->output_cols);
            if (!additional) {
                if (current) dataset_free(current);
                printf("Erreur: impossible de charger le dataset additionnel '%s'\n", config->datasets[i]);
                return NULL;
            }

            if (current) {
                printf("Fusion avec le dataset additionnel...\n");
                Dataset *merged = merge_datasets(current, additional);
                dataset_free(current);
                dataset_free(additional);
                if (!merged) {
                    printf("Erreur: impossible de fusionner avec le dataset additionnel '%s'\n", 
                           config->datasets[i]);
                    return NULL;
                }
                current = merged;
            } else {
                current = additional;
            }
        }

        if (!current) {
            printf("Erreur: aucun dataset n'a été chargé\n");
            return NULL;
        }

        printf("Dataset tabulaire chargé avec succès : %zu échantillons, %zu entrées, %zu sorties\n",
               current->num_samples, current->input_cols, current->output_cols);
        return current;
    }
}