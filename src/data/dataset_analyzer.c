#include "dataset_analyzer.h"
#include "../colored_output.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

// ============================================================================
// FONCTIONS UTILITAIRES DE BASE
// ============================================================================

void trim_whitespace(char *str) {
    if (!str) return;
    
    // Supprimer les espaces à la fin
    int len = strlen(str);
    while (len > 0 && isspace(str[len-1])) {
        str[--len] = '\0';
    }
    
    // Supprimer les espaces au début
    char *start = str;
    while (*start && isspace(*start)) start++;
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }
}

bool parse_field_list(const char *field_string, char fields[][MAX_FIELD_NAME], int *num_fields) {
    if (!field_string || !fields || !num_fields) return false;
    
    *num_fields = 0;
    char *str_copy = strdup(field_string);
    if (!str_copy) return false;
    
    char *token = strtok(str_copy, ",");
    while (token && *num_fields < MAX_FIELDS) {
        trim_whitespace(token);
        if (strlen(token) > 0) {
            strncpy(fields[*num_fields], token, MAX_FIELD_NAME - 1);
            fields[*num_fields][MAX_FIELD_NAME - 1] = '\0';
            (*num_fields)++;
        }
        token = strtok(NULL, ",");
    }
    
    free(str_copy);
    return (*num_fields > 0);
}

void calculate_stats(const float *values, size_t count, float *min, float *max, float *mean, float *std) {
    if (!values || count == 0 || !min || !max || !mean || !std) return;
    
    *min = values[0];
    *max = values[0];
    double sum = 0.0;
    
    // Calculer min, max et moyenne
    for (size_t i = 0; i < count; i++) {
        if (values[i] < *min) *min = values[i];
        if (values[i] > *max) *max = values[i];
        sum += values[i];
    }
    *mean = (float)(sum / count);
    
    // Calculer l'écart-type
    double variance_sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = values[i] - *mean;
        variance_sum += diff * diff;
    }
    *std = (float)sqrt(variance_sum / count);
}

bool detect_field_type_simple(const float *values, size_t count, FieldType *type) {
    if (!values || count == 0 || !type) return false;
    
    int binary_count = 0;
    int unique_values = 0;
    float unique_vals[10]; // Limiter à 10 valeurs uniques pour catégorique
    
    for (size_t i = 0; i < count && i < 1000; i++) { // Échantillonner max 1000 valeurs
        float val = values[i];
        
        // Vérifier si binaire (0 ou 1)
        if (val == 0.0f || val == 1.0f) {
            binary_count++;
        }
        
        // Compter les valeurs uniques (pour détecter catégorique)
        bool found = false;
        for (int j = 0; j < unique_values; j++) {
            if (fabs(unique_vals[j] - val) < 0.001f) {
                found = true;
                break;
            }
        }
        if (!found && unique_values < 10) {
            unique_vals[unique_values++] = val;
        }
    }
    
    size_t sample_count = (count < 1000) ? count : 1000;
    
    // Règles de détection
    if (binary_count == sample_count && unique_values <= 2) {
        *type = FIELD_BINARY;
    } else if (unique_values <= 5) {  // Peu de valeurs uniques = catégorique
        *type = FIELD_CATEGORICAL;
    } else {
        *type = FIELD_NUMERIC;
    }
    
    return true;
}

void normalize_numeric_field(float *values, size_t count, float min_val, float max_val) {
    if (!values || count == 0) return;
    
    float range = max_val - min_val;
    if (range < 0.001f) return; // Éviter division par zéro
    
    for (size_t i = 0; i < count; i++) {
        values[i] = (values[i] - min_val) / range;
    }
}

// ============================================================================
// ANALYSE DU DATASET À PARTIR DE LA CONFIGURATION
// ============================================================================

bool analyze_dataset_fields(const RichConfig *config, DatasetAnalyzer *analyzer) {
    if (!config || !analyzer) return false;
    
    printf("🔍 Analyse des champs du dataset: %s\n", config->dataset_name);
    
    // Initialiser l'analyzeur
    memset(analyzer, 0, sizeof(DatasetAnalyzer));
    
    // Vérifier si c'est un dataset d'images
    if (config->is_image_dataset) {
        printf("🖼️ Dataset d'images détecté - aucune analyse de champs nécessaire\n");
        return false; // Pas besoin d'analyse pour les images
    }
    
    printf("📊 Dataset tabulaire détecté - analyse des champs\n");
    
    // 🆕 LECTURE DYNAMIQUE DES CHAMPS DEPUIS LA CONFIGURATION YAML
    bool fields_found = false;
    
    // Vérifier si les champs sont définis dans la configuration
    if (strlen(config->input_fields) > 0 && strlen(config->output_fields) > 0) {
        printf("✅ Champs trouvés dans la configuration YAML\n");
        printf("📋 Input fields: %s\n", config->input_fields);
        printf("🎯 Output fields: %s\n", config->output_fields);
        
        // Parser les champs d'entrée depuis la configuration
        if (parse_field_list(config->input_fields, analyzer->input_fields, &analyzer->num_input_fields)) {
            printf("✅ %d champs d'entrée parsés avec succès\n", analyzer->num_input_fields);
            fields_found = true;
        } else {
            printf("❌ Erreur lors du parsing des champs d'entrée\n");
        }
        
        // Parser les champs de sortie depuis la configuration
        if (parse_field_list(config->output_fields, analyzer->output_fields, &analyzer->num_output_fields)) {
            printf("✅ %d champs de sortie parsés avec succès\n", analyzer->num_output_fields);
        } else {
            printf("❌ Erreur lors du parsing des champs de sortie\n");
            fields_found = false;
        }
    }
    
    // Fallback: utiliser des champs par défaut si pas trouvés dans la configuration
    if (!fields_found) {
        printf("⚠️ Champs non trouvés dans la configuration, utilisation de valeurs par défaut\n");
        
        // Détecter le type de dataset par le nom pour les valeurs par défaut
        if (strstr(config->dataset_name, "cancer") != NULL) {
            // Dataset cancer - 30 features
            const char *cancer_inputs = "radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst";
            const char *cancer_output = "diagnosis";
            
            parse_field_list(cancer_inputs, analyzer->input_fields, &analyzer->num_input_fields);
            parse_field_list(cancer_output, analyzer->output_fields, &analyzer->num_output_fields);
            
            printf("✅ Configuration cancer par défaut: %d inputs, %d outputs\n", 
                   analyzer->num_input_fields, analyzer->num_output_fields);
        } else {
            // Dataset par défaut (médical simulé)
            const char *default_inputs = "age,cholesterol,blood_pressure,bmi,exercise,smoking,family_history,stress";
            const char *default_output = "risk";
            
            parse_field_list(default_inputs, analyzer->input_fields, &analyzer->num_input_fields);
            parse_field_list(default_output, analyzer->output_fields, &analyzer->num_output_fields);
            
            printf("✅ Configuration par défaut: %d inputs, %d outputs\n", 
                   analyzer->num_input_fields, analyzer->num_output_fields);
        }
    }
    
    // Afficher les champs détectés
    printf("📋 Champs d'entrée (%d):\n", analyzer->num_input_fields);
    for (int i = 0; i < analyzer->num_input_fields; i++) {
        printf("   %d. %s\n", i+1, analyzer->input_fields[i]);
    }
    
    printf("🎯 Champs de sortie (%d):\n", analyzer->num_output_fields);
    for (int i = 0; i < analyzer->num_output_fields; i++) {
        printf("   %d. %s\n", i+1, analyzer->output_fields[i]);
    }
    
    // Afficher les options d'analyse automatique
    if (strlen(config->field_detection) > 0) {
        printf("🔍 Mode de détection: %s\n", config->field_detection);
    }
    if (config->auto_normalize) {
        printf("✅ Normalisation automatique activée\n");
    }
    if (config->auto_categorize) {
        printf("✅ Catégorisation automatique activée\n");
    }
    
    analyzer->is_analyzed = true;
    return true;
}

// ============================================================================
// TRAITEMENT DU DATASET TABULAIRE
// ============================================================================

bool process_tabular_dataset(const RichConfig *config, const DatasetAnalyzer *analyzer, Dataset **dataset) {
    if (!config || !analyzer || !dataset || !analyzer->is_analyzed) return false;
    
    printf("🔄 Traitement du dataset tabulaire avec analyse automatique\n");
    
    // Essayer de charger le dataset depuis le fichier
    FILE *file = fopen(config->dataset, "r");
    if (!file) {
        printf("⚠️ Fichier dataset non trouvé: %s\n", config->dataset);
        printf("🔄 Génération d'un dataset simulé basé sur l'analyse des champs\n");
        return false; // Laissera la fonction appelante créer un dataset simulé
    }
    
    printf("📂 Chargement du dataset depuis: %s\n", config->dataset);
    
    // Compter les lignes
    size_t num_samples = 0;
    char line[2048];
    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) > 1) num_samples++; // Ignorer les lignes vides
    }
    rewind(file);
    
    if (num_samples == 0) {
        printf("❌ Aucune donnée trouvée dans le fichier\n");
        fclose(file);
        return false;
    }
    
    printf("📊 %zu échantillons détectés\n", num_samples);
    
    // Créer le dataset
    *dataset = dataset_create(num_samples, analyzer->num_input_fields, analyzer->num_output_fields);
    if (!*dataset) {
        printf("❌ Erreur création dataset\n");
        fclose(file);
        return false;
    }
    
    // Charger les données brutes
    float **raw_inputs = malloc(num_samples * sizeof(float*));
    float **raw_outputs = malloc(num_samples * sizeof(float*));
    
    for (size_t i = 0; i < num_samples; i++) {
        raw_inputs[i] = malloc(analyzer->num_input_fields * sizeof(float));
        raw_outputs[i] = malloc(analyzer->num_output_fields * sizeof(float));
    }
    
    size_t sample_idx = 0;
    while (fgets(line, sizeof(line), file) && sample_idx < num_samples) {
        char *token = strtok(line, ",");
        int col = 0;
        int input_idx = 0;
        int output_idx = 0;
        
        while (token && col < (analyzer->num_input_fields + analyzer->num_output_fields)) {
            float value = atof(token);
            
            if (col == 0 && analyzer->num_output_fields > 0) {
                // Première colonne = output (format target_first)
                raw_outputs[sample_idx][output_idx++] = value;
            } else if (input_idx < analyzer->num_input_fields) {
                // Colonnes suivantes = inputs
                raw_inputs[sample_idx][input_idx++] = value;
            }
            
            col++;
            token = strtok(NULL, ",");
        }
        sample_idx++;
    }
    fclose(file);
    
    (*dataset)->num_samples = sample_idx;
    printf("✅ %zu échantillons chargés\n", sample_idx);
    
    // Analyser et normaliser chaque champ d'entrée
    printf("🔍 Analyse et normalisation des champs d'entrée:\n");
    for (int i = 0; i < analyzer->num_input_fields; i++) {
        // Extraire les valeurs de ce champ
        float *field_values = malloc(sample_idx * sizeof(float));
        for (size_t j = 0; j < sample_idx; j++) {
            field_values[j] = raw_inputs[j][i];
        }
        
        // Détecter le type et calculer les statistiques
        FieldType field_type;
        detect_field_type_simple(field_values, sample_idx, &field_type);
        
        float min_val, max_val, mean_val, std_val;
        calculate_stats(field_values, sample_idx, &min_val, &max_val, &mean_val, &std_val);
        
        printf("   📋 %s: ", analyzer->input_fields[i]);
        
        // Traitement selon le type
        switch (field_type) {
            case FIELD_NUMERIC:
                printf("numérique [%.3f, %.3f] → normalisation min-max\n", min_val, max_val);
                normalize_numeric_field(field_values, sample_idx, min_val, max_val);
                break;
                
            case FIELD_CATEGORICAL:
                printf("catégorique → binarisation 0/1\n");
                // Pour les catégoriques, mapper vers 0/1 basé sur la valeur médiane
                float median = (min_val + max_val) / 2.0f;
                for (size_t j = 0; j < sample_idx; j++) {
                    field_values[j] = (field_values[j] > median) ? 1.0f : 0.0f;
                }
                break;
                
            case FIELD_BINARY:
                printf("binaire → pas de traitement\n");
                // Déjà en format 0/1, pas de traitement nécessaire
                break;
        }
        
        // Copier les valeurs traitées dans le dataset final
        for (size_t j = 0; j < sample_idx; j++) {
            (*dataset)->inputs[j][i] = field_values[j];
        }
        
        free(field_values);
    }
    
    // Traiter les champs de sortie
    printf("🎯 Traitement des champs de sortie:\n");
    for (int i = 0; i < analyzer->num_output_fields; i++) {
        printf("   🎯 %s: classification binaire\n", analyzer->output_fields[i]);
        
        for (size_t j = 0; j < sample_idx; j++) {
            (*dataset)->outputs[j][i] = raw_outputs[j][i];
        }
    }
    
    // Nettoyer les données temporaires
    for (size_t i = 0; i < num_samples; i++) {
        free(raw_inputs[i]);
        free(raw_outputs[i]);
    }
    free(raw_inputs);
    free(raw_outputs);
    
    printf("✅ Dataset traité avec succès: %zu échantillons, %d features, %d outputs\n", 
           (*dataset)->num_samples, analyzer->num_input_fields, analyzer->num_output_fields);
    
    return true;
}

// ============================================================================
// FONCTION D'INTÉGRATION PRINCIPALE
// ============================================================================

Dataset* create_analyzed_dataset(const RichConfig *config) {
    if (!config) return NULL;
    
    // Test initial: images ou données tabulaires ?
    if (config->is_image_dataset) {
        printf("🖼️ Dataset d'images détecté - utilisation du chargeur d'images standard\n");
        return load_dataset_from_config(config);
    }
    
    printf("📊 Dataset tabulaire détecté - utilisation de l'analyseur automatique\n");
    
    // Analyser les champs du dataset
    DatasetAnalyzer analyzer;
    if (!analyze_dataset_fields(config, &analyzer)) {
        printf("❌ Échec de l'analyse des champs\n");
        return NULL;
    }
    
    // Traiter le dataset tabulaire
    Dataset *dataset = NULL;
    if (process_tabular_dataset(config, &analyzer, &dataset)) {
        printf("✅ Dataset tabulaire traité avec succès\n");
        return dataset;
    }
    
    // Si le traitement échoue, créer un dataset simulé basé sur l'analyse
    printf("🔄 Création d'un dataset simulé basé sur l'analyse des champs\n");
    
    size_t num_samples = 800;
    dataset = dataset_create(num_samples, analyzer.num_input_fields, analyzer.num_output_fields);
    if (!dataset) return NULL;
    
    // Générer des données simulées selon les champs analysés
    srand(42); // Seed fixe pour reproductibilité
    
    for (size_t i = 0; i < num_samples; i++) {
        // Générer les features d'entrée selon les noms des champs
        for (int j = 0; j < analyzer.num_input_fields; j++) {
            const char *field_name = analyzer.input_fields[j];
            
            if (strstr(field_name, "age") != NULL) {
                dataset->inputs[i][j] = 0.2f + 0.6f * ((float)rand() / RAND_MAX);
            } else if (strstr(field_name, "pressure") != NULL || strstr(field_name, "cholesterol") != NULL) {
                dataset->inputs[i][j] = 0.1f + 0.8f * ((float)rand() / RAND_MAX);
            } else if (strstr(field_name, "smoking") != NULL || strstr(field_name, "family") != NULL) {
                dataset->inputs[i][j] = ((float)rand() / RAND_MAX > 0.7f) ? 1.0f : 0.0f;
            } else {
                // Champ numérique générique
                dataset->inputs[i][j] = ((float)rand() / RAND_MAX);
            }
        }
        
        // Calculer la sortie basée sur un modèle complexe
        float risk_score = 0.0f;
        for (int j = 0; j < analyzer.num_input_fields; j++) {
            risk_score += dataset->inputs[i][j] * (0.1f + 0.1f * j); // Poids progressif
        }
        
        // Ajouter des interactions complexes
        if (analyzer.num_input_fields >= 2) {
            risk_score += dataset->inputs[i][0] * dataset->inputs[i][1] * 0.2f;
        }
        
        // Conversion en probabilité et seuillage
        float probability = 1.0f / (1.0f + expf(-(risk_score - 0.5f) * 4.0f));
        dataset->outputs[i][0] = (probability > 0.5f) ? 1.0f : 0.0f;
    }
    
    printf("✅ Dataset simulé créé: %zu échantillons avec champs analysés automatiquement\n", num_samples);
    return dataset;
} 