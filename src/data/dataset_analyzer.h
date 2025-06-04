#ifndef DATASET_ANALYZER_H
#define DATASET_ANALYZER_H

#include <stddef.h>
#include <stdbool.h>
#include "../rich_config.h"
#include "data_loader.h"

#define MAX_FIELD_NAME 128
#define MAX_FIELDS 64

// Type de champ détecté automatiquement
typedef enum {
    FIELD_NUMERIC,      // Champ numérique (normalisation min-max ou z-score)
    FIELD_CATEGORICAL,  // Champ catégorique (binarisation 0/1)
    FIELD_BINARY        // Champ déjà binaire (0/1)
} FieldType;

// Structure pour analyser et traiter les champs
typedef struct {
    char input_fields[MAX_FIELDS][MAX_FIELD_NAME];  // Liste des champs d'entrée
    char output_fields[MAX_FIELDS][MAX_FIELD_NAME]; // Liste des champs de sortie
    int num_input_fields;
    int num_output_fields;
    
    // Statistiques pour normalisation
    float input_min[MAX_FIELDS];
    float input_max[MAX_FIELDS];
    float input_mean[MAX_FIELDS];
    float input_std[MAX_FIELDS];
    FieldType input_types[MAX_FIELDS];
    
    float output_min[MAX_FIELDS];
    float output_max[MAX_FIELDS];
    FieldType output_types[MAX_FIELDS];
    
    // Informations du dataset
    size_t num_samples;
    bool is_analyzed;
} DatasetAnalyzer;

// Fonctions principales
bool analyze_dataset_fields(const RichConfig *config, DatasetAnalyzer *analyzer);
bool process_tabular_dataset(const RichConfig *config, const DatasetAnalyzer *analyzer, Dataset **dataset);
bool parse_field_list(const char *field_string, char fields[][MAX_FIELD_NAME], int *num_fields);

// Fonctions utilitaires
bool detect_field_type_simple(const float *values, size_t count, FieldType *type);
void normalize_numeric_field(float *values, size_t count, float min_val, float max_val);
void calculate_stats(const float *values, size_t count, float *min, float *max, float *mean, float *std);

// Fonction d'intégration principale pour test_all_with_real_dataset
Dataset* create_analyzed_dataset(const RichConfig *config);

#endif // DATASET_ANALYZER_H 