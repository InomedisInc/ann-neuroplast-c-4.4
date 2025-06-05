#ifndef CSV_EXPORT_COMPLETE_H
#define CSV_EXPORT_COMPLETE_H

#include <stddef.h>

// Structure pour toutes les métriques d'évaluation
typedef struct {
    char method[32];
    char optimizer[32];
    char activation[32];
    char full_name[128];
    // Métriques moyennes sur tous les essais
    float avg_accuracy;
    float avg_precision;
    float avg_recall;
    float avg_f1_score;
    float avg_auc_roc;
    // Meilleures métriques obtenues
    float best_accuracy;
    float best_precision;
    float best_recall;
    float best_f1_score;
    float best_auc_roc;
    // Informations de convergence
    int convergence_count;
    int total_trials;
    float convergence_rate;
} CombinationResultComplete;

/**
 * Exporte les résultats complets avec toutes les métriques vers un fichier CSV
 * @param results Array des résultats de combinaisons
 * @param result_count Nombre de résultats
 * @param config_path Chemin vers le fichier de configuration
 * @param input_cols Nombre de colonnes d'entrée
 * @param output_cols Nombre de colonnes de sortie
 * @param total_combinations Nombre total de combinaisons
 * @param max_epochs Nombre maximum d'époques
 * @param num_samples Nombre d'échantillons dans le dataset
 * @param dataset_name Nom du dataset pour l'affichage
 * @return 1 si succès, 0 si erreur
 */
int export_results_to_csv_complete(CombinationResultComplete *results, int result_count, 
                                   const char *config_path, size_t input_cols, size_t output_cols,
                                   int total_combinations, int max_epochs, size_t num_samples, 
                                   const char *dataset_name);

#endif // CSV_EXPORT_COMPLETE_H 