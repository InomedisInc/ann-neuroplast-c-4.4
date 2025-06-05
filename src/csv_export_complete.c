#include <stdio.h>
#include <time.h>
#include <string.h>

// Structure pour toutes les métriques
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

int export_results_to_csv_complete(CombinationResultComplete *results, int result_count, 
                                   const char *config_path, size_t input_cols, size_t output_cols,
                                   int total_combinations, int max_epochs, size_t num_samples) {
    printf("\n📊 EXPORT DES RÉSULTATS EN CSV (TOUTES MÉTRIQUES)...\n");
    
    // Extraire le nom du dataset du chemin de config
    const char *dataset_name = "dataset";
    if (config_path) {
        const char *last_slash = strrchr(config_path, '/');
        const char *filename = last_slash ? last_slash + 1 : config_path;
        dataset_name = filename;
    }
    
    char csv_filename[256];
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    snprintf(csv_filename, sizeof(csv_filename), "results_exhaustif_%.*s_%04d%02d%02d_%02d%02d%02d.csv",
            (int)(strrchr(dataset_name, '.') - dataset_name), dataset_name,
            tm_info->tm_year + 1900, tm_info->tm_mon + 1, tm_info->tm_mday,
            tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec);
    
    FILE *csv_file = fopen(csv_filename, "w");
    if (!csv_file) {
        printf("❌ Erreur lors de la création du fichier CSV\n");
        return 0;
    }
    
    // En-tête CSV avec métadonnées du dataset
    fprintf(csv_file, "# NEUROPLAST-ANN - Test Exhaustif Dataset %s\n", dataset_name);
    fprintf(csv_file, "# Dataset: %s (%zu échantillons)\n", config_path, num_samples);
    fprintf(csv_file, "# Architecture: Input(%zu)→256→128→Output(%zu)\n", input_cols, output_cols);
    fprintf(csv_file, "# Total combinaisons: %d\n", total_combinations);
    fprintf(csv_file, "# Essais par combinaison: 3\n");
    fprintf(csv_file, "# Époques max: %d\n", max_epochs);
    fprintf(csv_file, "# Toutes les métriques incluses: Accuracy, Precision, Recall, F1-Score, AUC-ROC\n");
    fprintf(csv_file, "#\n");
    
    // En-têtes de colonnes
    fprintf(csv_file, "Rang,Methode,Optimiseur,Activation,Combinaison_Complete,");
    fprintf(csv_file, "Avg_Accuracy_Pct,Avg_Precision_Pct,Avg_Recall_Pct,Avg_F1_Score_Pct,Avg_AUC_ROC_Pct,");
    fprintf(csv_file, "Best_Accuracy_Pct,Best_Precision_Pct,Best_Recall_Pct,Best_F1_Score_Pct,Best_AUC_ROC_Pct,");
    fprintf(csv_file, "Convergence_Count,Total_Trials,Taux_Convergence_Pct\n");
    
    // Données avec toutes les métriques
    for (int i = 0; i < result_count; i++) {
        fprintf(csv_file, "%d,%s,%s,%s,%s,",
               i + 1,
               results[i].method,
               results[i].optimizer,
               results[i].activation,
               results[i].full_name);
        
        // Métriques moyennes
        fprintf(csv_file, "%.2f,%.2f,%.2f,%.2f,%.2f,",
               results[i].avg_accuracy * 100,
               results[i].avg_precision * 100,
               results[i].avg_recall * 100,
               results[i].avg_f1_score * 100,
               results[i].avg_auc_roc * 100);
        
        // Meilleures métriques
        fprintf(csv_file, "%.2f,%.2f,%.2f,%.2f,%.2f,",
               results[i].best_accuracy * 100,
               results[i].best_precision * 100,
               results[i].best_recall * 100,
               results[i].best_f1_score * 100,
               results[i].best_auc_roc * 100);
        
        // Informations de convergence
        fprintf(csv_file, "%d,%d,%.2f\n",
               results[i].convergence_count,
               results[i].total_trials,
               results[i].convergence_rate * 100);
    }
    
    fclose(csv_file);
    printf("✅ Résultats exportés vers : %s\n", csv_filename);
    printf("📊 %d combinaisons sauvegardées avec toutes les métriques\n", result_count);
    printf("🏥 Métriques incluses : Accuracy, Precision, Recall, F1-Score, AUC-ROC\n");
    
    return 1;
} 