#ifndef ADAPTIVE_OPTIMIZER_H
#define ADAPTIVE_OPTIMIZER_H

#include "rich_config.h"

// Structure pour les paramètres adaptatifs en temps réel
typedef struct {
    double learning_rate;
    double momentum;
    double dropout_rate;
    double class_weight_ratio;
    int batch_size;
    
    // Historique des performances
    double best_accuracy;
    double current_accuracy;
    double previous_accuracy;
    
    // Compteurs d'adaptation
    int stagnation_epochs;
    int improvement_epochs;
    int total_epochs;
    int adaptation_count;
    
    // Seuils d'adaptation
    double target_accuracy;
    double improvement_threshold;
    int max_stagnation;
    
} AdaptiveParams;

// Fonctions principales
AdaptiveParams* adaptive_init_params(const RichConfig* cfg);
void adaptive_free_params(AdaptiveParams* params);

// Fonctions d'adaptation
void adaptive_update_performance(AdaptiveParams* params, double accuracy);
void adaptive_adjust_parameters(AdaptiveParams* params);

// Fonctions de configuration
void adaptive_generate_config(const AdaptiveParams* params, const char* base_config_path, 
                             const char* output_config_path, int iteration);

// Fonction principale d'optimisation intégrée
int run_adaptive_optimization(const RichConfig* cfg, const char* config_path);

#endif // ADAPTIVE_OPTIMIZER_H 