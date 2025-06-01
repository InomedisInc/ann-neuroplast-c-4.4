#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>

// Types de progression avec couleurs différentes
typedef enum {
    PROGRESS_GENERAL,      // Cyan - progression globale
    PROGRESS_TRIALS,       // Jaune - essais 
    PROGRESS_EPOCHS,       // Vert - époques
    PROGRESS_ITERATIONS    // Magenta - itérations
} ProgressType;

// Structure pour les métriques de progression
typedef struct {
    float loss;
    float accuracy;
    float learning_rate;
} ProgressMetrics;

// Structure pour une barre de progression individuelle
typedef struct {
    ProgressType type;
    char label[256];
    int current;
    int total;
    int width;
    int line_position;
    bool is_active;
    float last_loss;
    float last_accuracy;
    float last_lr;
} ProgressBar;

// Structure pour gérer plusieurs barres de progression
typedef struct {
    ProgressBar bars[20];  // Maximum 20 barres simultanées
    int num_bars;
    int base_line;         // Ligne de base pour l'affichage
    int info_zone_start;   // Ligne de début de la zone d'informations
    int max_bars;          // Nombre maximum de barres à afficher simultanément
} ProgressManager;

// Codes couleur ANSI
#define CYAN    "\033[96m"
#define YELLOW  "\033[93m"
#define GREEN   "\033[92m"
#define MAGENTA "\033[95m"
#define BOLD    "\033[1m"
#define RESET   "\033[0m"

// Caractères de progression
#define PROGRESS_CHAR "█"  // Caractère plein
#define EMPTY_CHAR    "░"  // Caractère vide

// Macros pour l'affichage
#define RESET_COLOR "\033[0m"

// Fonctions de base
void progress_init_manager(ProgressManager *manager);
void progress_init_manager_with_offset(ProgressManager *manager, int line_offset);
int progress_add_bar(ProgressManager *manager, ProgressType type, const char *label, int total, int width);
const char* progress_get_color(ProgressType type);
void progress_move_cursor(int line, int col);
void progress_hide_cursor(void);
void progress_show_cursor(void);
void progress_clear_line(int line);

// Fonctions globales
void progress_global_init(void);
void progress_global_init_with_offset(int offset);
void progress_global_cleanup(void);
void progress_show_header(const char* test_description, int num_combinations, int num_trials, int max_epochs);

// Nouvelles fonctions pour la gestion séparée des zones d'affichage
void progress_init_dual_zone(const char* header_title, int num_combinations, int num_trials, int max_epochs);
void progress_set_info_zone_line(int start_line);
int progress_get_info_zone_start(void);
void progress_clear_info_zone(void);
void progress_reserve_lines(int num_lines);

// Nouvelles fonctions pour l'affichage organisé par blocs
void progress_display_combination_header(int current_combo, int total_combos, 
                                       const char* method, const char* optimizer, const char* activation);
void progress_display_network_info(const char* architecture, const char* dataset_info, 
                                 float learning_rate, float* class_weights);
void progress_display_epoch_info(int epoch, int max_epochs, float loss, float accuracy, 
                                float precision, float recall, float f1_score);
void progress_display_trial_summary(int trial, int max_trials, float best_accuracy, 
                                   float best_f1, int convergence_epoch);
void progress_display_combination_summary(float avg_f1, float best_f1, int convergence_count, int total_trials);
void progress_prepare_next_combination(void);

// Anciennes fonctions (compatibilité)
int progress_global_add(ProgressType type, const char *label, int total, int width);
void progress_global_update(int bar_id, int current, float loss, float accuracy, float lr);
void progress_global_finish(int bar_id);
void progress_global_clear(void);
void progress_clear_temporary_bars(void);

// Nouvelles fonctions API simplifiée
int progress_create_bar(ProgressType type, const char* label, int max_value, ProgressMetrics* metrics);
void progress_update_bar(int bar_id, int current_value, ProgressMetrics* metrics);
void progress_remove_bar(int bar_id);
void progress_cleanup_all(void);

// Fonction utilitaire pour obtenir la position du curseur
int progress_get_current_line(void);

#endif // PROGRESS_BAR_H 