#ifndef COLORED_OUTPUT_H
#define COLORED_OUTPUT_H

// Codes couleur ANSI pour l'affichage
#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"      // Erreurs
#define COLOR_GREEN   "\033[32m"      // Succès
#define COLOR_YELLOW  "\033[33m"      // Avertissements
#define COLOR_BLUE    "\033[34m"      // Informations
#define COLOR_MAGENTA "\033[35m"      // Réseau/Architecture
#define COLOR_CYAN    "\033[36m"      // Dataset/Données
#define COLOR_WHITE   "\033[37m"      // Général

// Fonctions d'affichage coloré
void print_info(const char *message);
void print_success(const char *message);
void print_warning(const char *message);
void print_error(const char *message);
void print_network_info(const char *message);
void print_dataset_info(const char *message);
void print_dataset_success(const char *message);
void print_dataset_error(const char *message);

// Nouvelles fonctions qui évitent les superpositions avec les barres de progression
void print_info_safe(const char *message);
void print_success_safe(const char *message);
void print_warning_safe(const char *message);
void print_error_safe(const char *message);
void print_network_info_safe(const char *message);
void print_dataset_info_safe(const char *message);

// Contrôle des logs pendant les barres de progression
void colored_output_set_progress_mode(int enabled);
void colored_output_set_safe_line(int line);

#endif // COLORED_OUTPUT_H 