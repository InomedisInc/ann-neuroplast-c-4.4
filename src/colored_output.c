#include "colored_output.h"
#include <stdio.h>

// Variables globales pour la gestion des barres de progression
static int progress_mode_enabled = 0;
static int safe_output_line = 25;  // Ligne sûre pour l'affichage (après les barres de progression)

// Fonctions d'affichage coloré
void print_info(const char *message) {
    printf("%s[INFO]%s %s\n", COLOR_BLUE, COLOR_RESET, message);
}

void print_success(const char *message) {
    printf("%s[SUCCESS]%s %s\n", COLOR_GREEN, COLOR_RESET, message);
}

void print_warning(const char *message) {
    printf("%s[WARNING]%s %s\n", COLOR_YELLOW, COLOR_RESET, message);
}

void print_error(const char *message) {
    printf("%s[ERROR]%s %s\n", COLOR_RED, COLOR_RESET, message);
}

void print_network_info(const char *message) {
    printf("%s[NETWORK]%s %s\n", COLOR_MAGENTA, COLOR_RESET, message);
}

void print_dataset_info(const char *message) {
    printf("%s[DATASET]%s %s\n", COLOR_CYAN, COLOR_RESET, message);
}

void print_dataset_success(const char *message) {
    printf("%s[SUCCESS]%s %s\n", COLOR_GREEN, COLOR_RESET, message);
}

void print_dataset_error(const char *message) {
    printf("%s[ERROR]%s %s\n", COLOR_RED, COLOR_RESET, message);
}

// Fonction helper pour l'affichage sécurisé avec couleurs améliorées
static void print_safe(const char *prefix, const char *color, const char *message) {
    if (progress_mode_enabled) {
        // Sauvegarder le curseur
        printf("\033[s");
        // Aller à la ligne sûre
        printf("\033[%d;1H", safe_output_line);
        // Effacer la ligne
        printf("\033[K");
        // Afficher le message avec style amélioré
        printf("%s\033[1m%s\033[0m %s", color, prefix, message);
        // Restaurer le curseur
        printf("\033[u");
        fflush(stdout);
        // Passer à la ligne suivante pour le prochain message
        safe_output_line++;
    } else {
        // Mode normal avec couleurs améliorées
        printf("%s\033[1m%s\033[0m %s\n", color, prefix, message);
    }
}

// Nouvelles fonctions qui évitent les superpositions avec les barres de progression
void print_info_safe(const char *message) {
    print_safe("ℹ️  [INFO]", "\033[94m", message);
}

void print_success_safe(const char *message) {
    print_safe("✅ [SUCCESS]", "\033[92m", message);
}

void print_warning_safe(const char *message) {
    print_safe("⚠️  [WARNING]", "\033[93m", message);
}

void print_error_safe(const char *message) {
    print_safe("❌ [ERROR]", "\033[91m", message);
}

void print_network_info_safe(const char *message) {
    print_safe("🧠 [NETWORK]", "\033[95m", message);
}

void print_dataset_info_safe(const char *message) {
    print_safe("📊 [DATASET]", "\033[96m", message);
}

// Contrôle des logs pendant les barres de progression
void colored_output_set_progress_mode(int enabled) {
    progress_mode_enabled = enabled;
    if (enabled) {
        safe_output_line = 25;  // Réinitialiser la ligne sûre
    }
}

void colored_output_set_safe_line(int line) {
    safe_output_line = line;
} 