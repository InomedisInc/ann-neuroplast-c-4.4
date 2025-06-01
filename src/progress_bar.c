#include "progress_bar.h"
#include "colored_output.h"
#include <string.h>
#include <math.h>

// Instance globale du gestionnaire de progression
static ProgressManager g_manager = {0};

void progress_init_manager(ProgressManager *manager) {
    progress_init_manager_with_offset(manager, 0);
}

void progress_init_manager_with_offset(ProgressManager *manager, int line_offset) {
    if (!manager) manager = &g_manager;
    
    memset(manager, 0, sizeof(ProgressManager));
    manager->base_line = line_offset;
    manager->info_zone_start = line_offset + 5;  // Zone d'infos commence 5 lignes après les barres
    manager->max_bars = 4;  // Maximum 4 barres simultanées
    progress_hide_cursor();
}

int progress_add_bar(ProgressManager *manager, ProgressType type, const char *label, int total, int width) {
    if (!manager) manager = &g_manager;
    
    if (manager->num_bars >= 20) {
        return -1; // Trop de barres
    }
    
    int bar_id = manager->num_bars;
    ProgressBar *bar = &manager->bars[bar_id];
    
    bar->type = type;
    strncpy(bar->label, label, sizeof(bar->label) - 1);
    bar->label[sizeof(bar->label) - 1] = '\0';
    bar->current = 0;
    bar->total = total;
    bar->width = width;
    
    // Positionnement fixe selon le type de barre (après l'en-tête)
    switch (type) {
        case PROGRESS_GENERAL:
            bar->line_position = manager->base_line + 1; // Ligne "GENERAL:"
            break;
        case PROGRESS_TRIALS:
            bar->line_position = manager->base_line + 2; // Ligne "ESSAIS:"
            break;
        case PROGRESS_EPOCHS:
            bar->line_position = manager->base_line + 3; // Ligne "ÉPOQUES:"
            break;
        case PROGRESS_ITERATIONS:
            bar->line_position = manager->base_line + 4; // Ligne libre
            break;
        default:
            bar->line_position = manager->base_line + manager->num_bars + 1;
            break;
    }
    
    bar->is_active = true;
    bar->last_loss = 0.0f;
    bar->last_accuracy = 0.0f;
    bar->last_lr = 0.0f;
    
    manager->num_bars++;
    return bar_id;
}

const char* progress_get_color(ProgressType type) {
    switch (type) {
        case PROGRESS_GENERAL:    return CYAN;
        case PROGRESS_TRIALS:     return YELLOW;
        case PROGRESS_EPOCHS:     return GREEN;
        case PROGRESS_ITERATIONS: return MAGENTA;
        default:                  return RESET_COLOR;
    }
}

void progress_move_cursor(int line, int col) {
    printf("\033[%d;%dH", line + 1, col + 1);  // ANSI escape code pour positionner le curseur
}

void progress_hide_cursor(void) {
    printf("\033[?25l");  // Cache le curseur
    fflush(stdout);
}

void progress_show_cursor(void) {
    printf("\033[?25h");  // Affiche le curseur
    fflush(stdout);
}

void progress_clear_line(int line) {
    progress_move_cursor(line, 0);
    printf("\033[K");  // Efface la ligne à partir du curseur
}

void progress_update(ProgressManager *manager, int bar_id, int current, float loss, float accuracy, float lr) {
    if (!manager) manager = &g_manager;
    
    if (bar_id < 0 || bar_id >= manager->num_bars || !manager->bars[bar_id].is_active) {
        return;
    }
    
    ProgressBar *bar = &manager->bars[bar_id];
    bar->current = current;
    bar->last_loss = loss;
    bar->last_accuracy = accuracy;
    bar->last_lr = lr;
    
    // Calculer le pourcentage et la barre
    float percentage = (bar->total > 0) ? (float)current / bar->total : 0.0f;
    int filled_chars = (int)(percentage * bar->width);
    
    // Sauvegarder le curseur et aller à la ligne de la barre
    printf("\033[s");  // Sauvegarder position du curseur
    
    // Déterminer la ligne selon le type de barre (positions fixes)
    int target_line;
    switch (bar->type) {
        case PROGRESS_GENERAL:
            target_line = 12;  // Ligne GENERAL fixe
            break;
        case PROGRESS_TRIALS:
            target_line = 13;  // Ligne ESSAIS fixe
            break;
        case PROGRESS_EPOCHS:
            target_line = 14;  // Ligne EPOQUES fixe
            break;
        default:
            target_line = 15;  // Ligne par défaut
            break;
    }
    
    printf("\033[%d;1H", target_line);  // Aller à la ligne (1-indexed)
    printf("\033[K");  // Effacer toute la ligne
    
    // Préfixe selon le type avec couleurs fixes selon la demande utilisateur
    const char *prefix = "";
    const char *color = "";
    switch (bar->type) {
        case PROGRESS_GENERAL: 
            prefix = "GENERAL: "; 
            color = "\033[94m"; // Bleu pour GENERAL
            break;
        case PROGRESS_TRIALS:  
            prefix = "ESSAIS:  "; 
            color = "\033[93m"; // Jaune pour ESSAIS
            break;
        case PROGRESS_EPOCHS:  
            prefix = "EPOQUES: "; 
            color = "\033[92m"; // Vert pour EPOQUES
            break;
        default:               
            prefix = "        "; 
            color = "\033[95m"; // Magenta pour autres
            break;
    }
    
    // Afficher avec couleur et style amélioré (largeur fixe pour éviter les décalages)
    printf("%s\033[1m%-8s\033[0m ", color, prefix);
    
    // Afficher la barre de progression avec couleurs fixes selon le type de barre
    printf("[");
    
    // Déterminer la couleur de remplissage selon le type de barre (couleurs fixes)
    const char *fill_color = "";
    switch (bar->type) {
        case PROGRESS_GENERAL:
            fill_color = "\033[94m"; // Bleu pour GENERAL
            break;
        case PROGRESS_TRIALS:
            fill_color = "\033[93m"; // Jaune pour ESSAIS
            break;
        case PROGRESS_EPOCHS:
            fill_color = "\033[92m"; // Vert pour EPOQUES
            break;
        default:
            fill_color = "\033[95m"; // Magenta pour autres
            break;
    }
    
    for (int i = 0; i < bar->width; i++) {
        if (i < filled_chars) {
            // Utiliser la couleur fixe du type de barre
            printf("%s█\033[0m", fill_color);
        } else {
            printf("\033[90m░\033[0m"); // Gris clair pour vide
        }
    }
    printf("]");
    
    // Afficher le pourcentage avec largeur fixe
    printf(" %s%6.1f%%\033[0m", color, percentage * 100.0f);
    
    // Afficher le label avec largeur limitée
    printf(" %s\033[94m(%d/%d)\033[0m", bar->label, current, bar->total);
    
    // Afficher les métriques si disponibles avec couleurs distinctes et format compact
    if (loss > 0.0f || accuracy > 0.0f) {
        printf(" | \033[91mLoss:\033[0m \033[93m%.4f\033[0m | \033[92mAcc:\033[0m \033[96m%.1f%%\033[0m", 
               loss, accuracy * 100.0f);
    }
    
    if (lr > 0.0f) {
        printf(" | \033[95mLR:\033[0m \033[94m%.4f\033[0m", lr);
    }
    
    printf("\033[0m"); // Reset couleur
    
    // Restaurer le curseur à sa position d'origine
    printf("\033[u");  // Restaurer position du curseur
    fflush(stdout);
}

void progress_finish(ProgressManager *manager, int bar_id) {
    if (!manager) manager = &g_manager;
    
    if (bar_id < 0 || bar_id >= manager->num_bars) {
        return;
    }
    
    ProgressBar *bar = &manager->bars[bar_id];
    
    // Pour les barres temporaires (itérations), on efface simplement la ligne
    if (bar->type == PROGRESS_ITERATIONS) {
        progress_clear_line(bar->line_position);
        bar->is_active = false;
        return;
    }
    
    // Pour les autres barres, on affiche la version finale
    progress_clear_line(bar->line_position);
    progress_move_cursor(bar->line_position, 0);
    
    // Préfixe selon le type avec couleurs fixes
    const char *prefix = "";
    const char *color = "";
    switch (bar->type) {
        case PROGRESS_GENERAL: 
            prefix = "GENERAL: "; 
            color = "\033[94m"; // Bleu pour GENERAL
            break;
        case PROGRESS_TRIALS:  
            prefix = "ESSAIS:  "; 
            color = "\033[93m"; // Jaune pour ESSAIS
            break;
        case PROGRESS_EPOCHS:  
            prefix = "EPOQUES: "; 
            color = "\033[92m"; // Vert pour EPOQUES
            break;
        default:               
            prefix = "        "; 
            color = "\033[95m"; // Magenta pour autres
            break;
    }
    
    printf("%s%s%s[", color, BOLD, prefix);
    
    // Afficher la barre complètement remplie avec la couleur fixe du type
    for (int i = 0; i < bar->width; i++) {
        printf("█");
    }
    
    printf("] ✓ TERMINE %s", bar->label);
    
    // Afficher les métriques finales si disponibles
    if (bar->last_loss > 0.0f || bar->last_accuracy > 0.0f) {
        printf(" | Loss: %.4f | Acc: %.1f%%", 
               bar->last_loss, bar->last_accuracy * 100.0f);
    }
    
    if (bar->last_lr > 0.0f) {
        printf(" | LR: %.4f", bar->last_lr);
    }
    
    printf("%s", RESET_COLOR);
    fflush(stdout);
    
    // Marquer comme terminé
    bar->is_active = false;
}

void progress_clear_all(ProgressManager *manager) {
    if (!manager) manager = &g_manager;
    
    for (int i = 0; i < manager->num_bars; i++) {
        if (manager->bars[i].is_active) {
            progress_clear_line(manager->bars[i].line_position);
        }
    }
    
    memset(manager, 0, sizeof(ProgressManager));
    progress_show_cursor();
}

void progress_display_all(ProgressManager *manager) {
    if (!manager) manager = &g_manager;
    
    for (int i = 0; i < manager->num_bars; i++) {
        if (manager->bars[i].is_active) {
            progress_update(manager, i, manager->bars[i].current, 
                          manager->bars[i].last_loss, 
                          manager->bars[i].last_accuracy, 
                          manager->bars[i].last_lr);
        }
    }
}

// Fonctions globales pour faciliter l'utilisation
void progress_global_init(void) {
    progress_init_manager(&g_manager);
}

void progress_global_init_with_offset(int line_offset) {
    progress_init_manager_with_offset(&g_manager, line_offset);
}

int progress_global_add(ProgressType type, const char *label, int total, int width) {
    return progress_add_bar(&g_manager, type, label, total, width);
}

void progress_global_update(int bar_id, int current, float loss, float accuracy, float lr) {
    progress_update(&g_manager, bar_id, current, loss, accuracy, lr);
}

void progress_global_finish(int bar_id) {
    progress_finish(&g_manager, bar_id);
}

void progress_global_clear(void) {
    progress_clear_all(&g_manager);
}

// Nouvelle fonction pour nettoyer seulement les barres temporaires (itérations)
void progress_clear_temporary_bars(void) {
    ProgressManager *manager = &g_manager;
    
    for (int i = 0; i < manager->num_bars; i++) {
        if (manager->bars[i].is_active && manager->bars[i].type == PROGRESS_ITERATIONS) {
            progress_clear_line(manager->bars[i].line_position);
            manager->bars[i].is_active = false;
        }
    }
}

// Fonction pour obtenir la position actuelle du curseur
int progress_get_current_line(void) {
    // Cette fonction utilise une approche simple :
    // On envoie une requête de position de curseur et on lit la réponse
    // Mais pour simplifier, on va utiliser une approche différente
    return 0; // Temporaire
}

void progress_global_cleanup(void) {
    progress_cleanup_all();
}

// FONCTIONS UTILITAIRES POUR L'AFFICHAGE ADAPTATIF (DÉCLARÉES AVANT UTILISATION)

// Fonction utilitaire pour calculer la largeur visible d'une chaîne (sans codes ANSI)
static int calculate_visible_length(const char* str) {
    int visible_length = 0;
    int in_escape = 0;
    
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] == '\033') {
            in_escape = 1;
        } else if (in_escape && str[i] == 'm') {
            in_escape = 0;
        } else if (!in_escape) {
            // Compter les caractères Unicode comme 1 caractère visuel
            if ((unsigned char)str[i] >= 0x80) {
                // Caractère Unicode multi-byte, on compte comme 1
                visible_length++;
                // Ignorer les bytes suivants de ce caractère Unicode
                while (i + 1 < strlen(str) && (unsigned char)str[i + 1] >= 0x80 && (unsigned char)str[i + 1] < 0xC0) {
                    i++;
                }
            } else {
                visible_length++;
            }
        }
    }
    
    return visible_length;
}

// Fonction pour créer une ligne de bordure adaptative
static void print_adaptive_border_line(const char* corner_left, const char* fill, const char* corner_right, int content_width) {
    int total_width = content_width + 4; // 2 espaces + 2 bordures
    if (total_width < 40) total_width = 40; // Largeur minimale
    if (total_width > 80) total_width = 80; // Largeur maximale
    
    printf("%s", corner_left);
    for (int i = 0; i < total_width - 2; i++) {
        printf("%s", fill);
    }
    printf("%s\n", corner_right);
}

// Fonction pour créer une ligne de contenu adaptative
static void print_adaptive_content_line(const char* content, const char* color, int min_content_width) {
    int visible_length = calculate_visible_length(content);
    int content_width = (visible_length > min_content_width) ? visible_length : min_content_width;
    int total_width = content_width + 4; // 2 espaces + 2 bordures
    if (total_width < 40) total_width = 40; // Largeur minimale
    if (total_width > 80) total_width = 80; // Largeur maximale
    
    int padding = total_width - 4 - visible_length; // Espace restant après le contenu
    int left_padding = padding / 2;
    int right_padding = padding - left_padding;
    
    printf("%s║\033[0m", color);
    
    // Espacement à gauche
    for (int i = 0; i < left_padding + 1; i++) {
        printf(" ");
    }
    
    // Contenu
    printf("%s", content);
    
    // Espacement à droite
    for (int i = 0; i < right_padding + 1; i++) {
        printf(" ");
    }
    
    printf("%s║\033[0m\n", color);
}

// Nouvelle fonction pour afficher un en-tête descriptif amélioré
void progress_show_header(const char* test_description, int num_combinations, int num_trials, int max_epochs) {
    // Effacer l'écran et positionner le curseur en haut
    printf("\033[2J\033[H");
    printf("\n");
    
    // Calculer la largeur nécessaire pour l'en-tête
    char title_content[256];
    snprintf(title_content, sizeof(title_content), "🧠 NEUROPLAST TRAINING SYSTEM 🧠");
    int title_width = calculate_visible_length(title_content);
    
    char test_content[256];
    snprintf(test_content, sizeof(test_content), "TEST EN COURS: %s", test_description);
    int test_width = calculate_visible_length(test_content);
    
    char config_title[] = "CONFIGURATION DE TEST:";
    int config_title_width = calculate_visible_length(config_title);
    
    char combo_info[256];
    snprintf(combo_info, sizeof(combo_info), "• Nombre de combinaisons: %d", num_combinations);
    int combo_width = calculate_visible_length(combo_info);
    
    char trials_info[256];
    snprintf(trials_info, sizeof(trials_info), "• Essais par combinaison: %d", num_trials);
    int trials_width = calculate_visible_length(trials_info);
    
    char epochs_info[256];
    snprintf(epochs_info, sizeof(epochs_info), "• Époques maximum: %d", max_epochs);
    int epochs_width = calculate_visible_length(epochs_info);
    
    char progress_info[] = "BARRES DE PROGRESSION (les 3 barres seront affichées ci-dessous):";
    int progress_info_width = calculate_visible_length(progress_info);
    
    char bar1_info[] = "- Progression générale des combinaisons";
    int bar1_width = calculate_visible_length(bar1_info);
    
    char bar2_info[] = "- Progression des essais par combinaison";
    int bar2_width = calculate_visible_length(bar2_info);
    
    char bar3_info[] = "- Progression des époques par essai";
    int bar3_width = calculate_visible_length(bar3_info);
    
    // Prendre la largeur maximale
    int max_width = title_width;
    if (test_width > max_width) max_width = test_width;
    if (config_title_width > max_width) max_width = config_title_width;
    if (combo_width > max_width) max_width = combo_width;
    if (trials_width > max_width) max_width = trials_width;
    if (epochs_width > max_width) max_width = epochs_width;
    if (progress_info_width > max_width) max_width = progress_info_width;
    if (bar1_width > max_width) max_width = bar1_width;
    if (bar2_width > max_width) max_width = bar2_width;
    if (bar3_width > max_width) max_width = bar3_width;
    
    // EN-TÊTE PRINCIPAL avec largeur adaptative
    printf("\033[96m");
    print_adaptive_border_line("╔", "═", "╗", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line("\033[92m🧠 NEUROPLAST TRAINING SYSTEM 🧠\033[0m", "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_border_line("╠", "═", "╣", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line(test_content, "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line("", "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line("\033[94mCONFIGURATION DE TEST:\033[0m", "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line(combo_info, "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line(trials_info, "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line(epochs_info, "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line("", "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line("\033[93mBARRES DE PROGRESSION\033[0m (les 3 barres seront affichées ci-dessous):", "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line("\033[94m- Progression générale des combinaisons\033[0m", "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line("\033[94m- Progression des essais par combinaison\033[0m", "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line("\033[94m- Progression des époques par essai\033[0m", "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_border_line("╚", "═", "╝", max_width);
    printf("\033[0m");
    printf("\n");
    
    // Réserve exactement 4 lignes pour les barres de progression avec couleurs
    printf("\033[94mGENERAL:\033[0m  \033[90m[En attente...]\033[0m\n");
    printf("\033[93mESSAIS:\033[0m   \033[90m[En attente...]\033[0m\n");  
    printf("\033[92mEPOQUES:\033[0m  \033[90m[En attente...]\033[0m\n");
    printf("\n");
    
    fflush(stdout);
}

// Nouvelles fonctions pour l'API simplifiée
int progress_create_bar(ProgressType type, const char* label, int max_value, ProgressMetrics* metrics) {
    (void)metrics; // Supprimer le warning de paramètre inutilisé
    return progress_global_add(type, label, max_value, 40);
}

void progress_update_bar(int bar_id, int current_value, ProgressMetrics* metrics) {
    float loss = (metrics && metrics->loss > 0) ? metrics->loss : 0.0f;
    float accuracy = (metrics && metrics->accuracy > 0) ? metrics->accuracy : 0.0f;
    float lr = (metrics && metrics->learning_rate > 0) ? metrics->learning_rate : 0.0f;
    
    progress_global_update(bar_id, current_value, loss, accuracy, lr);
}

void progress_remove_bar(int bar_id) {
    progress_global_finish(bar_id);
}

void progress_cleanup_all(void) {
    progress_global_clear();
}

// NOUVELLES FONCTIONS POUR LA GESTION SÉPARÉE DES ZONES D'AFFICHAGE

void progress_init_dual_zone(const char* header_title, int num_combinations, int num_trials, int max_epochs) {
    // Effacer l'écran et positionner le curseur en haut
    printf("\033[2J\033[H");
    
    // Calculer la largeur nécessaire pour l'en-tête
    char title_content[256];
    snprintf(title_content, sizeof(title_content), "🧠 NEUROPLAST TRAINING SYSTEM 🧠");
    int title_width = calculate_visible_length(title_content);
    
    char test_content[256];
    snprintf(test_content, sizeof(test_content), "TEST: %s", header_title);
    int test_width = calculate_visible_length(test_content);
    
    char config_content[256];
    snprintf(config_content, sizeof(config_content), "Configuration: %d combinaisons × %d essais × %d époques max", 
             num_combinations, num_trials, max_epochs);
    int config_width = calculate_visible_length(config_content);
    
    // Prendre la largeur maximale
    int max_width = title_width;
    if (test_width > max_width) max_width = test_width;
    if (config_width > max_width) max_width = config_width;
    
    // EN-TÊTE PRINCIPAL avec largeur adaptative
    printf("\033[96m");
    print_adaptive_border_line("╔", "═", "╗", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line("\033[92m🧠 NEUROPLAST TRAINING SYSTEM 🧠\033[0m", "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_border_line("╠", "═", "╣", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line(test_content, "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line("", "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_content_line(config_content, "\033[96m", max_width);
    printf("\033[0m");
    
    printf("\033[96m");
    print_adaptive_border_line("╚", "═", "╝", max_width);
    printf("\033[0m");
    printf("\n");
    
    // SECTION BARRES DE PROGRESSION avec largeur adaptative
    char progress_title[] = "📊 BARRES DE PROGRESSION 📊";
    int progress_width = calculate_visible_length(progress_title);
    
    printf("\033[95m");
    print_adaptive_border_line("╔", "═", "╗", progress_width);
    printf("\033[0m");
    
    printf("\033[95m");
    print_adaptive_content_line("\033[93m📊 BARRES DE PROGRESSION 📊\033[0m", "\033[95m", progress_width);
    printf("\033[0m");
    
    printf("\033[95m");
    print_adaptive_border_line("╚", "═", "╝", progress_width);
    printf("\033[0m");
    
    // Lignes fixes pour les barres de progression (avec espacement réservé)
    printf("\033[94mGENERAL:\033[0m  \033[90m[░░░░░░░░░░░░░░░░░░░░░░░░░]   0.0%% Initialisation...\033[0m\n");              // Ligne 12 - Bleu
    printf("\033[93mESSAIS:\033[0m   \033[90m[░░░░░░░░░░░░░░░░░░░░░░░░░]   0.0%% Initialisation...\033[0m\n");              // Ligne 13 - Jaune  
    printf("\033[92mEPOQUES:\033[0m  \033[90m[░░░░░░░░░░░░░░░░░░░░░]   0.0%% Initialisation...\033[0m\n");              // Ligne 14 - Vert
    printf("\n");                                            // Ligne 15
    
    // SECTION INFORMATIONS D'ENTRAÎNEMENT avec largeur adaptative
    char info_title[] = "🔥 INFORMATIONS D'ENTRAÎNEMENT 🔥";
    int info_width = calculate_visible_length(info_title);
    
    printf("\033[94m");
    print_adaptive_border_line("╔", "═", "╗", info_width);
    printf("\033[0m");
    
    printf("\033[94m");
    print_adaptive_content_line("\033[91m🔥 INFORMATIONS D'ENTRAÎNEMENT 🔥\033[0m", "\033[94m", info_width);
    printf("\033[0m");
    
    printf("\033[94m");
    print_adaptive_border_line("╚", "═", "╝", info_width);
    printf("\033[0m");
    printf("\n");  // Ligne 19 - ici commence la zone d'informations
    
    // Initialiser le gestionnaire de barres avec les bonnes positions
    progress_global_init_with_offset(11);  // Les barres commencent à la ligne 12
    
    // Configurer la zone d'informations dans le système d'affichage sécurisé
    colored_output_set_progress_mode(1);
    colored_output_set_safe_line(20);  // Les informations commencent à la ligne 20
    
    // Stocker la ligne de début de la zone d'informations
    g_manager.info_zone_start = 20;
    
    fflush(stdout);
}

void progress_display_combination_header(int current_combo, int total_combos, 
                                       const char* method, const char* optimizer, const char* activation) {
    // Aller à la zone d'informations et afficher l'en-tête de combinaison
    printf("\033[s");  // Sauvegarder curseur
    printf("\033[%d;1H", g_manager.info_zone_start);  // Aller à la zone d'infos
    
    // Effacer plusieurs lignes pour faire de la place
    for (int i = 0; i < 12; i++) {
        printf("\033[K\n");  // Effacer ligne et passer à la suivante
    }
    
    // Revenir au début de la zone d'infos
    printf("\033[%d;1H", g_manager.info_zone_start);
    
    // Calculer la largeur nécessaire pour l'en-tête de combinaison
    char combo_title[256];
    snprintf(combo_title, sizeof(combo_title), "🧪 COMBINAISON %d/%d 🧪", current_combo, total_combos);
    int combo_title_width = calculate_visible_length(combo_title);
    
    char combo_details[256];
    snprintf(combo_details, sizeof(combo_details), "🧠 Méthode: %s     ⚡ Optimiseur: %s     🎯 Activation: %s", 
             method, optimizer, activation);
    int combo_details_width = calculate_visible_length(combo_details);
    
    int combo_max_width = combo_title_width > combo_details_width ? combo_title_width : combo_details_width;
    
    // Afficher l'en-tête de la combinaison avec largeur adaptative
    printf("\033[93m");
    print_adaptive_border_line("╔", "═", "╗", combo_max_width);
    printf("\033[0m");
    
    printf("\033[93m");
    print_adaptive_content_line(combo_title, "\033[93m", combo_max_width);
    printf("\033[0m");
    
    printf("\033[93m");
    print_adaptive_border_line("╠", "═", "╣", combo_max_width);
    printf("\033[0m");
    
    printf("\033[93m");
    print_adaptive_content_line(combo_details, "\033[93m", combo_max_width);
    printf("\033[0m");
    
    printf("\033[93m");
    print_adaptive_border_line("╚", "═", "╝", combo_max_width);
    printf("\033[0m");
    printf("\n");
    
    // Mettre à jour la ligne de début pour les prochains messages (format plus compact)
    g_manager.info_zone_start += 7;
    colored_output_set_safe_line(g_manager.info_zone_start);
    
    printf("\033[u");  // Restaurer curseur
    fflush(stdout);
}

void progress_display_network_info(const char* architecture, const char* dataset_info, 
                                 float learning_rate, float* class_weights) {
    // Afficher les informations du réseau dans la zone d'informations avec couleurs
    char network_info[256];
    snprintf(network_info, sizeof(network_info), 
            "🏗️  Architecture: %s", architecture);
    print_info_safe(network_info);
    
    char dataset_msg[256];
    snprintf(dataset_msg, sizeof(dataset_msg), 
            "📊 Dataset: %s", dataset_info);
    print_dataset_info_safe(dataset_msg);
    
    char lr_info[256];
    snprintf(lr_info, sizeof(lr_info), 
            "⚡ Learning Rate: %.6f", learning_rate);
    print_network_info_safe(lr_info);
    
    if (class_weights) {
        char weights_info[256];
        snprintf(weights_info, sizeof(weights_info), 
                "⚖️  Class Weights: [%.2f, %.2f]", class_weights[0], class_weights[1]);
        print_network_info_safe(weights_info);
    }
    
    // Ligne de séparation colorée avant les époques
    print_info_safe("\033[90m────────────────────────────────────────────────────────────────────────────────\033[0m");
}

void progress_display_epoch_info(int epoch, int max_epochs, float loss, float accuracy, 
                                float precision, float recall, float f1_score) {
    char epoch_info[256];
    snprintf(epoch_info, sizeof(epoch_info), 
            "📈 Époque \033[96m%d/%d\033[0m | \033[91mLoss:\033[0m \033[93m%.4f\033[0m | \033[92mAcc:\033[0m \033[96m%.1f%%\033[0m | \033[95mF1:\033[0m \033[94m%.1f%%\033[0m | \033[93mPrec:\033[0m \033[92m%.1f%%\033[0m | \033[91mRec:\033[0m \033[96m%.1f%%\033[0m",
            epoch + 1, max_epochs, loss, accuracy * 100, f1_score * 100, 
            precision * 100, recall * 100);
    print_info_safe(epoch_info);
}

void progress_display_trial_summary(int trial, int max_trials, float best_accuracy, 
                                   float best_f1, int convergence_epoch) {
    char trial_summary[256];
    if (convergence_epoch >= 0) {
        snprintf(trial_summary, sizeof(trial_summary), 
                "✅ Essai \033[96m%d/%d\033[0m terminé | \033[92mMeilleure Acc:\033[0m \033[96m%.1f%%\033[0m | \033[95mMeilleur F1:\033[0m \033[94m%.1f%%\033[0m | \033[93mConvergence:\033[0m \033[92mépoque %d\033[0m",
                trial + 1, max_trials, best_accuracy * 100, best_f1 * 100, convergence_epoch + 1);
    } else {
        snprintf(trial_summary, sizeof(trial_summary), 
                "⚠️  Essai \033[96m%d/%d\033[0m terminé | \033[92mMeilleure Acc:\033[0m \033[96m%.1f%%\033[0m | \033[95mMeilleur F1:\033[0m \033[94m%.1f%%\033[0m | \033[91mPas de convergence\033[0m",
                trial + 1, max_trials, best_accuracy * 100, best_f1 * 100);
    }
    print_success_safe(trial_summary);
}

void progress_display_combination_summary(float avg_f1, float best_f1, int convergence_count, int total_trials) {
    // Ligne de séparation colorée avec largeur adaptative
    char separator[256];
    snprintf(separator, sizeof(separator), "────────────────────────────────────────────────────────────────────────────────");
    int sep_width = 60; // Largeur fixe pour les séparateurs
    
    // Créer une ligne de séparation adaptative
    printf("\033[90m");
    for (int i = 0; i < sep_width; i++) {
        printf("─");
    }
    printf("\033[0m\n");
    
    // Créer le contenu du résumé
    char summary_content[512];
    snprintf(summary_content, sizeof(summary_content), 
            "🏆 RÉSUMÉ COMBINAISON | F1 Moyen: %.1f%% | Meilleur F1: %.1f%% | Convergence: %d/%d (%.0f%%)",
            avg_f1 * 100, best_f1 * 100, convergence_count, total_trials, 
            ((float)convergence_count / total_trials) * 100);
    
    int summary_width = calculate_visible_length(summary_content);
    
    // Afficher le résumé avec boîte adaptative
    printf("\033[92m");
    print_adaptive_border_line("╔", "═", "╗", summary_width);
    printf("\033[0m");
    
    printf("\033[92m");
    print_adaptive_content_line(summary_content, "\033[92m", summary_width);
    printf("\033[0m");
    
    printf("\033[92m");
    print_adaptive_border_line("╚", "═", "╝", summary_width);
    printf("\033[0m");
    
    // Double ligne de séparation colorée pour marquer la fin de la combinaison
    printf("\033[93m");
    for (int i = 0; i < summary_width + 4; i++) {
        printf("═");
    }
    printf("\033[0m\n");
    printf("\n");  // Ligne vide pour aérer
}

void progress_prepare_next_combination(void) {
    // Effacer complètement la zone d'informations avant la prochaine combinaison
    printf("\033[s");  // Sauvegarder curseur
    
    // Effacer toute la zone d'informations (20 lignes à partir de la ligne 20)
    for (int line = 20; line < 40; line++) {
        printf("\033[%d;1H", line);  // Aller à la ligne
        printf("\033[K");  // Effacer toute la ligne
    }
    
    // Réinitialiser la zone d'informations pour la prochaine combinaison
    g_manager.info_zone_start = 20;  // Revenir au début de la zone d'infos
    colored_output_set_safe_line(20);
    
    // Repositionner le curseur au début de la zone d'informations
    printf("\033[20;1H");
    
    printf("\033[u");  // Restaurer curseur
    fflush(stdout);
}

void progress_set_info_zone_line(int start_line) {
    if (start_line > 0) {
        g_manager.info_zone_start = start_line;
        colored_output_set_safe_line(start_line);
    }
}

int progress_get_info_zone_start(void) {
    return g_manager.info_zone_start;
}

void progress_clear_info_zone(void) {
    // Effacer toute la zone d'informations en partant de info_zone_start
    for (int line = g_manager.info_zone_start; line < g_manager.info_zone_start + 20; line++) {
        progress_clear_line(line);
    }
    
    // Remettre le curseur au début de la zone d'informations
    progress_move_cursor(g_manager.info_zone_start, 0);
    colored_output_set_safe_line(g_manager.info_zone_start);
}

void progress_reserve_lines(int num_lines) {
    // Réserver des lignes supplémentaires pour l'affichage de résultats
    printf("\n");
    for (int i = 0; i < num_lines; i++) {
        printf("\n");
    }
} 