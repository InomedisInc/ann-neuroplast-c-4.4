#include "rich_config.h"
#include "yaml_parser_rich.h"
#include <string.h>
#include <stdbool.h>

// Fonction utilitaire pour fusionner les paramètres
static void merge_params(Param *dest_params, int *dest_num_params, 
                        const Param *src_params, int src_num_params) {
    // Pour chaque paramètre source
    for (int i = 0; i < src_num_params; i++) {
        bool found = false;
        // Cherche si le paramètre existe déjà dans la destination
        for (int j = 0; j < *dest_num_params; j++) {
            if (strcmp(dest_params[j].key, src_params[i].key) == 0) {
                // Si oui, met à jour la valeur
                dest_params[j].value = src_params[i].value;
                found = true;
                break;
            }
        }
        // Si non et qu'il y a de la place, ajoute le paramètre
        if (!found && *dest_num_params < MAX_PARAMS) {
            strncpy(dest_params[*dest_num_params].key, src_params[i].key, MAX_NAME-1);
            dest_params[*dest_num_params].key[MAX_NAME-1] = '\0';
            dest_params[*dest_num_params].value = src_params[i].value;
            (*dest_num_params)++;
        }
    }
}

void merge_rich_configs(RichConfig *dest, const RichConfig *src) {
    // Fusionne les paramètres globaux si non nuls dans la source
    if (src->dataset[0] != '\0') {
        strncpy(dest->dataset, src->dataset, sizeof(dest->dataset)-1);
        dest->dataset[sizeof(dest->dataset)-1] = '\0';
    }
    if (src->batch_size > 0) dest->batch_size = src->batch_size;
    if (src->max_epochs > 0) dest->max_epochs = src->max_epochs;
    if (src->learning_rate > 0) dest->learning_rate = src->learning_rate;

    // Fusionne les méthodes neuroplastiques
    for (int i = 0; i < src->num_neuroplast_methods; i++) {
        bool found = false;
        for (int j = 0; j < dest->num_neuroplast_methods; j++) {
            if (strcmp(dest->neuroplast_methods[j].name, src->neuroplast_methods[i].name) == 0) {
                found = true;
                break;
            }
        }
        if (!found && dest->num_neuroplast_methods < MAX_METHODS) {
            strncpy(dest->neuroplast_methods[dest->num_neuroplast_methods].name,
                   src->neuroplast_methods[i].name, MAX_NAME-1);
            dest->neuroplast_methods[dest->num_neuroplast_methods].name[MAX_NAME-1] = '\0';
            dest->num_neuroplast_methods++;
        }
    }

    // Fusionne les activations
    for (int i = 0; i < src->num_activations; i++) {
        bool found = false;
        for (int j = 0; j < dest->num_activations; j++) {
            if (strcmp(dest->activations[j].name, src->activations[i].name) == 0) {
                // Met à jour les champs existants
                if (src->activations[i].optimization_method[0] != '\0') {
                    strncpy(dest->activations[j].optimization_method,
                           src->activations[i].optimization_method, MAX_NAME-1);
                    dest->activations[j].optimization_method[MAX_NAME-1] = '\0';
                }
                if (src->activations[i].optimized_with[0] != '\0') {
                    strncpy(dest->activations[j].optimized_with,
                           src->activations[i].optimized_with, MAX_NAME-1);
                    dest->activations[j].optimized_with[MAX_NAME-1] = '\0';
                }
                merge_params(dest->activations[j].params, &dest->activations[j].num_params,
                           src->activations[i].params, src->activations[i].num_params);
                found = true;
                break;
            }
        }
        if (!found && dest->num_activations < MAX_METHODS) {
            memcpy(&dest->activations[dest->num_activations],
                   &src->activations[i], sizeof(Activation));
            dest->num_activations++;
        }
    }

    // Fusionne les optimiseurs
    for (int i = 0; i < src->num_optimizers; i++) {
        bool found = false;
        for (int j = 0; j < dest->num_optimizers; j++) {
            if (strcmp(dest->optimizers[j].name, src->optimizers[i].name) == 0) {
                merge_params(dest->optimizers[j].params, &dest->optimizers[j].num_params,
                           src->optimizers[i].params, src->optimizers[i].num_params);
                found = true;
                break;
            }
        }
        if (!found && dest->num_optimizers < MAX_METHODS) {
            memcpy(&dest->optimizers[dest->num_optimizers],
                   &src->optimizers[i], sizeof(OptimizerDef));
            dest->num_optimizers++;
        }
    }

    // Fusionne les métriques
    for (int i = 0; i < src->num_metrics; i++) {
        bool found = false;
        for (int j = 0; j < dest->num_metrics; j++) {
            if (strcmp(dest->metrics[j].name, src->metrics[i].name) == 0) {
                found = true;
                break;
            }
        }
        if (!found && dest->num_metrics < MAX_METHODS) {
            strncpy(dest->metrics[dest->num_metrics].name,
                   src->metrics[i].name, MAX_NAME-1);
            dest->metrics[dest->num_metrics].name[MAX_NAME-1] = '\0';
            dest->num_metrics++;
        }
    }
}

int parse_yaml_rich_config(const char *filename, RichConfig *cfg) {
    return parse_yaml_rich(filename, cfg);
} 