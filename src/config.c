#include "config.h"
#include <stdlib.h>

void free_list(char **list, int count) {
    if (list) {
        for (int i = 0; i < count; i++) {
            free(list[i]);
        }
        free(list);
    }
}

void free_config(Config *config) {
    if (config->config_file) free(config->config_file);
    if (config->mode) free(config->mode);
    if (config->optim_config) free(config->optim_config);

    free_list(config->neuroplast_methods, config->num_neuroplast_methods);
    free_list(config->optimizers, config->num_optimizers);
    free_list(config->activations, config->num_activations);
}