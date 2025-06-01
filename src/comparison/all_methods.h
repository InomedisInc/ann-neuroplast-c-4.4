#ifndef ALL_METHODS_H
#define ALL_METHODS_H

#include <stdbool.h>

// Déclaration principale de la fonction de comparaison
bool compare_all_methods(const char *config_file, 
                         const char *optim_config,
                         char **neuroplast_methods, int num_neuroplast_methods,
                         char **optimizers, int num_optimizers,
                         char **activations, int num_activations,
                         int seed);

#endif /* ALL_METHODS_H */