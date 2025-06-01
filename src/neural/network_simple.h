#ifndef NETWORK_SIMPLE_H
#define NETWORK_SIMPLE_H

#include "network.h"

// Structure pour paramètres configurables
typedef struct {
    float learning_rate;
    float momentum;
    float dropout_rate;
    float l2_lambda;
    float optimal_threshold;
    float class_weight_ratio;  // ratio classe minoritaire / majoritaire
    int use_momentum;
    int use_dropout;
} NetworkConfig;

// Fonction pour créer une configuration par défaut
NetworkConfig create_default_config();

// Fonction pour créer une configuration optimisée selon l'optimiseur
NetworkConfig create_config_for_optimizer(const char *optimizer_name);

// Fonctions simplifiées et robustes
NeuralNetwork *network_create_simple(size_t n_layers, const size_t *layer_sizes, const char **activations);

// Nouvelle fonction avec configuration personnalisée
NeuralNetwork *network_create_simple_configured(size_t n_layers, const size_t *layer_sizes, 
                                                const char **activations, NetworkConfig config);

void network_forward_simple(NeuralNetwork *net, float *input);
void network_backward_simple(NeuralNetwork *net, float *input, float *target, float learning_rate);
void network_free_simple(NeuralNetwork *net);
float *network_output_simple(NeuralNetwork *net);

// Nouvelles fonctions pour équilibrage et anti-overfitting
void network_set_dropout_simple(NeuralNetwork *net, int use_dropout);
float optimize_threshold_simple(NeuralNetwork *net, float inputs[][21], float targets[], int num_samples);
int predict_with_optimal_threshold_simple(NeuralNetwork *net, float *input);

#endif 