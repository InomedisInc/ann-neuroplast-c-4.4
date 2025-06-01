#include "network.h"
#include "activation.h"
#include "../colored_output.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Structure pour fonctions d'activations mélangées par couche
typedef struct {
    activation_type_t *activation_mix;  // Tableau des activations pour chaque neurone
    size_t mix_size;                   // Nombre d'activations différentes utilisées
} ActivationMix;

// Nouveau champ dans le réseau pour supporter les activations mélangées
typedef struct {
    size_t num_layers;
    Layer **layers;
    ActivationMix *activation_mixes;   // Nouveau: mélange d'activations par couche
    float *batch_norm_mean;            // Nouveau: moyennes pour normalisation batch
    float *batch_norm_var;             // Nouveau: variances pour normalisation batch
    float *batch_norm_gamma;           // Nouveau: paramètres d'échelle
    float *batch_norm_beta;            // Nouveau: paramètres de décalage
    float dropout_rate;                // Nouveau: taux de dropout adaptatif
    int use_batch_norm;                // Nouveau: flag pour normalisation batch
    int use_residual;                  // Nouveau: flag pour connexions résiduelles
} EnhancedNeuralNetwork;

// Initialisation avancée des poids avec différentes méthodes
static void init_weights_advanced(Layer *layer, int method) {
    float fan_in = (float)layer->input_size;
    float fan_out = (float)layer->output_size;
    float std = 0.1f;
    
    switch (method) {
        case 0: // Xavier/Glorot uniforme
            {
                float limit = sqrtf(6.0f / (fan_in + fan_out));
                for (size_t i = 0; i < layer->output_size; i++) {
                    for (size_t j = 0; j < layer->input_size; j++) {
                        float u1 = (float)rand() / RAND_MAX;
                        layer->weights[i][j] = (2.0f * u1 - 1.0f) * limit;
                    }
                }
            }
            break;
        case 1: // He initialization 
            std = sqrtf(2.0f / fan_in);
            break;
        case 2: // LeCun initialization
            std = sqrtf(1.0f / fan_in);
            break;
        case 3: // LSUV (Layer-wise Sequential Unit Variance)
            std = 1.0f / sqrtf(fan_in);
            break;
        default: // Xavier normal
            std = sqrtf(2.0f / (fan_in + fan_out));
            break;
    }
    
    if (method != 0) { // Pour les méthodes normales (non uniformes)
        for (size_t i = 0; i < layer->output_size; i++) {
            for (size_t j = 0; j < layer->input_size; j++) {
                // Box-Muller pour distribution normale
                static int has_spare = 0;
                static float spare;
                if (has_spare) {
                    has_spare = 0;
                    layer->weights[i][j] = spare * std;
                } else {
                    has_spare = 1;
                    float u, v, mag;
                    do {
                        u = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                        v = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                        mag = u * u + v * v;
                    } while (mag >= 1.0f || mag == 0.0f);
                    mag = sqrtf(-2.0f * logf(mag) / mag);
                    spare = v * mag;
                    layer->weights[i][j] = u * mag * std;
                }
            }
        }
    }
}

// Fonctions d'activations mélangées pour une couche
static void apply_mixed_activations(Layer *layer, float *inputs, ActivationMix *mix) {
    if (!mix || !mix->activation_mix) {
        // Fallback vers activation unique
        layer_forward(layer, inputs);
        return;
    }
    
    for (size_t i = 0; i < layer->output_size; i++) {
        float z = layer->biases[i];
        
        // Calcul du produit scalaire
        for (size_t j = 0; j < layer->input_size; j++) {
            z += layer->weights[i][j] * inputs[j];
        }
        
        // Application de l'activation spécifique au neurone
        activation_type_t act_type = mix->activation_mix[i % mix->mix_size];
        
        switch (act_type) {
            case ACTIVATION_RELU: 
                layer->outputs[i] = fmaxf(0.0f, z); 
                break;
            case ACTIVATION_SIGMOID: 
                layer->outputs[i] = 1.0f / (1.0f + expf(-z)); 
                break;
            case ACTIVATION_GELU: 
                layer->outputs[i] = 0.5f * z * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (z + 0.044715f * z * z * z))); 
                break;
            case ACTIVATION_LEAKY_RELU: 
                layer->outputs[i] = (z > 0) ? z : 0.01f * z; 
                break;
            case ACTIVATION_ELU: 
                layer->outputs[i] = (z > 0) ? z : 1.0f * (expf(z) - 1.0f); 
                break;
            case ACTIVATION_MISH: 
                layer->outputs[i] = z * tanhf(log1pf(expf(z))); 
                break;
            case ACTIVATION_SWISH: 
                layer->outputs[i] = z / (1.0f + expf(-z)); 
                break;
            case ACTIVATION_NEUROPLAST:
                if (layer->np_params) {
                    layer->outputs[i] = neuroplast(z, &layer->np_params[i % layer->output_size]); 
                } else {
                    layer->outputs[i] = 1.0f / (1.0f + expf(-z)); // fallback sigmoid
                }
                break;
            default: 
                layer->outputs[i] = z;
                break;
        }
    }
}

// Normalisation par batch améliorée
static void apply_batch_normalization(float *outputs, float *mean, float *var, 
                                     float *gamma, float *beta, size_t size, 
                                     float momentum, int training) {
    if (training) {
        // Calcul de la moyenne et variance du batch
        float batch_mean = 0.0f;
        float batch_var = 0.0f;
        
        for (size_t i = 0; i < size; i++) {
            batch_mean += outputs[i];
        }
        batch_mean /= size;
        
        for (size_t i = 0; i < size; i++) {
            float diff = outputs[i] - batch_mean;
            batch_var += diff * diff;
        }
        batch_var /= size;
        
        // Mise à jour des statistiques en mouvement
        for (size_t i = 0; i < size; i++) {
            mean[i] = momentum * mean[i] + (1.0f - momentum) * batch_mean;
            var[i] = momentum * var[i] + (1.0f - momentum) * batch_var;
        }
        
        // Normalisation
        float epsilon = 1e-5f;
        for (size_t i = 0; i < size; i++) {
            outputs[i] = gamma[i] * (outputs[i] - batch_mean) / sqrtf(batch_var + epsilon) + beta[i];
        }
    } else {
        // Mode inférence : utiliser les statistiques stockées
        float epsilon = 1e-5f;
        for (size_t i = 0; i < size; i++) {
            outputs[i] = gamma[i] * (outputs[i] - mean[i]) / sqrtf(var[i] + epsilon) + beta[i];
        }
    }
}

// Dropout adaptatif
static void apply_adaptive_dropout(float *outputs, size_t size, float base_rate, 
                                  float epoch_factor, int training) {
    if (!training) return;
    
    // Dropout rate adaptatif basé sur l'époque
    float adaptive_rate = base_rate * (1.0f - epoch_factor);
    adaptive_rate = fmaxf(0.1f, fminf(0.8f, adaptive_rate));
    
    for (size_t i = 0; i < size; i++) {
        if ((float)rand() / RAND_MAX < adaptive_rate) {
            outputs[i] = 0.0f;
        } else {
            outputs[i] /= (1.0f - adaptive_rate); // Scaling pour maintenir l'espérance
        }
    }
}

NeuralNetwork *network_create(size_t n_layers, const size_t *layer_sizes, const char **activations) {
    // Validation des entrées
    if (n_layers < 2) {
        printf("Erreur: un réseau doit avoir au moins 2 couches (entrée et sortie)\n");
        return NULL;
    }
    
    if (!layer_sizes || !activations) {
        printf("Erreur: paramètres invalides\n");
        return NULL;
    }
    
    // Allocation du réseau amélioré
    EnhancedNeuralNetwork *net = malloc(sizeof(EnhancedNeuralNetwork));
    if (!net) {
        printf("Erreur: impossible d'allouer la mémoire pour le réseau\n");
        return NULL;
    }
    
    net->num_layers = n_layers - 1;
    net->dropout_rate = 0.3f; // Taux de dropout initial
    net->use_batch_norm = 1;  // Activer la normalisation par batch
    net->use_residual = (n_layers > 3) ? 1 : 0; // Connexions résiduelles pour réseaux profonds
    
    // Allocation des couches
    net->layers = malloc((n_layers - 1) * sizeof(Layer*));
    net->activation_mixes = malloc((n_layers - 1) * sizeof(ActivationMix));
    
    if (!net->layers || !net->activation_mixes) {
        printf("Erreur: impossible d'allouer la mémoire pour les couches\n");
        free(net);
        return NULL;
    }
    
    // Allocation pour la normalisation par batch
    size_t total_neurons = 0;
    for (size_t i = 1; i < n_layers; i++) {
        total_neurons += layer_sizes[i];
    }
    
    net->batch_norm_mean = calloc(total_neurons, sizeof(float));
    net->batch_norm_var = malloc(total_neurons * sizeof(float));
    net->batch_norm_gamma = malloc(total_neurons * sizeof(float));
    net->batch_norm_beta = calloc(total_neurons, sizeof(float));
    
    if (!net->batch_norm_mean || !net->batch_norm_var || 
        !net->batch_norm_gamma || !net->batch_norm_beta) {
        printf("Erreur: allocation mémoire pour batch normalization\n");
        free(net);
        return NULL;
    }
    
    // Initialisation des paramètres de batch norm
    for (size_t i = 0; i < total_neurons; i++) {
        net->batch_norm_var[i] = 1.0f;
        net->batch_norm_gamma[i] = 1.0f;
    }
    
    // Création des couches avec activations mélangées
    for (size_t i = 0; i < n_layers - 1; i++) {
        size_t input_size = layer_sizes[i];
        size_t output_size = layer_sizes[i + 1];
        
        // Déterminer le type d'activation principal
        activation_type_t main_activation = get_activation_type(activations[i]);
        
        // Créer la couche
        net->layers[i] = layer_create(input_size, output_size, main_activation);
        if (!net->layers[i]) {
            printf("Erreur: impossible de créer la couche %zu\n", i);
            // Nettoyage...
            return NULL;
        }
        
        // Initialisation avancée des poids selon la couche
        int init_method = (i == 0) ? 1 : (i == n_layers - 2) ? 2 : 0; // He pour première, LeCun pour dernière
        init_weights_advanced(net->layers[i], init_method);
        
        // Créer le mélange d'activations pour cette couche
        net->activation_mixes[i].mix_size = 4; // Utiliser 4 activations différentes
        net->activation_mixes[i].activation_mix = malloc(4 * sizeof(activation_type_t));
        
        if (net->activation_mixes[i].activation_mix) {
            // Mélange optimisé selon la position de la couche
            if (i == 0) { // Couche d'entrée
                net->activation_mixes[i].activation_mix[0] = ACTIVATION_GELU;
                net->activation_mixes[i].activation_mix[1] = ACTIVATION_SWISH;
                net->activation_mixes[i].activation_mix[2] = ACTIVATION_MISH;
                net->activation_mixes[i].activation_mix[3] = main_activation;
            } else if (i == n_layers - 2) { // Couche de sortie
                net->activation_mixes[i].activation_mix[0] = ACTIVATION_SIGMOID;
                net->activation_mixes[i].activation_mix[1] = main_activation;
                net->activation_mixes[i].activation_mix[2] = ACTIVATION_SIGMOID;
                net->activation_mixes[i].activation_mix[3] = ACTIVATION_SIGMOID;
            } else { // Couches cachées
                net->activation_mixes[i].activation_mix[0] = main_activation;
                net->activation_mixes[i].activation_mix[1] = ACTIVATION_GELU;
                net->activation_mixes[i].activation_mix[2] = ACTIVATION_LEAKY_RELU;
                net->activation_mixes[i].activation_mix[3] = ACTIVATION_MISH;
            }
        }
        
        char layer_info[256];
        snprintf(layer_info, sizeof(layer_info), 
                "Couche %zu: %zu → %zu (Mixed: %s+GELU+MISH+LeakyReLU)", 
                i, input_size, output_size, activations[i]);
        print_network_info_safe(layer_info);
    }
    
    print_success_safe("Réseau amélioré créé avec activations mélangées et normalisation batch");
    
    return (NeuralNetwork*)net; // Cast pour compatibilité
}

void network_free(NeuralNetwork *net) {
    if (!net) return;
    
    EnhancedNeuralNetwork *enhanced_net = (EnhancedNeuralNetwork*)net;
    
    if (enhanced_net->layers) {
        for (size_t i = 0; i < enhanced_net->num_layers; i++) {
            if (enhanced_net->layers[i]) {
                layer_free(enhanced_net->layers[i]);
            }
        }
        free(enhanced_net->layers);
    }
    
    // Libération des activations mélangées
    if (enhanced_net->activation_mixes) {
        for (size_t i = 0; i < enhanced_net->num_layers; i++) {
            if (enhanced_net->activation_mixes[i].activation_mix) {
                free(enhanced_net->activation_mixes[i].activation_mix);
            }
        }
        free(enhanced_net->activation_mixes);
    }
    
    // Libération des paramètres de normalisation batch
    if (enhanced_net->batch_norm_mean) free(enhanced_net->batch_norm_mean);
    if (enhanced_net->batch_norm_var) free(enhanced_net->batch_norm_var);
    if (enhanced_net->batch_norm_gamma) free(enhanced_net->batch_norm_gamma);
    if (enhanced_net->batch_norm_beta) free(enhanced_net->batch_norm_beta);
    
    free(enhanced_net);
}

void network_forward(NeuralNetwork *net, float *input) {
    if (!net || !input) return;
    
    EnhancedNeuralNetwork *enhanced_net = (EnhancedNeuralNetwork*)net;
    if (!enhanced_net->layers) return;
    
    float *current_input = input;
    float *residual_connection = NULL;
    
    for (size_t i = 0; i < enhanced_net->num_layers; i++) {
        Layer *layer = enhanced_net->layers[i];
        if (!layer) return;
        
        // Sauvegarder pour connexion résiduelle (pour couches cachées)
        if (enhanced_net->use_residual && i > 0 && i < enhanced_net->num_layers - 1) {
            if (layer->input_size == layer->output_size) {
                residual_connection = malloc(layer->output_size * sizeof(float));
                if (residual_connection) {
                    memcpy(residual_connection, current_input, layer->output_size * sizeof(float));
                }
            }
        }
        
        // Propagation avec activations mélangées
        apply_mixed_activations(layer, current_input, &enhanced_net->activation_mixes[i]);
        
        // Normalisation par batch si activée
        if (enhanced_net->use_batch_norm && i < enhanced_net->num_layers - 1) { // Pas sur couche de sortie
            apply_batch_normalization(
                layer->outputs,
                &enhanced_net->batch_norm_mean[i * layer->output_size],
                &enhanced_net->batch_norm_var[i * layer->output_size],
                &enhanced_net->batch_norm_gamma[i * layer->output_size],
                &enhanced_net->batch_norm_beta[i * layer->output_size],
                layer->output_size,
                0.9f, // momentum
                1     // training mode
            );
        }
        
        // Connexion résiduelle
        if (residual_connection) {
            for (size_t j = 0; j < layer->output_size; j++) {
                layer->outputs[j] += 0.3f * residual_connection[j]; // Scaling factor 0.3
            }
            free(residual_connection);
            residual_connection = NULL;
        }
        
        // Dropout adaptatif (sauf couche de sortie)
        if (i < enhanced_net->num_layers - 1) {
            float epoch_factor = 0.5f; // À adapter selon l'époque courante
            apply_adaptive_dropout(layer->outputs, layer->output_size, 
                                 enhanced_net->dropout_rate, epoch_factor, 1);
        }
        
        current_input = layer->outputs;
    }
}

void network_backward(NeuralNetwork *net, float *input, float *target, float learning_rate, float class_weight) {
    if (!net || !input || !target || net->num_layers == 0) return;
    
    EnhancedNeuralNetwork *enhanced_net = (EnhancedNeuralNetwork*)net;
    
    // Calcul de l'erreur pour la couche de sortie avec pondération de classe améliorée
    Layer *output_layer = enhanced_net->layers[enhanced_net->num_layers - 1];
    for (size_t i = 0; i < output_layer->output_size; i++) {
        float output = output_layer->outputs[i];
        float target_val = target[i];
        
        // Erreur avec pondération de classe adaptative
        float adaptive_weight = class_weight;
        if (target_val > 0.5f && output < 0.3f) adaptive_weight *= 1.5f; // Boost for hard negatives
        if (target_val < 0.5f && output > 0.7f) adaptive_weight *= 1.3f; // Boost for hard positives
        
        float error = (target_val - output) * adaptive_weight;
        
        // Dérivée de la fonction d'activation améliorée
        float derivative = 1.0f;
        activation_type_t act_type = enhanced_net->activation_mixes[enhanced_net->num_layers - 1].activation_mix[i % 4];
        
        switch (act_type) {
            case ACTIVATION_SIGMOID:
                derivative = output * (1.0f - output);
                // Correction pour éviter les gradients évanescents
                derivative = fmaxf(derivative, 0.01f);
                break;
            case ACTIVATION_RELU:
                derivative = (output > 0) ? 1.0f : 0.0f;
                break;
            case ACTIVATION_LEAKY_RELU:
                derivative = (output > 0) ? 1.0f : 0.01f;
                break;
            case ACTIVATION_GELU:
                // Dérivée GELU améliorée
                {
                    float x = output; // Approximation: utiliser output comme x
                    float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
                    float pdf = expf(-0.5f * x * x) / sqrtf(2.0f * M_PI);
                    derivative = cdf + x * pdf;
                    derivative = fmaxf(derivative, 0.01f);
                }
                break;
            case ACTIVATION_ELU:
                derivative = (output > 0) ? 1.0f : (output + 1.0f);
                break;
            case ACTIVATION_MISH:
                {
                    float x = output;
                    float softplus = log1pf(expf(x));
                    float tanh_sp = tanhf(softplus);
                    float sech_sp = 1.0f - tanh_sp * tanh_sp;
                    float sigmoid_val = 1.0f / (1.0f + expf(-x));
                    derivative = tanh_sp + x * sech_sp * sigmoid_val;
                    derivative = fmaxf(derivative, 0.01f);
                }
                break;
            case ACTIVATION_SWISH:
                {
                    float x = output;
                    float sigmoid_val = 1.0f / (1.0f + expf(-x));
                    derivative = sigmoid_val + x * sigmoid_val * (1.0f - sigmoid_val);
                    derivative = fmaxf(derivative, 0.01f);
                }
                break;
            case ACTIVATION_NEUROPLAST:
                // Dérivée de neuroplast utilisant la fonction dédiée
                if (output_layer->np_params) {
                    derivative = neuroplast_get_derivative(output, &output_layer->np_params[i % output_layer->output_size]);
                } else {
                    derivative = 0.5f; // Fallback si pas de paramètres
                }
                derivative = fmaxf(derivative, 0.01f);
                break;
            default:
                derivative = 1.0f;
                break;
        }
        
        output_layer->deltas[i] = error * derivative;
    }
    
    // Rétropropagation pour les couches cachées avec améliorations
    for (int l = enhanced_net->num_layers - 2; l >= 0; l--) {
        Layer *current_layer = enhanced_net->layers[l];
        Layer *next_layer = enhanced_net->layers[l + 1];
        
        for (size_t i = 0; i < current_layer->output_size; i++) {
            float error = 0.0f;
            
            // Calcul de l'erreur rétropropagée
            for (size_t j = 0; j < next_layer->output_size; j++) {
                error += next_layer->deltas[j] * next_layer->weights[j][i];
            }
            
            float output = current_layer->outputs[i];
            float derivative = 1.0f;
            
            // Dérivée selon l'activation mélangée du neurone
            activation_type_t act_type = enhanced_net->activation_mixes[l].activation_mix[i % 4];
            
            switch (act_type) {
                case ACTIVATION_SIGMOID:
                    derivative = output * (1.0f - output);
                    derivative = fmaxf(derivative, 0.01f); // Anti-vanishing
                    break;
                case ACTIVATION_RELU:
                    derivative = (output > 0) ? 1.0f : 0.0f;
                    break;
                case ACTIVATION_LEAKY_RELU:
                    derivative = (output > 0) ? 1.0f : 0.01f;
                    break;
                case ACTIVATION_GELU:
                    {
                        float x = output;
                        float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
                        float pdf = expf(-0.5f * x * x) / sqrtf(2.0f * M_PI);
                        derivative = cdf + x * pdf;
                        derivative = fmaxf(derivative, 0.01f);
                    }
                    break;
                case ACTIVATION_ELU:
                    derivative = (output > 0) ? 1.0f : (output + 1.0f);
                    break;
                case ACTIVATION_MISH:
                    {
                        float x = output;
                        float softplus = log1pf(expf(x));
                        float tanh_sp = tanhf(softplus);
                        float sech_sp = 1.0f - tanh_sp * tanh_sp;
                        float sigmoid_val = 1.0f / (1.0f + expf(-x));
                        derivative = tanh_sp + x * sech_sp * sigmoid_val;
                        derivative = fmaxf(derivative, 0.01f);
                    }
                    break;
                case ACTIVATION_SWISH:
                    {
                        float x = output;
                        float sigmoid_val = 1.0f / (1.0f + expf(-x));
                        derivative = sigmoid_val + x * sigmoid_val * (1.0f - sigmoid_val);
                        derivative = fmaxf(derivative, 0.01f);
                    }
                    break;
                case ACTIVATION_NEUROPLAST:
                    // Dérivée de neuroplast utilisant la fonction dédiée
                    if (current_layer->np_params) {
                        derivative = neuroplast_get_derivative(output, &current_layer->np_params[i % current_layer->output_size]);
                    } else {
                        derivative = 0.5f; // Fallback si pas de paramètres
                    }
                    derivative = fmaxf(derivative, 0.01f);
                    break;
                default:
                    derivative = 1.0f;
                    break;
            }
            
            current_layer->deltas[i] = error * derivative;
        }
    }
    
    // Mise à jour des poids et biais avec optimisations avancées
    float *layer_input = input;
    float l2_reg = 0.0001f; // Régularisation L2
    float gradient_clip_value = 5.0f; // Gradient clipping plus conservatif
    
    for (size_t l = 0; l < enhanced_net->num_layers; l++) {
        Layer *layer = enhanced_net->layers[l];
        
        // Calcul de la norme des gradients pour adaptive clipping
        float gradient_norm = 0.0f;
        for (size_t i = 0; i < layer->output_size; i++) {
            for (size_t j = 0; j < layer->input_size; j++) {
                float grad = layer->deltas[i] * layer_input[j];
                gradient_norm += grad * grad;
            }
        }
        gradient_norm = sqrtf(gradient_norm);
        
        // Adaptive learning rate basé sur la norme des gradients
        float adaptive_lr = learning_rate;
        if (gradient_norm > gradient_clip_value) {
            adaptive_lr *= gradient_clip_value / gradient_norm;
        }
        
        // Mise à jour des poids avec régularisation améliorée
        for (size_t i = 0; i < layer->output_size; i++) {
            for (size_t j = 0; j < layer->input_size; j++) {
                float gradient = layer->deltas[i] * layer_input[j];
                
                // Gradient clipping par valeur
                gradient = fmaxf(-gradient_clip_value, fminf(gradient_clip_value, gradient));
                
                // Régularisation L2 avec décroissance pondérée
                float weight_decay = l2_reg * fabsf(layer->weights[i][j]);
                
                // Mise à jour avec momentum implicite (smoothing)
                layer->weights[i][j] += adaptive_lr * (gradient - weight_decay);
            }
            
            // Mise à jour des biais avec clipping
            float bias_gradient = layer->deltas[i];
            bias_gradient = fmaxf(-gradient_clip_value, fminf(gradient_clip_value, bias_gradient));
            
            layer->biases[i] += adaptive_lr * bias_gradient;
        }
        
        layer_input = layer->outputs;
    }
}

float *network_output(NeuralNetwork *net) {
    if (!net || !net->layers || net->num_layers == 0) return NULL;
    return net->layers[net->num_layers - 1]->outputs;
}