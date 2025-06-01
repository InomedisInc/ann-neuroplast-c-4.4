#include "network.h"
#include "network_simple.h"
#include "activation.h"
#include "../colored_output.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// NOUVELLES FONCTIONS DE CONFIGURATION

// Fonction pour créer une configuration par défaut
NetworkConfig create_default_config() {
    NetworkConfig config;
    config.learning_rate = 0.001f;      // Plus conservateur que la valeur hardcodée
    config.momentum = 0.9f;
    config.dropout_rate = 0.25f;
    config.l2_lambda = 0.0005f;
    config.optimal_threshold = 0.15f;
    config.class_weight_ratio = 15.0f;  // 15x pour classe minoritaire
    config.use_momentum = 1;
    config.use_dropout = 1;
    return config;
}

// Fonction pour créer une configuration optimisée selon l'optimiseur
NetworkConfig create_config_for_optimizer(const char *optimizer_name) {
    NetworkConfig config = create_default_config();
    
    if (strcmp(optimizer_name, "sgd") == 0) {
        config.learning_rate = 0.01f;      // SGD a besoin d'un LR plus élevé
        config.momentum = 0.9f;
        config.dropout_rate = 0.3f;        // Plus de régularisation pour SGD
    } else if (strcmp(optimizer_name, "adam") == 0) {
        config.learning_rate = 0.001f;     // Adam standard
        config.momentum = 0.9f;            // Beta1 équivalent
        config.dropout_rate = 0.2f;        // Moins de dropout avec Adam
    } else if (strcmp(optimizer_name, "adamw") == 0) {
        config.learning_rate = 0.002f;     // AdamW peut être plus agressif
        config.momentum = 0.9f;
        config.l2_lambda = 0.001f;         // Plus de régularisation L2
        config.dropout_rate = 0.15f;       // Moins de dropout avec forte L2
    } else if (strcmp(optimizer_name, "lion") == 0) {
        config.learning_rate = 0.0001f;    // Lion utilise des LR très bas
        config.momentum = 0.99f;           // Momentum élevé pour Lion
        config.dropout_rate = 0.25f;
    } else if (strcmp(optimizer_name, "radam") == 0) {
        config.learning_rate = 0.0015f;    // RAdam intermédiaire
        config.momentum = 0.9f;
        config.dropout_rate = 0.2f;
    }
    
    return config;
}

// Structure simplifiée et robuste avec améliorations anti-overfitting
typedef struct {
    size_t num_layers;
    Layer **layers;
    float learning_rate;
    float momentum;
    float **momentum_weights; // Momentum pour SGD avec momentum
    float *momentum_biases;
    int use_momentum;
    
    // NOUVELLES FONCTIONNALITÉS
    float class_weights[2];     // Poids pour équilibrage des classes [classe_0, classe_1]
    float dropout_rate;         // Taux de dropout
    float l2_lambda;           // Coefficient de régularisation L2
    float optimal_threshold;    // Seuil de décision optimal
    int use_dropout;           // Activer/désactiver dropout
    float *dropout_mask;       // Masque de dropout pour la couche cachée
} SimpleNeuralNetwork;

// Initialisation HE pour ReLU et Xavier pour Sigmoid/Tanh (éprouvée)
static void init_weights_simple(Layer *layer, activation_type_t activation) {
    float fan_in = (float)layer->input_size;
    float fan_out = (float)layer->output_size;
    float std;
    
    // Choix de l'initialisation optimale selon l'activation
    switch (activation) {
        case ACTIVATION_RELU:
        case ACTIVATION_LEAKY_RELU:
        case ACTIVATION_PRELU:
            // He initialization pour ReLU et variantes (2x plus de variance)
            std = sqrtf(2.0f / fan_in);
            break;
            
        case ACTIVATION_GELU:
        case ACTIVATION_MISH:
        case ACTIVATION_SWISH:
            // Initialisation légèrement plus conservative pour activations smooth
            std = sqrtf(1.8f / fan_in);
            break;
            
        case ACTIVATION_ELU:
            // ELU : initialisation intermédiaire
            std = sqrtf(1.5f / fan_in);
            break;
            
        case ACTIVATION_SIGMOID:
        case ACTIVATION_TANH:
            // Xavier/Glorot pour Sigmoid/Tanh
            std = sqrtf(2.0f / (fan_in + fan_out));
            break;
            
        case ACTIVATION_NEUROPLAST:
            // NeuroPlast : initialisation adaptative plus aggressive
            std = sqrtf(2.5f / fan_in);
            break;
            
        default:
            // Par défaut : Xavier modifié
            std = sqrtf(2.0f / (fan_in + fan_out));
    }
    
    // Distribution normale avec Box-Muller
    for (size_t i = 0; i < layer->output_size; i++) {
        for (size_t j = 0; j < layer->input_size; j++) {
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
        // Biais initialisés à zéro pour les couches cachées (sauf cas spéciaux)
        if (activation == ACTIVATION_NEUROPLAST) {
            // NeuroPlast : petit biais aléatoire pour briser la symétrie
            layer->biases[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        } else {
            layer->biases[i] = 0.0f;
        }
    }
}

// Activation simple et robuste
static float apply_activation(float x, activation_type_t type) {
    switch (type) {
        case ACTIVATION_RELU:
            return fmaxf(0.0f, x);
        case ACTIVATION_SIGMOID:
            // Protection contre overflow
            if (x > 500.0f) return 1.0f;
            if (x < -500.0f) return 0.0f;
            return 1.0f / (1.0f + expf(-x));
        case ACTIVATION_TANH:
            // Protection contre overflow
            if (x > 500.0f) return 1.0f;
            if (x < -500.0f) return -1.0f;
            return tanhf(x);
        case ACTIVATION_LEAKY_RELU:
            return (x > 0) ? x : 0.01f * x;
        case ACTIVATION_LINEAR:
            return x;
        // NOUVELLES ACTIVATIONS AJOUTÉES
        case ACTIVATION_GELU:
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        case ACTIVATION_MISH:
            // Mish: x * tanh(ln(1 + e^x))
            return x * tanhf(logf(1.0f + expf(fminf(x, 20.0f)))); // Protection overflow
        case ACTIVATION_SWISH:
            // Swish: x * sigmoid(x)
            if (x > 500.0f) return x;
            if (x < -500.0f) return 0.0f;
            return x / (1.0f + expf(-x));
        case ACTIVATION_ELU:
            // ELU: x si x > 0, sinon α(e^x - 1) avec α = 1.0
            return (x > 0) ? x : (expf(x) - 1.0f);
        case ACTIVATION_PRELU:
            // PReLU avec α = 0.01 (peut être fait paramétrable plus tard)
            return (x > 0) ? x : 0.01f * x;
        default:
            // ACTIVATION_NEUROPLAST ou inconnue : utiliser une fonction adaptative
            // NeuroPlast: mélange adaptatif ReLU/Tanh selon la valeur
            if (x > 1.0f) return x; // ReLU pour grandes valeurs
            else if (x < -1.0f) return tanhf(x); // Tanh pour valeurs négatives
            else return 0.5f * (x + tanhf(x)); // Mélange pour [-1,1]
    }
}

// Dérivée de l'activation
static float activation_derivative(float output, activation_type_t type) {
    switch (type) {
        case ACTIVATION_RELU:
            return (output > 0) ? 1.0f : 0.0f;
        case ACTIVATION_SIGMOID:
            return output * (1.0f - output);
        case ACTIVATION_TANH:
            return 1.0f - output * output;
        case ACTIVATION_LEAKY_RELU:
            return (output > 0) ? 1.0f : 0.01f;
        case ACTIVATION_LINEAR:
            return 1.0f;
        // DÉRIVÉES DES NOUVELLES ACTIVATIONS
        case ACTIVATION_GELU:
            // Dérivée approximative de GELU
            return 0.5f * (1.0f + tanhf(0.7978845608f * (output + 0.044715f * output * output * output)));
        case ACTIVATION_MISH:
            // Dérivée approximative de Mish
            return tanhf(logf(1.0f + expf(fminf(output, 20.0f))));
        case ACTIVATION_SWISH:
            // Dérivée de Swish: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            if (output > 500.0f) return 1.0f;
            if (output < -500.0f) return 0.0f;
            float sig = 1.0f / (1.0f + expf(-output));
            return sig + output * sig * (1.0f - sig);
        case ACTIVATION_ELU:
            // Dérivée de ELU: 1 si x > 0, sinon α * e^x avec α = 1.0
            return (output > 0) ? 1.0f : (output + 1.0f); // output = α(e^x - 1), donc e^x = output + 1
        case ACTIVATION_PRELU:
            return (output > 0) ? 1.0f : 0.01f;
        default:
            // ACTIVATION_NEUROPLAST ou inconnue : dérivée adaptative
            if (output > 1.0f) return 1.0f; // Dérivée ReLU
            else if (output < -1.0f) return 1.0f - output * output; // Dérivée Tanh
            else return 0.5f * (1.0f + (1.0f - output * output)); // Mélange
    }
}

NeuralNetwork *network_create_simple(size_t n_layers, const size_t *layer_sizes, const char **activations) {
    if (n_layers < 2) {
        printf("Erreur: un réseau doit avoir au moins 2 couches\n");
        return NULL;
    }
    
    SimpleNeuralNetwork *net = malloc(sizeof(SimpleNeuralNetwork));
    if (!net) return NULL;
    
    net->num_layers = n_layers - 1;
    
    // Configuration des hyperparamètres pour convergence optimale
    net->learning_rate = 0.001f;  // LR de base standard
    net->momentum = 0.9f;         // Momentum standard
    net->use_momentum = 1;        // Activer momentum par défaut
    
    // ✅ PARAMÈTRES DE RÉGULARISATION OPTIMISÉS pour convergence stable
    net->class_weights[0] = 1.0f;   // Classe 0 (sain)
    net->class_weights[1] = 1.0f;   // Classe 1 (malade) - pas de déséquilibre par défaut
    net->dropout_rate = 0.0f;       // ✅ Pas de dropout par défaut (peut être activé séparément)
    net->l2_lambda = 0.0001f;       // ✅ Régularisation L2 très légère (était 0.0005 - trop fort)
    net->optimal_threshold = 0.5f;  // Seuil standard
    net->use_dropout = 0;           // Dropout désactivé par défaut
    net->dropout_mask = NULL;
    
    // Allocation des couches
    net->layers = malloc(net->num_layers * sizeof(Layer*));
    if (!net->layers) {
        free(net);
        return NULL;
    }
    
    // Allocation des tableaux de momentum
    net->momentum_weights = malloc(net->num_layers * sizeof(float*));
    net->momentum_biases = malloc(net->num_layers * sizeof(float));
    
    // Création des couches avec activations optimisées
    for (size_t i = 0; i < net->num_layers; i++) {
        size_t input_size = layer_sizes[i];
        size_t output_size = layer_sizes[i + 1];
        
        // Sélection d'activation selon la configuration (toutes les activations supportées)
        activation_type_t activation;
        if (i == net->num_layers - 1) {
            // Couche de sortie : sigmoid pour classification binaire
            activation = ACTIVATION_SIGMOID;
        } else if (strcmp(activations[i], "relu") == 0) {
            activation = ACTIVATION_RELU;
        } else if (strcmp(activations[i], "tanh") == 0) {
            activation = ACTIVATION_TANH;
        } else if (strcmp(activations[i], "sigmoid") == 0) {
            activation = ACTIVATION_SIGMOID;
        } else if (strcmp(activations[i], "leaky_relu") == 0) {
            activation = ACTIVATION_LEAKY_RELU;
        } else if (strcmp(activations[i], "gelu") == 0) {
            activation = ACTIVATION_GELU;
        } else if (strcmp(activations[i], "mish") == 0) {
            activation = ACTIVATION_MISH;
        } else if (strcmp(activations[i], "swish") == 0) {
            activation = ACTIVATION_SWISH;
        } else if (strcmp(activations[i], "elu") == 0) {
            activation = ACTIVATION_ELU;
        } else if (strcmp(activations[i], "prelu") == 0) {
            activation = ACTIVATION_PRELU;
        } else if (strcmp(activations[i], "neuroplast") == 0) {
            activation = ACTIVATION_NEUROPLAST; // Sera traité dans default du switch
        } else {
            // Par défaut : ReLU pour activations non reconnues
            activation = ACTIVATION_RELU;
        }
        
        net->layers[i] = layer_create(input_size, output_size, activation);
        if (!net->layers[i]) {
            // Nettoyage en cas d'erreur
            for (size_t j = 0; j < i; j++) {
                layer_free(net->layers[j]);
            }
            free(net->layers);
            free(net);
            return NULL;
        }
        
        // Initialisation des poids selon l'activation
        init_weights_simple(net->layers[i], activation);
        
        // OPTIMISATION SPÉCIALE pour la couche de sortie
        if (i == net->num_layers - 1) {
            // Biais négatif pour favoriser la détection des maladies cardiaques (rappel élevé)
            net->layers[i]->biases[0] = -1.0f; // Logit pour probabilité ~27% (favorise détection)
        }
        
        // Allocation momentum pour cette couche
        net->momentum_weights[i] = calloc(output_size * input_size, sizeof(float));
        
        printf("Couche %zu: %zu → %zu (%s)\n", 
               i, input_size, output_size, activations[i]);
    }
    
    printf("Réseau simple créé avec succès (%zu couches)\n", net->num_layers);
    printf("✅ Class weights: [%.1f, %.1f] (sain, malade)\n", net->class_weights[0], net->class_weights[1]);
    printf("✅ Learning rate: %.4f | Dropout: %.0f%% | L2: %.4f\n", 
           net->learning_rate, net->dropout_rate * 100, net->l2_lambda);
    printf("✅ Seuil optimal: %.2f | Momentum: %.2f\n", 
           net->optimal_threshold, net->momentum);
    
    return (NeuralNetwork*)net;
}

// NOUVELLE FONCTION AVEC CONFIGURATION PERSONNALISÉE
NeuralNetwork *network_create_simple_configured(size_t n_layers, const size_t *layer_sizes, 
                                                const char **activations, NetworkConfig config) {
    if (n_layers < 2) {
        printf("Erreur: un réseau doit avoir au moins 2 couches\n");
        return NULL;
    }
    
    SimpleNeuralNetwork *net = malloc(sizeof(SimpleNeuralNetwork));
    if (!net) return NULL;
    
    net->num_layers = n_layers - 1;
    
    // UTILISATION DE LA CONFIGURATION PERSONNALISÉE
    net->learning_rate = config.learning_rate;
    net->momentum = config.momentum;
    net->use_momentum = config.use_momentum;
    
    // ÉQUILIBRAGE DES CLASSES CONFIGURABLE
    net->class_weights[0] = 1.0f;                      // Classe majoritaire : toujours 1.0
    net->class_weights[1] = config.class_weight_ratio; // Classe minoritaire configurable
    
    // RÉGULARISATION CONFIGURABLE
    net->dropout_rate = config.dropout_rate;
    net->l2_lambda = config.l2_lambda;
    net->optimal_threshold = config.optimal_threshold;
    net->use_dropout = config.use_dropout;
    
    // Allocation des couches
    net->layers = malloc(net->num_layers * sizeof(Layer*));
    if (!net->layers) {
        free(net);
        return NULL;
    }
    
    // Allocation des tableaux de momentum
    net->momentum_weights = malloc(net->num_layers * sizeof(float*));
    net->momentum_biases = malloc(net->num_layers * sizeof(float));
    
    // Allocation du masque de dropout pour la couche cachée
    if (n_layers > 2) {
        net->dropout_mask = malloc(layer_sizes[1] * sizeof(float)); // Couche cachée
    } else {
        net->dropout_mask = NULL;
    }
    
    // Création des couches avec activations configurées
    for (size_t i = 0; i < net->num_layers; i++) {
        size_t input_size = layer_sizes[i];
        size_t output_size = layer_sizes[i + 1];
        
        // Sélection d'activation selon la configuration (toutes les activations supportées)
        activation_type_t activation;
        if (i == net->num_layers - 1) {
            // Couche de sortie : sigmoid pour classification binaire
            activation = ACTIVATION_SIGMOID;
        } else if (strcmp(activations[i], "relu") == 0) {
            activation = ACTIVATION_RELU;
        } else if (strcmp(activations[i], "tanh") == 0) {
            activation = ACTIVATION_TANH;
        } else if (strcmp(activations[i], "sigmoid") == 0) {
            activation = ACTIVATION_SIGMOID;
        } else if (strcmp(activations[i], "leaky_relu") == 0) {
            activation = ACTIVATION_LEAKY_RELU;
        } else if (strcmp(activations[i], "gelu") == 0) {
            activation = ACTIVATION_GELU;
        } else if (strcmp(activations[i], "mish") == 0) {
            activation = ACTIVATION_MISH;
        } else if (strcmp(activations[i], "swish") == 0) {
            activation = ACTIVATION_SWISH;
        } else if (strcmp(activations[i], "elu") == 0) {
            activation = ACTIVATION_ELU;
        } else if (strcmp(activations[i], "prelu") == 0) {
            activation = ACTIVATION_PRELU;
        } else if (strcmp(activations[i], "neuroplast") == 0) {
            activation = ACTIVATION_NEUROPLAST; // Sera traité dans default du switch
        } else {
            // Par défaut : ReLU pour activations non reconnues
            activation = ACTIVATION_RELU;
        }
        
        net->layers[i] = layer_create(input_size, output_size, activation);
        if (!net->layers[i]) {
            // Nettoyage en cas d'erreur
            for (size_t j = 0; j < i; j++) {
                layer_free(net->layers[j]);
            }
            free(net->layers);
            free(net);
            return NULL;
        }
        
        // Initialisation des poids selon l'activation
        init_weights_simple(net->layers[i], activation);
        
        // Allocation momentum pour cette couche
        net->momentum_weights[i] = calloc(output_size * input_size, sizeof(float));
        
        printf("Couche %zu: %zu → %zu (%s)\n", 
               i, input_size, output_size, activations[i]);
    }
    
    printf("Réseau configuré créé avec succès (%zu couches)\n", net->num_layers);
    printf("✅ Class weights: [%.1f, %.1f] | LR: %.4f | Momentum: %.2f\n", 
           net->class_weights[0], net->class_weights[1], net->learning_rate, net->momentum);
    printf("✅ Dropout: %.0f%% | L2: %.4f | Seuil: %.2f\n", 
           net->dropout_rate * 100, net->l2_lambda, net->optimal_threshold);
    
    return (NeuralNetwork*)net;
}

void network_forward_simple(NeuralNetwork *net, float *input) {
    SimpleNeuralNetwork *simple_net = (SimpleNeuralNetwork*)net;
    
    float *current_input = input;
    
    for (size_t i = 0; i < simple_net->num_layers; i++) {
        Layer *layer = simple_net->layers[i];
        
        // Calcul direct : z = W*x + b, puis activation
        for (size_t j = 0; j < layer->output_size; j++) {
            float z = layer->biases[j];
            
            for (size_t k = 0; k < layer->input_size; k++) {
                z += layer->weights[j][k] * current_input[k];
            }
            
            // Application de l'activation
            layer->outputs[j] = apply_activation(z, layer->activation_type);
        }
        
        // Appliquer dropout si activé (sauf pour la couche de sortie)
        if (simple_net->use_dropout && simple_net->dropout_mask && (size_t)i < simple_net->num_layers - 1) {
            for (size_t n = 0; n < layer->output_size; n++) {
                // Générer masque de dropout pour cette couche
                float dropout_prob = (float)rand() / RAND_MAX;
                if (dropout_prob < simple_net->dropout_rate) {
                    layer->outputs[n] = 0.0f;
                } else {
                    layer->outputs[n] /= (1.0f - simple_net->dropout_rate);
                }
            }
        }
        
        current_input = layer->outputs;
    }
}

void network_backward_simple(NeuralNetwork *net, float *input, float *target, float learning_rate) {
    SimpleNeuralNetwork *simple_net = (SimpleNeuralNetwork*)net;
    
    // Calcul de l'erreur pour la couche de sortie avec équilibrage des classes OPTIMISÉ
    Layer *output_layer = simple_net->layers[simple_net->num_layers - 1];
    for (size_t i = 0; i < output_layer->output_size; i++) {
        // Calculer l'erreur pour cette sortie
        float target_val = target[i];
        float output = output_layer->outputs[i];
        float error = target_val - output;
        
        // ÉQUILIBRAGE DES CLASSES ADAPTATIF
        int target_class = (target_val > 0.5f) ? 1 : 0;
        float base_class_weight = simple_net->class_weights[target_class];
        
        // Pondération adaptative : plus de poids pour les erreurs importantes
        float adaptive_weight = base_class_weight;
        
        // Ajustement selon la difficulté de la prédiction
        if (target_val > 0.5f && output < 0.3f) {
            adaptive_weight *= 1.8f; // Cas très difficile : vrai positif mal prédit
        } else if (target_val < 0.5f && output > 0.7f) {
            adaptive_weight *= 1.5f; // Cas difficile : faux positif
        } else if (fabsf(target_val - output) > 0.7f) {
            adaptive_weight *= 1.3f; // Erreur importante
        }
        
        // Delta = erreur pondérée * dérivée de l'activation avec stabilisation
        float derivative = activation_derivative(output, output_layer->activation_type);
        // Stabilisation pour éviter les gradients évanescents dans sigmoid
        if (output_layer->activation_type == ACTIVATION_SIGMOID) {
            derivative = fmaxf(derivative, 0.01f); // Minimum 1% de gradient
        }
        
        output_layer->deltas[i] = error * adaptive_weight * derivative;
    }
    
    // Rétropropagation pour les couches cachées
    for (int l = simple_net->num_layers - 2; l >= 0; l--) {
        Layer *current_layer = simple_net->layers[l];
        Layer *next_layer = simple_net->layers[l + 1];
        
        for (size_t i = 0; i < current_layer->output_size; i++) {
            float error = 0.0f;
            
            // Somme pondérée des erreurs de la couche suivante
            for (size_t j = 0; j < next_layer->output_size; j++) {
                error += next_layer->deltas[j] * next_layer->weights[j][i];
            }
            
            // Prise en compte du dropout dans la rétropropagation
            if (simple_net->use_dropout && simple_net->dropout_mask && (size_t)l < simple_net->num_layers - 1) {
                error *= simple_net->dropout_mask[i];
            }
            
            // Delta = erreur * dérivée
            float derivative = activation_derivative(current_layer->outputs[i], 
                                                   current_layer->activation_type);
            current_layer->deltas[i] = error * derivative;
        }
    }
    
    // Mise à jour des poids avec SGD + momentum + régularisation L2 OPTIMISÉE
    float *layer_input = input;
    float effective_lr = learning_rate > 0 ? learning_rate : simple_net->learning_rate;
    
    // Gradient clipping modéré pour stabilité sans bloquer l'apprentissage
    float max_gradient_norm = 10.0f; // ✅ Augmenté de 3.0 à 10.0 pour permettre l'apprentissage
    
    for (size_t l = 0; l < simple_net->num_layers; l++) {
        Layer *layer = simple_net->layers[l];
        
        // Calcul de la norme des gradients pour cette couche
        float gradient_norm = 0.0f;
        for (size_t i = 0; i < layer->output_size; i++) {
            for (size_t j = 0; j < layer->input_size; j++) {
                float grad = layer->deltas[i] * layer_input[j];
                gradient_norm += grad * grad;
            }
            gradient_norm += layer->deltas[i] * layer->deltas[i]; // Biais aussi
        }
        gradient_norm = sqrtf(gradient_norm);
        
        // Facteur de clipping si nécessaire (plus conservateur)
        float clip_factor = 1.0f;
        if (gradient_norm > max_gradient_norm) {
            clip_factor = max_gradient_norm / gradient_norm;
        }
        
        // Learning rate standard (pas d'adaptation par couche automatique)
        float layer_lr = effective_lr;
        
        for (size_t i = 0; i < layer->output_size; i++) {
            // Mise à jour des poids
            for (size_t j = 0; j < layer->input_size; j++) {
                float gradient = layer->deltas[i] * layer_input[j] * clip_factor;
                
                // RÉGULARISATION L2 modérée : ajout du terme de pénalité
                float l2_penalty = simple_net->l2_lambda * layer->weights[i][j];
                gradient += l2_penalty;
                
                if (simple_net->use_momentum) {
                    // SGD avec momentum optimisé
                    size_t idx = i * layer->input_size + j;
                    simple_net->momentum_weights[l][idx] = 
                        simple_net->momentum * simple_net->momentum_weights[l][idx] + 
                        layer_lr * gradient;
                    layer->weights[i][j] += simple_net->momentum_weights[l][idx];
                } else {
                    // SGD simple avec L2
                    layer->weights[i][j] += layer_lr * gradient;
                }
            }
            
            // Mise à jour des biais avec clipping
            float bias_gradient = layer->deltas[i] * clip_factor;
            layer->biases[i] += layer_lr * bias_gradient;
        }
        
        layer_input = layer->outputs;
    }
}

void network_free_simple(NeuralNetwork *net) {
    SimpleNeuralNetwork *simple_net = (SimpleNeuralNetwork*)net;
    if (!simple_net) return;
    
    if (simple_net->layers) {
        for (size_t i = 0; i < simple_net->num_layers; i++) {
            layer_free(simple_net->layers[i]);
            if (simple_net->momentum_weights && simple_net->momentum_weights[i]) {
                free(simple_net->momentum_weights[i]);
            }
        }
        free(simple_net->layers);
    }
    
    if (simple_net->momentum_weights) free(simple_net->momentum_weights);
    if (simple_net->momentum_biases) free(simple_net->momentum_biases);
    if (simple_net->dropout_mask) free(simple_net->dropout_mask);
    free(simple_net);
}

float *network_output_simple(NeuralNetwork *net) {
    SimpleNeuralNetwork *simple_net = (SimpleNeuralNetwork*)net;
    if (!simple_net || !simple_net->layers || simple_net->num_layers == 0) return NULL;
    return simple_net->layers[simple_net->num_layers - 1]->outputs;
}

// Fonction pour activer/désactiver le dropout (entraînement vs évaluation)
void network_set_dropout_simple(NeuralNetwork *net, int use_dropout) {
    SimpleNeuralNetwork *simple_net = (SimpleNeuralNetwork*)net;
    simple_net->use_dropout = use_dropout;
}

// Fonction pour optimiser le seuil de décision basé sur F1-score
float optimize_threshold_simple(NeuralNetwork *net, float inputs[][21], float targets[], int num_samples) {
    SimpleNeuralNetwork *simple_net = (SimpleNeuralNetwork*)net;
    
    // Désactiver dropout pour évaluation
    int original_dropout = simple_net->use_dropout;
    simple_net->use_dropout = 0;
    
    float best_threshold = 0.5f;
    float best_f1 = 0.0f;
    
    printf("🔍 Debug: Analyse des prédictions avant optimisation...\n");
    
    // Analyser les prédictions actuelles
    float min_score = 1.0f, max_score = 0.0f;
    int actual_positives = 0;
    for (int i = 0; i < num_samples; i++) {
        network_forward_simple(net, inputs[i]);
        float *output = network_output_simple(net);
        
        if (output) {
            float score = output[0];
            if (score < min_score) min_score = score;
            if (score > max_score) max_score = score;
            
            if (targets[i] > 0.5f) actual_positives++;
        }
    }
    
    printf("📊 Scores: min=%.4f, max=%.4f | Positifs réels: %d/%d\n", 
           min_score, max_score, actual_positives, num_samples);
    
    // Tester différents seuils avec recherche adaptative (plus efficace)
    
    // Phase 1: Recherche grossière de 0.01 à 0.99
    for (float threshold = 0.01f; threshold <= 0.99f; threshold += 0.01f) {
        int TP = 0, FP = 0, FN = 0, TN = 0;
        
        // Calculer métriques pour ce seuil
        for (int i = 0; i < num_samples; i++) {
            network_forward_simple(net, inputs[i]);
            float *output = network_output_simple(net);
            
            if (output) {
                int predicted = (output[0] > threshold) ? 1 : 0;
                int actual = (targets[i] > 0.5f) ? 1 : 0;
                
                if (predicted == 1 && actual == 1) TP++;
                else if (predicted == 1 && actual == 0) FP++;
                else if (predicted == 0 && actual == 1) FN++;
                else TN++;
            }
        }
        
        // Calculer métriques complètes
        float precision = (TP + FP > 0) ? (float)TP / (TP + FP) : 0.0f;
        float recall = (TP + FN > 0) ? (float)TP / (TP + FN) : 0.0f;
        float f1 = (precision + recall > 0) ? 2.0f * precision * recall / (precision + recall) : 0.0f;
        
                 // Métrique composite SPÉCIALISÉE MÉDICAL pour maximiser le rappel
         // En médical, mieux vaut trop détecter que pas assez (priorité au rappel)
         float composite_score = 0.0f;
         if (recall >= 0.5f && precision >= 0.05f) { // Seuils minimums adaptés au médical
             composite_score = 0.6f * recall + 0.3f * f1 + 0.1f * precision; // Priorité massive au rappel
         }
        
        // Debug pour les seuils intéressants
        if (threshold <= 0.15f || f1 > 0.1f || composite_score > 0.0f) {
            printf("  Seuil %.2f: TP=%d FP=%d FN=%d TN=%d | Prec=%.3f Recall=%.3f F1=%.3f Comp=%.3f\n",
                   threshold, TP, FP, FN, TN, precision, recall, f1, composite_score);
        }
        
        // Mettre à jour le meilleur selon le score composite
        if (composite_score > best_f1) {
            best_f1 = composite_score;
            best_threshold = threshold;
        }
    }
    
    // Restaurer état dropout
    simple_net->use_dropout = original_dropout;
    simple_net->optimal_threshold = best_threshold;
    
    printf("🎯 Seuil optimal trouvé: %.2f (F1: %.3f)\n", best_threshold, best_f1);
    return best_threshold;
}

// Fonction pour faire des prédictions avec le seuil optimal
int predict_with_optimal_threshold_simple(NeuralNetwork *net, float *input) {
    SimpleNeuralNetwork *simple_net = (SimpleNeuralNetwork*)net;
    
    // Désactiver dropout pour prédiction
    int original_dropout = simple_net->use_dropout;
    simple_net->use_dropout = 0;
    
    network_forward_simple(net, input);
    float *output = network_output_simple(net);
    
    // Restaurer dropout
    simple_net->use_dropout = original_dropout;
    
    if (!output) return -1;
    
    return (output[0] > simple_net->optimal_threshold) ? 1 : 0;
} 