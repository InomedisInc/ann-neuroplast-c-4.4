#include "layer.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "activation.h"

// Générateur de nombres aléatoires pour distribution normale (Box-Muller)
static float box_muller_normal(float mean, float std) {
    static int has_spare = 0;
    static float spare;
    
    if (has_spare) {
        has_spare = 0;
        return spare * std + mean;
    }
    
    has_spare = 1;
    float u, v, mag;
    do {
        u = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        v = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        mag = u * u + v * v;
    } while (mag >= 1.0f || mag == 0.0f);
    
    mag = sqrtf(-2.0f * logf(mag) / mag);
    spare = v * mag;
    return u * mag * std + mean;
}

Layer *layer_create(size_t input_size, size_t output_size, int activation_type) {
    Layer *layer = malloc(sizeof(Layer));
    if (!layer) return NULL;
    
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation_type = activation_type;

    // Allocation des poids
    layer->weights = malloc(output_size * sizeof(float *));
    if (!layer->weights) {
        free(layer);
        return NULL;
    }
    
    for (size_t i = 0; i < output_size; i++) {
        layer->weights[i] = malloc(input_size * sizeof(float));
        if (!layer->weights[i]) {
            for (size_t j = 0; j < i; j++) {
                free(layer->weights[j]);
            }
            free(layer->weights);
            free(layer);
            return NULL;
        }
        
        // Initialisation améliorée des poids selon l'activation
        float std;
        switch (activation_type) {
            case ACTIVATION_RELU:
            case ACTIVATION_LEAKY_RELU:
            case ACTIVATION_ELU:
            case ACTIVATION_PRELU:
                // He initialization pour ReLU et variantes
                std = sqrtf(2.0f / input_size);
                break;
            case ACTIVATION_SIGMOID:
            case ACTIVATION_GELU:
            case ACTIVATION_MISH:
            case ACTIVATION_SWISH:
                // Xavier/Glorot pour sigmoid et variantes
                std = sqrtf(1.0f / input_size);
                break;
            case ACTIVATION_NEUROPLAST:
                // Initialisation spéciale pour NeuroPlast
                std = sqrtf(2.0f / (input_size + output_size));
                break;
            default:
                // Xavier/Glorot par défaut
                std = sqrtf(2.0f / (input_size + output_size));
                break;
        }
        
        for (size_t j = 0; j < input_size; j++) {
            layer->weights[i][j] = box_muller_normal(0.0f, std);
        }
    }

    // Allocation et initialisation des autres composants
    layer->biases = calloc(output_size, sizeof(float));
    layer->outputs = calloc(output_size, sizeof(float));
    layer->deltas = calloc(output_size, sizeof(float));
    
    if (!layer->biases || !layer->outputs || !layer->deltas) {
        layer_free(layer);
        return NULL;
    }
    
    // Initialisation des paramètres NeuroPlast si nécessaire
    if (activation_type == ACTIVATION_NEUROPLAST) {
        layer->np_params = malloc(output_size * sizeof(NeuroPlastParams));
        if (!layer->np_params) {
            layer_free(layer);
            return NULL;
        }
        for (size_t i = 0; i < output_size; i++) {
            neuroplast_init_params(&layer->np_params[i], 2.5f, 0.2f, 0.3f, 2.5f);
        }
    } else {
        layer->np_params = NULL;
    }
    
    return layer;
}

void layer_free(Layer *layer) {
    if (!layer) return;
    
    if (layer->weights) {
        for (size_t i = 0; i < layer->output_size; i++) {
            if (layer->weights[i]) {
                free(layer->weights[i]);
            }
        }
        free(layer->weights);
    }
    
    if (layer->biases) free(layer->biases);
    if (layer->outputs) free(layer->outputs);
    if (layer->deltas) free(layer->deltas);
    if (layer->np_params) free(layer->np_params);
    free(layer);
}

void layer_forward(Layer *layer, float *input) {
    if (!layer || !input) return;
    
    for (size_t i = 0; i < layer->output_size; i++) {
        float z = layer->biases[i];
        
        // Calcul du produit scalaire avec vérification de sécurité
        for (size_t j = 0; j < layer->input_size; j++) {
            z += layer->weights[i][j] * input[j];
        }
        
        // Application de la fonction d'activation
        switch (layer->activation_type) {
            case ACTIVATION_RELU: 
                layer->outputs[i] = relu(z); 
                break;
            case ACTIVATION_SIGMOID: 
                layer->outputs[i] = sigmoid(z); 
                break;
            case ACTIVATION_GELU: 
                layer->outputs[i] = gelu(z); 
                break;
            case ACTIVATION_NEUROPLAST: 
                // Vérification de sécurité pour les paramètres neuroplast
                if (layer->np_params) {
                    layer->outputs[i] = neuroplast(z, &layer->np_params[i]); 
                } else {
                    // Fallback vers sigmoid si paramètres non initialisés
                    layer->outputs[i] = sigmoid(z);
                }
                break;
            case ACTIVATION_LEAKY_RELU: 
                layer->outputs[i] = leaky_relu(z, 0.01f); 
                break;
            case ACTIVATION_ELU: 
                layer->outputs[i] = elu(z, 1.0f); 
                break;
            case ACTIVATION_MISH: 
                layer->outputs[i] = mish(z); 
                break;
            case ACTIVATION_SWISH: 
                layer->outputs[i] = swish(z); 
                break;
            case ACTIVATION_PRELU: 
                layer->outputs[i] = prelu(z, 0.01f); 
                break;
            default: 
                layer->outputs[i] = z;
                break;
        }
    }
}

void layer_backward(Layer *layer, float *input, float *delta, float learning_rate) {
    if (!layer || !input || !delta) return;
    
    // Calcul des gradients pour cette couche
    for (size_t i = 0; i < layer->output_size; i++) {
        // Le delta est déjà calculé par la couche suivante ou la fonction de coût
        layer->deltas[i] = delta[i];
        
        // Mise à jour des poids
        for (size_t j = 0; j < layer->input_size; j++) {
            float gradient = layer->deltas[i] * input[j];
            
            // Gradient clipping
            if (gradient > 1.0f) gradient = 1.0f;
            if (gradient < -1.0f) gradient = -1.0f;
            
            // Mise à jour avec régularisation L2
            float l2_reg = 0.0001f;
            float reg_term = l2_reg * layer->weights[i][j];
            layer->weights[i][j] += learning_rate * (gradient - reg_term);
        }
        
        // Mise à jour des biais
        float bias_gradient = layer->deltas[i];
        if (bias_gradient > 1.0f) bias_gradient = 1.0f;
        if (bias_gradient < -1.0f) bias_gradient = -1.0f;
        
        layer->biases[i] += learning_rate * bias_gradient;
    }
}