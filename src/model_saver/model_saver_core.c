#include "model_saver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Copier un réseau de neurones
static NeuralNetwork *copy_network(NeuralNetwork *original) {
    if (!original) return NULL;
    
    NeuralNetwork *copy = malloc(sizeof(NeuralNetwork));
    if (!copy) return NULL;
    
    copy->num_layers = original->num_layers;
    copy->layers = malloc(sizeof(Layer*) * copy->num_layers);
    if (!copy->layers) {
        free(copy);
        return NULL;
    }
    
    for (size_t i = 0; i < copy->num_layers; i++) {
        Layer *orig_layer = original->layers[i];
        Layer *new_layer = layer_create(orig_layer->input_size, 
                                       orig_layer->output_size, 
                                       orig_layer->activation_type);
        if (!new_layer) {
            // Nettoyer en cas d'erreur
            for (size_t j = 0; j < i; j++) {
                layer_free(copy->layers[j]);
            }
            free(copy->layers);
            free(copy);
            return NULL;
        }
        
        // Copier les poids
        for (size_t j = 0; j < new_layer->output_size; j++) {
            for (size_t k = 0; k < new_layer->input_size; k++) {
                new_layer->weights[j][k] = orig_layer->weights[j][k];
            }
            new_layer->biases[j] = orig_layer->biases[j];
        }
        
        copy->layers[i] = new_layer;
    }
    
    return copy;
}

// Ajouter un modèle candidat
int model_saver_add_candidate(ModelSaver *saver, 
                             NeuralNetwork *network,
                             Trainer *trainer,
                             float accuracy,
                             float loss,
                             float validation_accuracy,
                             float validation_loss,
                             int epoch) {
    if (!saver || !network || !trainer) return -1;
    
    float score = model_saver_calculate_score(accuracy, loss, validation_accuracy, validation_loss);
    
    // Vérifier si ce modèle mérite d'être dans le top 10
    int insert_position = -1;
    
    if (saver->count < 10) {
        insert_position = saver->count;
    } else {
        // Trouver le modèle avec le plus faible score
        float min_score = saver->models[0].score;
        int min_index = 0;
        for (int i = 1; i < 10; i++) {
            if (saver->models[i].score < min_score) {
                min_score = saver->models[i].score;
                min_index = i;
            }
        }
        
        if (score > min_score) {
            insert_position = min_index;
        }
    }
    
    if (insert_position == -1) {
        return 0; // Modèle pas assez bon
    }
    
    // Libérer l'ancien modèle s'il existe
    if (saver->models[insert_position].network) {
        network_free(saver->models[insert_position].network);
        saver->models[insert_position].network = NULL;
    }
    if (saver->models[insert_position].metadata.layer_sizes) {
        free(saver->models[insert_position].metadata.layer_sizes);
        saver->models[insert_position].metadata.layer_sizes = NULL;
    }
    if (saver->models[insert_position].metadata.activation_names) {
        for (size_t j = 0; j < saver->models[insert_position].metadata.num_layers; j++) {
            if (saver->models[insert_position].metadata.activation_names[j]) {
                free(saver->models[insert_position].metadata.activation_names[j]);
            }
        }
        free(saver->models[insert_position].metadata.activation_names);
        saver->models[insert_position].metadata.activation_names = NULL;
    }
    
    // Copier le réseau
    saver->models[insert_position].network = copy_network(network);
    if (!saver->models[insert_position].network) {
        return -1;
    }
    
    // Remplir les métadonnées
    ModelMetadata *meta = &saver->models[insert_position].metadata;
    meta->accuracy = accuracy;
    meta->loss = loss;
    meta->validation_accuracy = validation_accuracy;
    meta->validation_loss = validation_loss;
    meta->epoch = epoch;
    meta->timestamp = time(NULL);
    meta->learning_rate = trainer->learning_rate;
    meta->batch_size = trainer->batch_size;
    meta->num_layers = network->num_layers;
    
    snprintf(meta->model_name, sizeof(meta->model_name), "model_%d", saver->next_model_id++);
    strncpy(meta->optimizer_name, trainer->optimizer_name, sizeof(meta->optimizer_name) - 1);
    strncpy(meta->strategy_name, trainer->strategy_name, sizeof(meta->strategy_name) - 1);
    
    // Copier les tailles des couches
    meta->layer_sizes = malloc(sizeof(size_t) * meta->num_layers);
    if (meta->layer_sizes) {
        for (size_t i = 0; i < meta->num_layers; i++) {
            meta->layer_sizes[i] = network->layers[i]->output_size;
        }
    }
    
    saver->models[insert_position].score = score;
    
    if (saver->count < 10) {
        saver->count++;
    }
    
    return 1; // Succès
} 