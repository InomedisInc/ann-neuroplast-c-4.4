#include "model_saver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Format H5 personnalisé (texte structuré JSON-like)
// Plus lisible et compatible avec Python

// Sauvegarder un modèle au format H5
int model_saver_save_h5(const SavedModel *model, const char *filepath) {
    if (!model || !filepath) return -1;
    
    FILE *file = fopen(filepath, "w");
    if (!file) return -1;
    
    // En-tête JSON-like
    fprintf(file, "{\n");
    fprintf(file, "  \"format\": \"NEURH5\",\n");
    fprintf(file, "  \"version\": 1,\n");
    fprintf(file, "  \"timestamp\": %ld,\n", model->metadata.timestamp);
    
    // Métadonnées
    fprintf(file, "  \"metadata\": {\n");
    fprintf(file, "    \"model_name\": \"%s\",\n", model->metadata.model_name);
    fprintf(file, "    \"accuracy\": %.6f,\n", model->metadata.accuracy);
    fprintf(file, "    \"loss\": %.6f,\n", model->metadata.loss);
    fprintf(file, "    \"validation_accuracy\": %.6f,\n", model->metadata.validation_accuracy);
    fprintf(file, "    \"validation_loss\": %.6f,\n", model->metadata.validation_loss);
    fprintf(file, "    \"epoch\": %d,\n", model->metadata.epoch);
    fprintf(file, "    \"optimizer\": \"%s\",\n", model->metadata.optimizer_name);
    fprintf(file, "    \"strategy\": \"%s\",\n", model->metadata.strategy_name);
    fprintf(file, "    \"learning_rate\": %.6f,\n", model->metadata.learning_rate);
    fprintf(file, "    \"batch_size\": %d,\n", model->metadata.batch_size);
    fprintf(file, "    \"num_layers\": %zu\n", model->metadata.num_layers);
    fprintf(file, "  },\n");
    
    // Architecture du réseau
    fprintf(file, "  \"architecture\": {\n");
    fprintf(file, "    \"layer_sizes\": [");
    for (size_t i = 0; i < model->metadata.num_layers; i++) {
        fprintf(file, "%zu", model->metadata.layer_sizes[i]);
        if (i < model->metadata.num_layers - 1) fprintf(file, ", ");
    }
    fprintf(file, "],\n");
    
    fprintf(file, "    \"activation_types\": [");
    for (size_t i = 0; i < model->network->num_layers; i++) {
        fprintf(file, "%d", model->network->layers[i]->activation_type);
        if (i < model->network->num_layers - 1) fprintf(file, ", ");
    }
    fprintf(file, "]\n");
    fprintf(file, "  },\n");
    
    // Paramètres du modèle
    fprintf(file, "  \"parameters\": {\n");
    fprintf(file, "    \"layers\": [\n");
    
    for (size_t i = 0; i < model->network->num_layers; i++) {
        Layer *layer = model->network->layers[i];
        
        fprintf(file, "      {\n");
        fprintf(file, "        \"layer_id\": %zu,\n", i);
        fprintf(file, "        \"input_size\": %zu,\n", layer->input_size);
        fprintf(file, "        \"output_size\": %zu,\n", layer->output_size);
        fprintf(file, "        \"activation_type\": %d,\n", layer->activation_type);
        
        // Poids
        fprintf(file, "        \"weights\": [\n");
        for (size_t j = 0; j < layer->output_size; j++) {
            fprintf(file, "          [");
            for (size_t k = 0; k < layer->input_size; k++) {
                fprintf(file, "%.8f", layer->weights[j][k]);
                if (k < layer->input_size - 1) fprintf(file, ", ");
            }
            fprintf(file, "]");
            if (j < layer->output_size - 1) fprintf(file, ",");
            fprintf(file, "\n");
        }
        fprintf(file, "        ],\n");
        
        // Biais
        fprintf(file, "        \"biases\": [");
        for (size_t j = 0; j < layer->output_size; j++) {
            fprintf(file, "%.8f", layer->biases[j]);
            if (j < layer->output_size - 1) fprintf(file, ", ");
        }
        fprintf(file, "]");
        
        // Paramètres de neuroplasticité
        if (layer->np_params) {
            fprintf(file, ",\n        \"neuroplast_params\": {\n");
            fprintf(file, "          \"alpha\": %.8f,\n", layer->np_params->alpha);
            fprintf(file, "          \"beta\": %.8f,\n", layer->np_params->beta);
            fprintf(file, "          \"gamma\": %.8f,\n", layer->np_params->gamma);
            fprintf(file, "          \"delta\": %.8f\n", layer->np_params->delta);
            fprintf(file, "        }");
        }
        
        fprintf(file, "\n      }");
        if (i < model->network->num_layers - 1) fprintf(file, ",");
        fprintf(file, "\n");
    }
    
    fprintf(file, "    ]\n");
    fprintf(file, "  }\n");
    fprintf(file, "}\n");
    
    fclose(file);
    return 0;
}

// Fonction utilitaire pour parser un nombre depuis une chaîne
static float parse_float(const char *str) {
    return strtof(str, NULL);
}

static int parse_int(const char *str) {
    return atoi(str);
}

// Charger un modèle au format H5 (parsing JSON simple) - VERSION CORRIGÉE
NeuralNetwork *model_saver_load_h5(const char *filepath, ModelMetadata *metadata) {
    if (!filepath) return NULL;
    
    FILE *file = fopen(filepath, "r");
    if (!file) return NULL;
    
    // Initialiser les métadonnées si fournies
    if (metadata) {
        memset(metadata, 0, sizeof(ModelMetadata));
        metadata->layer_sizes = NULL;
        metadata->activation_names = NULL;
    }
    
    char line[1024];
    NeuralNetwork *network = NULL;
    size_t num_layers = 0;
    size_t current_layer = 0;
    int in_weights = 0;
    size_t weight_row = 0;
    
    // Première passe : compter les couches
    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "\"num_layers\":")) {
            char *ptr = strchr(line, ':');
            if (ptr) {
                num_layers = parse_int(ptr + 1);
                break;
            }
        }
    }
    
    if (num_layers == 0) {
        fclose(file);
        return NULL;
    }
    
    // Créer le réseau
    network = malloc(sizeof(NeuralNetwork));
    if (!network) {
        fclose(file);
        return NULL;
    }
    
    network->num_layers = num_layers;
    network->layers = malloc(sizeof(Layer*) * num_layers);
    if (!network->layers) {
        free(network);
        fclose(file);
        return NULL;
    }
    
    // Initialiser les couches à NULL
    for (size_t i = 0; i < num_layers; i++) {
        network->layers[i] = NULL;
    }
    
    // Rembobiner et parser complètement
    rewind(file);
    
    size_t input_size = 0, output_size = 0;
    int activation_type = 0;
    
    while (fgets(line, sizeof(line), file)) {
        // Parser les métadonnées si demandées
        if (metadata) {
            if (strstr(line, "\"model_name\":")) {
                char *start = strchr(line, '"');
                if (start) {
                    start = strchr(start + 1, '"');
                    if (start) {
                        start++;
                        char *end = strchr(start, '"');
                        if (end) {
                            size_t len = end - start;
                            if (len < sizeof(metadata->model_name)) {
                                strncpy(metadata->model_name, start, len);
                                metadata->model_name[len] = '\0';
                            }
                        }
                    }
                }
            }
            else if (strstr(line, "\"accuracy\":")) {
                char *ptr = strchr(line, ':');
                if (ptr) metadata->accuracy = parse_float(ptr + 1);
            }
            else if (strstr(line, "\"loss\":") && !strstr(line, "validation_loss")) {
                char *ptr = strchr(line, ':');
                if (ptr) metadata->loss = parse_float(ptr + 1);
            }
            else if (strstr(line, "\"validation_accuracy\":")) {
                char *ptr = strchr(line, ':');
                if (ptr) metadata->validation_accuracy = parse_float(ptr + 1);
            }
            else if (strstr(line, "\"validation_loss\":")) {
                char *ptr = strchr(line, ':');
                if (ptr) metadata->validation_loss = parse_float(ptr + 1);
            }
            else if (strstr(line, "\"epoch\":")) {
                char *ptr = strchr(line, ':');
                if (ptr) metadata->epoch = parse_int(ptr + 1);
            }
            else if (strstr(line, "\"learning_rate\":")) {
                char *ptr = strchr(line, ':');
                if (ptr) metadata->learning_rate = parse_float(ptr + 1);
            }
            else if (strstr(line, "\"batch_size\":")) {
                char *ptr = strchr(line, ':');
                if (ptr) metadata->batch_size = parse_int(ptr + 1);
            }
        }
        
        // Parser l'architecture
        if (strstr(line, "\"input_size\":")) {
            char *ptr = strchr(line, ':');
            if (ptr) input_size = parse_int(ptr + 1);
        }
        else if (strstr(line, "\"output_size\":")) {
            char *ptr = strchr(line, ':');
            if (ptr) output_size = parse_int(ptr + 1);
        }
        else if (strstr(line, "\"activation_type\":")) {
            char *ptr = strchr(line, ':');
            if (ptr) activation_type = parse_int(ptr + 1);
            
            // Créer la couche maintenant qu'on a toutes les infos
            if (current_layer < num_layers && input_size > 0 && output_size > 0) {
                network->layers[current_layer] = layer_create(input_size, output_size, activation_type);
                if (!network->layers[current_layer]) {
                    // Nettoyer en cas d'erreur
                    for (size_t i = 0; i < current_layer; i++) {
                        if (network->layers[i]) layer_free(network->layers[i]);
                    }
                    free(network->layers);
                    free(network);
                    fclose(file);
                    return NULL;
                }
            }
        }
        else if (strstr(line, "\"weights\":")) {
            in_weights = 1;
            weight_row = 0;
        }
        else if (strstr(line, "\"biases\":")) {
            in_weights = 0;
            
            // Parser les biais sur cette ligne
            char *ptr = strchr(line, '[');
            if (ptr && current_layer < num_layers && network->layers[current_layer]) {
                ptr++;
                for (size_t i = 0; i < output_size && *ptr; i++) {
                    network->layers[current_layer]->biases[i] = parse_float(ptr);
                    ptr = strchr(ptr, ',');
                    if (ptr) ptr++;
                }
                current_layer++;
            }
        }
        else if (in_weights && strstr(line, "[") && current_layer < num_layers && 
                 network->layers[current_layer] && weight_row < output_size) {
            // Parser une ligne de poids
            char *ptr = strchr(line, '[');
            if (ptr) {
                ptr++;
                for (size_t i = 0; i < input_size && *ptr; i++) {
                    network->layers[current_layer]->weights[weight_row][i] = parse_float(ptr);
                    ptr = strchr(ptr, ',');
                    if (ptr) ptr++;
                }
                weight_row++;
            }
        }
    }
    
    // Finaliser les métadonnées
    if (metadata) {
        metadata->num_layers = num_layers;
        metadata->timestamp = time(NULL);
        
        // Allouer et remplir les tailles des couches seulement si nécessaire
        if (num_layers > 0) {
            metadata->layer_sizes = malloc(sizeof(size_t) * num_layers);
            if (metadata->layer_sizes) {
                for (size_t i = 0; i < num_layers; i++) {
                    if (network->layers[i]) {
                        metadata->layer_sizes[i] = network->layers[i]->output_size;
                    } else {
                        metadata->layer_sizes[i] = 0;
                    }
                }
            }
        } else {
            metadata->layer_sizes = NULL;
        }
        
        // Ne pas allouer activation_names pour l'instant (éviter les fuites)
        metadata->activation_names = NULL;
    }
    
    fclose(file);
    return network;
} 