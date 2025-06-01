#include "model_saver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Format PTH personnalisé (binaire structuré)
typedef struct {
    char magic[8];        // "NEURPTH\0"
    uint32_t version;     // Version du format
    uint32_t num_layers;  // Nombre de couches
    uint64_t timestamp;   // Timestamp de sauvegarde
} PTHHeader;

typedef struct {
    uint32_t input_size;
    uint32_t output_size;
    uint32_t activation_type;
    // Suivi par les poids et biais
} PTHLayerHeader;

// Sauvegarder un modèle au format PTH
int model_saver_save_pth(const SavedModel *model, const char *filepath) {
    if (!model || !filepath) return -1;
    
    FILE *file = fopen(filepath, "wb");
    if (!file) return -1;
    
    // Écrire l'en-tête principal
    PTHHeader header = {0};
    strcpy(header.magic, "NEURPTH");
    header.version = 1;
    header.num_layers = model->metadata.num_layers;
    header.timestamp = model->metadata.timestamp;
    
    if (fwrite(&header, sizeof(PTHHeader), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    
    // Écrire les métadonnées
    if (fwrite(&model->metadata, sizeof(ModelMetadata), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    
    // Écrire les tailles des couches
    if (fwrite(model->metadata.layer_sizes, sizeof(size_t), 
               model->metadata.num_layers, file) != model->metadata.num_layers) {
        fclose(file);
        return -1;
    }
    
    // Écrire chaque couche
    for (size_t i = 0; i < model->network->num_layers; i++) {
        Layer *layer = model->network->layers[i];
        
        // En-tête de la couche
        PTHLayerHeader layer_header = {
            .input_size = layer->input_size,
            .output_size = layer->output_size,
            .activation_type = layer->activation_type
        };
        
        if (fwrite(&layer_header, sizeof(PTHLayerHeader), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        
        // Écrire les poids
        for (size_t j = 0; j < layer->output_size; j++) {
            if (fwrite(layer->weights[j], sizeof(float), 
                      layer->input_size, file) != layer->input_size) {
                fclose(file);
                return -1;
            }
        }
        
        // Écrire les biais
        if (fwrite(layer->biases, sizeof(float), 
                  layer->output_size, file) != layer->output_size) {
            fclose(file);
            return -1;
        }
        
        // Écrire les paramètres de neuroplasticité si présents
        if (layer->np_params) {
            if (fwrite(layer->np_params, sizeof(NeuroPlastParams), 1, file) != 1) {
                fclose(file);
                return -1;
            }
        }
    }
    
    fclose(file);
    return 0;
}

// Charger un modèle au format PTH
NeuralNetwork *model_saver_load_pth(const char *filepath, ModelMetadata *metadata) {
    if (!filepath) return NULL;
    
    FILE *file = fopen(filepath, "rb");
    if (!file) return NULL;
    
    // Lire l'en-tête
    PTHHeader header;
    if (fread(&header, sizeof(PTHHeader), 1, file) != 1) {
        fclose(file);
        return NULL;
    }
    
    // Vérifier la signature
    if (strcmp(header.magic, "NEURPTH") != 0) {
        fclose(file);
        return NULL;
    }
    
    // Lire les métadonnées
    if (metadata) {
        if (fread(metadata, sizeof(ModelMetadata), 1, file) != 1) {
            fclose(file);
            return NULL;
        }
        
        // Allouer et lire les tailles des couches
        metadata->layer_sizes = malloc(sizeof(size_t) * header.num_layers);
        if (fread(metadata->layer_sizes, sizeof(size_t), 
                 header.num_layers, file) != header.num_layers) {
            free(metadata->layer_sizes);
            fclose(file);
            return NULL;
        }
    } else {
        // Ignorer les métadonnées
        fseek(file, sizeof(ModelMetadata) + sizeof(size_t) * header.num_layers, SEEK_CUR);
    }
    
    // Créer le réseau
    NeuralNetwork *network = malloc(sizeof(NeuralNetwork));
    if (!network) {
        fclose(file);
        return NULL;
    }
    
    network->num_layers = header.num_layers;
    network->layers = malloc(sizeof(Layer*) * network->num_layers);
    if (!network->layers) {
        free(network);
        fclose(file);
        return NULL;
    }
    
    // Charger chaque couche
    for (size_t i = 0; i < network->num_layers; i++) {
        PTHLayerHeader layer_header;
        if (fread(&layer_header, sizeof(PTHLayerHeader), 1, file) != 1) {
            // Nettoyer en cas d'erreur
            for (size_t j = 0; j < i; j++) {
                layer_free(network->layers[j]);
            }
            free(network->layers);
            free(network);
            fclose(file);
            return NULL;
        }
        
        // Créer la couche
        Layer *layer = layer_create(layer_header.input_size, 
                                   layer_header.output_size, 
                                   layer_header.activation_type);
        if (!layer) {
            for (size_t j = 0; j < i; j++) {
                layer_free(network->layers[j]);
            }
            free(network->layers);
            free(network);
            fclose(file);
            return NULL;
        }
        
        // Charger les poids
        for (size_t j = 0; j < layer->output_size; j++) {
            if (fread(layer->weights[j], sizeof(float), 
                     layer->input_size, file) != layer->input_size) {
                layer_free(layer);
                for (size_t k = 0; k < i; k++) {
                    layer_free(network->layers[k]);
                }
                free(network->layers);
                free(network);
                fclose(file);
                return NULL;
            }
        }
        
        // Charger les biais
        if (fread(layer->biases, sizeof(float), 
                 layer->output_size, file) != layer->output_size) {
            layer_free(layer);
            for (size_t k = 0; k < i; k++) {
                layer_free(network->layers[k]);
            }
            free(network->layers);
            free(network);
            fclose(file);
            return NULL;
        }
        
        // Charger les paramètres de neuroplasticité
        if (layer->np_params) {
            if (fread(layer->np_params, sizeof(NeuroPlastParams), 1, file) != 1) {
                // Non critique, continuer
            }
        }
        
        network->layers[i] = layer;
    }
    
    fclose(file);
    return network;
} 