#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <stddef.h>
#include <stdbool.h>
#include "dataset.h"
#include "../rich_config.h"

// Structure pour stocker les informations d'une image
typedef struct {
    char filepath[512];
    int label;
    char class_name[64];
} ImageInfo;

// Structure pour stocker un ensemble d'images
typedef struct {
    ImageInfo *images;
    size_t count;
    size_t capacity;
    char **class_names;
    size_t num_classes;
} ImageSet;

// Charge les informations des images depuis un répertoire
ImageSet *load_image_set(const char *directory_path);

// Libère la mémoire d'un ImageSet
void free_image_set(ImageSet *set);

// Charge une image et la convertit en données numériques
float *load_image_data(const char *filepath, int width, int height, int channels);

// Charge un dataset d'images à partir de la configuration
Dataset *load_image_dataset_from_config(const RichConfig *config);

// Convertit un ImageSet en Dataset
Dataset *convert_image_set_to_dataset(const ImageSet *set, int width, int height, int channels, size_t num_classes);

// Fusionne plusieurs ImageSets
ImageSet *merge_image_sets(const ImageSet *set1, const ImageSet *set2);

// Affiche les statistiques d'un ImageSet
void print_image_set_stats(const ImageSet *set, const char *set_name);

#endif 