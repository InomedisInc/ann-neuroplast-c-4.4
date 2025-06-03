#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdbool.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include "dataset.h"
#include "../rich_config.h"

// Structure pour stocker les informations d'une image
typedef struct {
    char filepath[512];     // Chemin complet vers l'image
    int label;              // Étiquette numérique (0, 1, 2, ...)
    char class_name[64];    // Nom de la classe (ex: "NORMAL", "PNEUMONIA")
} ImageInfo;

// Structure pour un ensemble d'images
typedef struct {
    ImageInfo *images;      // Tableau d'informations d'images
    size_t count;           // Nombre d'images
    size_t capacity;        // Capacité allouée
    char **class_names;     // Noms des classes
    size_t num_classes;     // Nombre de classes
} ImageSet;

// Structure pour les informations de couche convolutionnelle
typedef struct {
    size_t input_width;
    size_t input_height;
    size_t input_channels;
    size_t num_filters;
    size_t kernel_size;
    size_t pool_size;
    size_t output_width;
    size_t output_height;
} ConvInfo;

// Fonctions pour le chargement et la gestion des images
bool is_image_file(const char *filename);
ImageSet *create_image_set(size_t initial_capacity);
void free_image_set(ImageSet *set);
bool add_image_to_set(ImageSet *set, const char *filepath, int label, const char *class_name);
ImageSet *load_image_set(const char *directory_path);
void shuffle_image_set(ImageSet *set);
Dataset *convert_image_set_to_dataset(const ImageSet *set, int width, int height, int channels, size_t num_classes);
float *load_image_data(const char *filepath, int width, int height, int channels);
void resize_image_nearest(const unsigned char *input, int input_width, int input_height, 
                         unsigned char *output, int output_width, int output_height, int channels);

// Fonctions pour la configuration
Dataset *load_image_dataset_from_config(const RichConfig *config);
ImageSet *merge_image_sets(const ImageSet *set1, const ImageSet *set2);
void print_image_set_stats(const ImageSet *set, const char *set_name);

#endif
