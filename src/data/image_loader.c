#include "image_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <libgen.h>
#include <time.h>
#include "../colored_output.h"

// Inclusion de stb_image pour le chargement d'images
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Fonction utilitaire pour vérifier si un fichier est une image
static bool is_image_file(const char *filename) {
    const char *ext = strrchr(filename, '.');
    if (!ext) return false;
    
    return (strcasecmp(ext, ".jpg") == 0 || 
            strcasecmp(ext, ".jpeg") == 0 ||
            strcasecmp(ext, ".png") == 0 ||
            strcasecmp(ext, ".bmp") == 0 ||
            strcasecmp(ext, ".tga") == 0);
}

// Fonction utilitaire pour mélanger un ImageSet (Fisher-Yates shuffle)
static void shuffle_image_set(ImageSet *set) {
    if (!set || set->count <= 1) return;
    
    srand(time(NULL));
    for (size_t i = set->count - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        
        // Échanger les éléments i et j
        ImageInfo temp = set->images[i];
        set->images[i] = set->images[j];
        set->images[j] = temp;
    }
}

ImageSet *load_image_set(const char *directory_path) {
    if (!directory_path) {
        printf("Erreur: chemin de répertoire invalide (NULL)\n");
        return NULL;
    }

    DIR *dir = opendir(directory_path);
    if (!dir) {
        printf("Erreur: impossible d'ouvrir le répertoire '%s'\n", directory_path);
        return NULL;
    }

    ImageSet *set = malloc(sizeof(ImageSet));
    if (!set) {
        printf("Erreur: allocation mémoire pour ImageSet\n");
        closedir(dir);
        return NULL;
    }

    set->capacity = 1000;
    set->count = 0;
    set->images = malloc(set->capacity * sizeof(ImageInfo));
    set->class_names = malloc(100 * sizeof(char*)); // Max 100 classes
    set->num_classes = 0;

    if (!set->images || !set->class_names) {
        printf("Erreur: allocation mémoire pour les images\n");
        free_image_set(set);
        closedir(dir);
        return NULL;
    }

    // Parcourir les sous-répertoires (chaque sous-répertoire = une classe)
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue; // Ignorer . et ..

        char class_path[512];
        snprintf(class_path, sizeof(class_path), "%s/%s", directory_path, entry->d_name);

        struct stat statbuf;
        if (stat(class_path, &statbuf) != 0 || !S_ISDIR(statbuf.st_mode)) {
            continue; // Ignorer les fichiers non-répertoires
        }

        // Ajouter la classe si elle n'existe pas déjà
        int class_index = -1;
        for (size_t i = 0; i < set->num_classes; i++) {
            if (strcmp(set->class_names[i], entry->d_name) == 0) {
                class_index = i;
                break;
            }
        }
        
        if (class_index == -1) {
            set->class_names[set->num_classes] = strdup(entry->d_name);
            class_index = set->num_classes;
            set->num_classes++;
        }

        // Parcourir les images dans ce répertoire de classe
        DIR *class_dir = opendir(class_path);
        if (!class_dir) continue;

        struct dirent *img_entry;
        while ((img_entry = readdir(class_dir)) != NULL) {
            if (!is_image_file(img_entry->d_name)) continue;

            // Redimensionner si nécessaire
            if (set->count >= set->capacity) {
                set->capacity *= 2;
                ImageInfo *new_images = realloc(set->images, set->capacity * sizeof(ImageInfo));
                if (!new_images) {
                    printf("Erreur: redimensionnement du tableau d'images\n");
                    closedir(class_dir);
                    closedir(dir);
                    free_image_set(set);
                    return NULL;
                }
                set->images = new_images;
            }

            // Ajouter l'image
            ImageInfo *img = &set->images[set->count];
            snprintf(img->filepath, sizeof(img->filepath), "%s/%s", class_path, img_entry->d_name);
            img->label = class_index;
            strncpy(img->class_name, entry->d_name, sizeof(img->class_name) - 1);
            img->class_name[sizeof(img->class_name) - 1] = '\0';
            set->count++;
        }
        closedir(class_dir);
    }
    closedir(dir);

    char info_msg[256];
    snprintf(info_msg, sizeof(info_msg), 
            "ImageSet chargé: %zu images, %zu classes depuis '%s'", 
            set->count, set->num_classes, directory_path);
    print_dataset_info(info_msg);

    return set;
}

void free_image_set(ImageSet *set) {
    if (!set) return;
    
    if (set->images) free(set->images);
    
    if (set->class_names) {
        for (size_t i = 0; i < set->num_classes; i++) {
            if (set->class_names[i]) free(set->class_names[i]);
        }
        free(set->class_names);
    }
    
    free(set);
}

float *load_image_data(const char *filepath, int width, int height, int channels) {
    if (!filepath) return NULL;

    int img_width, img_height, img_channels;
    unsigned char *img_data = stbi_load(filepath, &img_width, &img_height, &img_channels, channels);
    
    if (!img_data) {
        printf("Erreur: impossible de charger l'image '%s': %s\n", filepath, stbi_failure_reason());
        return NULL;
    }

    // Redimensionner l'image si nécessaire (simple nearest neighbor)
    size_t target_size = width * height * channels;
    float *float_data = malloc(target_size * sizeof(float));
    if (!float_data) {
        printf("Erreur: allocation mémoire pour les données d'image\n");
        stbi_image_free(img_data);
        return NULL;
    }

    // Conversion simple - si les dimensions correspondent exactement
    if (img_width == width && img_height == height && img_channels == channels) {
        for (size_t i = 0; i < target_size; i++) {
            // Normalisation améliorée : [0,1] puis centrage autour de 0.5
            float normalized = img_data[i] / 255.0f;
            float_data[i] = (normalized - 0.5f) * 2.0f; // Normalisation [-1,1]
        }
    } else {
        // Redimensionnement simple (nearest neighbor)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int src_x = (x * img_width) / width;
                int src_y = (y * img_height) / height;
                
                for (int c = 0; c < channels; c++) {
                    int src_idx = (src_y * img_width + src_x) * img_channels + c;
                    int dst_idx = (y * width + x) * channels + c;
                    
                    if (c < img_channels) {
                        // Normalisation améliorée : [0,1] puis centrage autour de 0.5
                        float normalized = img_data[src_idx] / 255.0f;
                        float_data[dst_idx] = (normalized - 0.5f) * 2.0f; // Normalisation [-1,1]
                    } else {
                        float_data[dst_idx] = 0.0f; // Padding si moins de canaux
                    }
                }
            }
        }
    }

    stbi_image_free(img_data);
    return float_data;
}

Dataset *convert_image_set_to_dataset(const ImageSet *set, int width, int height, int channels, size_t num_classes) {
    if (!set || set->count == 0) {
        printf("Erreur: ImageSet invalide ou vide\n");
        return NULL;
    }

    size_t input_size = width * height * channels;
    
    // Pour la classification binaire, utiliser 1 sortie au lieu de num_classes
    size_t output_size = (num_classes == 2) ? 1 : num_classes;
    
    Dataset *dataset = dataset_create(set->count, input_size, output_size);
    if (!dataset) {
        printf("Erreur: création du dataset\n");
        return NULL;
    }

    // Créer une copie de l'ImageSet pour le mélanger
    ImageSet *shuffled_set = malloc(sizeof(ImageSet));
    if (!shuffled_set) {
        printf("Erreur: allocation mémoire pour le mélange\n");
        dataset_free(dataset);
        return NULL;
    }
    
    // Copier les données
    shuffled_set->count = set->count;
    shuffled_set->capacity = set->capacity;
    shuffled_set->num_classes = set->num_classes;
    shuffled_set->images = malloc(set->count * sizeof(ImageInfo));
    shuffled_set->class_names = set->class_names; // Partager les noms de classes
    
    if (!shuffled_set->images) {
        printf("Erreur: allocation mémoire pour les images mélangées\n");
        free(shuffled_set);
        dataset_free(dataset);
        return NULL;
    }
    
    // Copier les images
    for (size_t i = 0; i < set->count; i++) {
        shuffled_set->images[i] = set->images[i];
    }
    
    // Mélanger les données
    shuffle_image_set(shuffled_set);
    printf("✅ Dataset mélangé pour améliorer l'apprentissage\n");

    char progress_msg[256];
    for (size_t i = 0; i < shuffled_set->count; i++) {
        if (i % 100 == 0) {
            snprintf(progress_msg, sizeof(progress_msg), 
                    "Conversion des images: %zu/%zu", i, shuffled_set->count);
            print_dataset_info(progress_msg);
        }

        // Charger les données de l'image
        float *img_data = load_image_data(shuffled_set->images[i].filepath, width, height, channels);
        if (!img_data) {
            printf("Erreur: chargement de l'image '%s'\n", shuffled_set->images[i].filepath);
            free(shuffled_set->images);
            free(shuffled_set);
            dataset_free(dataset);
            return NULL;
        }

        // Copier les données d'entrée
        for (size_t j = 0; j < input_size; j++) {
            dataset->inputs[i][j] = img_data[j];
        }
        free(img_data);

        // Créer le vecteur de sortie selon le type de classification
        if (num_classes == 2) {
            // Classification binaire : 1 sortie (0 ou 1)
            dataset->outputs[i][0] = (float)shuffled_set->images[i].label;
        } else {
            // Classification multi-classe : one-hot encoding
            for (size_t j = 0; j < num_classes; j++) {
                dataset->outputs[i][j] = (j == (size_t)shuffled_set->images[i].label) ? 1.0f : 0.0f;
            }
        }
    }

    // Nettoyer la copie mélangée
    free(shuffled_set->images);
    free(shuffled_set);

    dataset->num_samples = set->count;
    
    char success_msg[256];
    snprintf(success_msg, sizeof(success_msg), 
            "Dataset d'images créé: %zu échantillons, %zu entrées, %zu sorties (mélangé)", 
            dataset->num_samples, dataset->input_cols, dataset->output_cols);
    print_dataset_success(success_msg);
    
    return dataset;
}

ImageSet *merge_image_sets(const ImageSet *set1, const ImageSet *set2) {
    if (!set1 || !set2) return NULL;

    ImageSet *merged = malloc(sizeof(ImageSet));
    if (!merged) return NULL;

    merged->capacity = set1->count + set2->count;
    merged->count = 0;
    merged->images = malloc(merged->capacity * sizeof(ImageInfo));
    merged->class_names = malloc(100 * sizeof(char*));
    merged->num_classes = 0;

    if (!merged->images || !merged->class_names) {
        free_image_set(merged);
        return NULL;
    }

    // Copier les classes de set1
    for (size_t i = 0; i < set1->num_classes; i++) {
        merged->class_names[merged->num_classes] = strdup(set1->class_names[i]);
        merged->num_classes++;
    }

    // Ajouter les classes de set2 si elles n'existent pas
    for (size_t i = 0; i < set2->num_classes; i++) {
        bool exists = false;
        for (size_t j = 0; j < merged->num_classes; j++) {
            if (strcmp(merged->class_names[j], set2->class_names[i]) == 0) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            merged->class_names[merged->num_classes] = strdup(set2->class_names[i]);
            merged->num_classes++;
        }
    }

    // Copier les images de set1
    for (size_t i = 0; i < set1->count; i++) {
        merged->images[merged->count] = set1->images[i];
        merged->count++;
    }

    // Copier les images de set2 avec mise à jour des labels
    for (size_t i = 0; i < set2->count; i++) {
        merged->images[merged->count] = set2->images[i];
        
        // Trouver le nouveau label dans merged
        for (size_t j = 0; j < merged->num_classes; j++) {
            if (strcmp(merged->class_names[j], set2->images[i].class_name) == 0) {
                merged->images[merged->count].label = j;
                break;
            }
        }
        merged->count++;
    }

    return merged;
}

void print_image_set_stats(const ImageSet *set, const char *set_name) {
    if (!set) return;

    printf("\n=== Statistiques %s ===\n", set_name ? set_name : "ImageSet");
    printf("Nombre total d'images: %zu\n", set->count);
    printf("Nombre de classes: %zu\n", set->num_classes);
    
    printf("Classes détectées:\n");
    for (size_t i = 0; i < set->num_classes; i++) {
        size_t count = 0;
        for (size_t j = 0; j < set->count; j++) {
            if (set->images[j].label == (int)i) count++;
        }
        printf("  - %s: %zu images\n", set->class_names[i], count);
    }
    printf("\n");
}

Dataset *load_image_dataset_from_config(const RichConfig *config) {
    if (!config || !config->is_image_dataset) {
        printf("Erreur: configuration invalide ou non-image\n");
        return NULL;
    }

    printf("=== Chargement du dataset d'images ===\n");
    
    // Charger le dataset d'entraînement (obligatoire)
    ImageSet *train_set = NULL;
    if (config->image_train_dir[0] != '\0') {
        printf("Chargement du dataset d'entraînement...\n");
        train_set = load_image_set(config->image_train_dir);
        if (!train_set) {
            printf("Erreur: impossible de charger le dataset d'entraînement\n");
            return NULL;
        }
        print_image_set_stats(train_set, "Train");
    }

    // Charger le dataset de test (obligatoire)
    ImageSet *test_set = NULL;
    if (config->image_test_dir[0] != '\0') {
        printf("Chargement du dataset de test...\n");
        test_set = load_image_set(config->image_test_dir);
        if (!test_set) {
            printf("Erreur: impossible de charger le dataset de test\n");
            if (train_set) free_image_set(train_set);
            return NULL;
        }
        print_image_set_stats(test_set, "Test");
    }

    // Charger le dataset de validation (optionnel)
    ImageSet *val_set = NULL;
    if (config->image_val_dir[0] != '\0') {
        printf("Chargement du dataset de validation...\n");
        val_set = load_image_set(config->image_val_dir);
        if (val_set) {
            print_image_set_stats(val_set, "Validation");
        } else {
            printf("Avertissement: impossible de charger le dataset de validation (optionnel)\n");
        }
    }

    // Fusionner tous les datasets
    ImageSet *combined_set = train_set;
    
    if (test_set) {
        ImageSet *temp = merge_image_sets(combined_set, test_set);
        if (combined_set != train_set) free_image_set(combined_set);
        free_image_set(test_set);
        combined_set = temp;
        
        if (!combined_set) {
            printf("Erreur: fusion des datasets train/test\n");
            if (train_set) free_image_set(train_set);
            if (val_set) free_image_set(val_set);
            return NULL;
        }
    }

    if (val_set) {
        ImageSet *temp = merge_image_sets(combined_set, val_set);
        if (combined_set != train_set) free_image_set(combined_set);
        free_image_set(val_set);
        combined_set = temp;
        
        if (!combined_set) {
            printf("Erreur: fusion avec le dataset de validation\n");
            if (train_set) free_image_set(train_set);
            return NULL;
        }
    }

    if (!combined_set) {
        printf("Erreur: aucun dataset d'images chargé\n");
        return NULL;
    }

    print_image_set_stats(combined_set, "Dataset Combiné");

    // Convertir en Dataset
    Dataset *dataset = convert_image_set_to_dataset(
        combined_set, 
        config->image_width, 
        config->image_height, 
        config->image_channels, 
        combined_set->num_classes
    );

    free_image_set(combined_set);
    return dataset;
} 