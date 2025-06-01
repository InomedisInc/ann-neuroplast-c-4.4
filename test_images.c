#include <stdio.h>
#include <stdlib.h>
#include "src/data/image_loader.h"
#include "src/colored_output.h"

int main() {
    printf("=== Test de chargement d'images ===\n");
    
    // Test 1: Vérifier si le répertoire existe
    const char *train_dir = "/Users/fabricevaussenat/SynologyDrive/data-science/chest_xray/chest_xray/train";
    
    printf("Test 1: Chargement du répertoire d'entraînement...\n");
    ImageSet *train_set = load_image_set(train_dir);
    
    if (!train_set) {
        printf("❌ Erreur: impossible de charger le répertoire d'entraînement\n");
        printf("Vérifiez que le chemin existe: %s\n", train_dir);
        return 1;
    }
    
    printf("✅ Répertoire d'entraînement chargé avec succès\n");
    print_image_set_stats(train_set, "Train");
    
    // Test 2: Conversion en dataset
    printf("\nTest 2: Conversion en dataset...\n");
    Dataset *dataset = convert_image_set_to_dataset(train_set, 8, 8, 1, train_set->num_classes);
    
    if (!dataset) {
        printf("❌ Erreur: impossible de convertir en dataset\n");
        free_image_set(train_set);
        return 1;
    }
    
    printf("✅ Dataset créé avec succès\n");
    printf("Dimensions: %zu échantillons, %zu entrées, %zu sorties\n", 
           dataset->num_samples, dataset->input_cols, dataset->output_cols);
    
    // Test 3: Vérifier quelques échantillons
    printf("\nTest 3: Vérification des données...\n");
    for (size_t i = 0; i < 3 && i < dataset->num_samples; i++) {
        printf("Échantillon %zu: ", i);
        printf("Entrée[0]=%.3f, Entrée[63]=%.3f, ", 
               dataset->inputs[i][0], dataset->inputs[i][63]);
        printf("Sortie=%.1f\n", dataset->outputs[i][0]);
    }
    
    printf("\n✅ Test terminé avec succès !\n");
    
    // Nettoyage
    dataset_free(dataset);
    free_image_set(train_set);
    
    return 0;
} 