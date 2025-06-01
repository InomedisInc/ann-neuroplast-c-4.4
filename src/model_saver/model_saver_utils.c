#include "model_saver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Afficher le classement des modèles
void model_saver_print_rankings(ModelSaver *saver) {
    if (!saver) return;
    
    printf("\n=== TOP %d MODÈLES ===\n", saver->count);
    printf("Rang | Nom du modèle | Score | Précision | Perte | Val. Précision | Val. Perte | Époque\n");
    printf("-----|---------------|-------|-----------|-------|----------------|------------|-------\n");
    
    // Trier les modèles par score décroissant
    SavedModel temp_models[10];
    memcpy(temp_models, saver->models, sizeof(SavedModel) * saver->count);
    
    // Tri à bulles simple
    for (int i = 0; i < saver->count - 1; i++) {
        for (int j = 0; j < saver->count - i - 1; j++) {
            if (temp_models[j].score < temp_models[j + 1].score) {
                SavedModel temp = temp_models[j];
                temp_models[j] = temp_models[j + 1];
                temp_models[j + 1] = temp;
            }
        }
    }
    
    for (int i = 0; i < saver->count; i++) {
        printf("%4d | %-13s | %5.3f | %9.3f | %5.3f | %14.3f | %10.3f | %6d\n",
               i + 1,
               temp_models[i].metadata.model_name,
               temp_models[i].score,
               temp_models[i].metadata.accuracy,
               temp_models[i].metadata.loss,
               temp_models[i].metadata.validation_accuracy,
               temp_models[i].metadata.validation_loss,
               temp_models[i].metadata.epoch);
    }
    printf("\n");
}

// Sauvegarder tous les modèles
int model_saver_save_all(ModelSaver *saver, SaveFormat format) {
    if (!saver) return -1;
    
    char filepath[512];
    int success_count = 0;
    
    for (int i = 0; i < saver->count; i++) {
        SavedModel *model = &saver->models[i];
        
        if (format == FORMAT_PTH || format == FORMAT_BOTH) {
            snprintf(filepath, sizeof(filepath), "%s/%s.pth", 
                    saver->save_directory, model->metadata.model_name);
            
            if (model_saver_save_pth(model, filepath) == 0) {
                printf("Sauvegardé: %s\n", filepath);
                success_count++;
            } else {
                printf("Erreur lors de la sauvegarde: %s\n", filepath);
            }
        }
        
        if (format == FORMAT_H5 || format == FORMAT_BOTH) {
            snprintf(filepath, sizeof(filepath), "%s/%s.h5", 
                    saver->save_directory, model->metadata.model_name);
            
            if (model_saver_save_h5(model, filepath) == 0) {
                printf("Sauvegardé: %s\n", filepath);
                success_count++;
            } else {
                printf("Erreur lors de la sauvegarde: %s\n", filepath);
            }
        }
    }
    
    return success_count;
}

// Charger un modèle spécifique
NeuralNetwork *model_saver_load_model(const char *filepath, ModelMetadata *metadata) {
    if (!filepath) return NULL;
    
    // Déterminer le format par l'extension
    const char *ext = strrchr(filepath, '.');
    if (!ext) return NULL;
    
    if (strcmp(ext, ".pth") == 0) {
        return model_saver_load_pth(filepath, metadata);
    } else if (strcmp(ext, ".h5") == 0) {
        return model_saver_load_h5(filepath, metadata);
    }
    
    return NULL;
}

// Exporter l'interface Python
int model_saver_export_python_interface(ModelSaver *saver, const char *output_file) {
    if (!saver || !output_file) return -1;
    
    FILE *file = fopen(output_file, "w");
    if (!file) return -1;
    
    fprintf(file, "#!/usr/bin/env python3\n");
    fprintf(file, "# -*- coding: utf-8 -*-\n");
    fprintf(file, "\"\"\"\n");
    fprintf(file, "Interface Python pour charger les modèles sauvegardés\n");
    fprintf(file, "Généré automatiquement par model_saver\n");
    fprintf(file, "\"\"\"\n\n");
    
    fprintf(file, "import json\nimport numpy as np\nfrom typing import Dict, List, Tuple, Optional\n\n");
    
    fprintf(file, "class NeuralNetworkLoader:\n");
    fprintf(file, "    \"\"\"Classe pour charger et utiliser les modèles sauvegardés\"\"\"\n\n");
    
    fprintf(file, "    def __init__(self, model_directory: str = '%s'):\n", saver->save_directory);
    fprintf(file, "        self.model_directory = model_directory\n");
    fprintf(file, "        self.available_models = [\n");
    
    for (int i = 0; i < saver->count; i++) {
        fprintf(file, "            '%s',\n", saver->models[i].metadata.model_name);
    }
    
    fprintf(file, "        ]\n\n");
    
    fprintf(file, "    def load_model_h5(self, model_name: str) -> Dict:\n");
    fprintf(file, "        \"\"\"Charger un modèle au format H5 (JSON)\"\"\"\n");
    fprintf(file, "        filepath = f'{self.model_directory}/{model_name}.h5'\n");
    fprintf(file, "        with open(filepath, 'r') as f:\n");
    fprintf(file, "            return json.load(f)\n\n");
    
    fprintf(file, "    def get_model_weights(self, model_name: str) -> List[np.ndarray]:\n");
    fprintf(file, "        \"\"\"Extraire les poids d'un modèle\"\"\"\n");
    fprintf(file, "        model_data = self.load_model_h5(model_name)\n");
    fprintf(file, "        weights = []\n");
    fprintf(file, "        \n");
    fprintf(file, "        for layer in model_data['parameters']['layers']:\n");
    fprintf(file, "            layer_weights = np.array(layer['weights'], dtype=np.float32)\n");
    fprintf(file, "            weights.append(layer_weights)\n");
    fprintf(file, "        \n");
    fprintf(file, "        return weights\n\n");
    
    fprintf(file, "    def get_model_biases(self, model_name: str) -> List[np.ndarray]:\n");
    fprintf(file, "        \"\"\"Extraire les biais d'un modèle\"\"\"\n");
    fprintf(file, "        model_data = self.load_model_h5(model_name)\n");
    fprintf(file, "        biases = []\n");
    fprintf(file, "        \n");
    fprintf(file, "        for layer in model_data['parameters']['layers']:\n");
    fprintf(file, "            layer_biases = np.array(layer['biases'], dtype=np.float32)\n");
    fprintf(file, "            biases.append(layer_biases)\n");
    fprintf(file, "        \n");
    fprintf(file, "        return biases\n\n");
    
    fprintf(file, "    def predict(self, model_name: str, input_data: np.ndarray) -> np.ndarray:\n");
    fprintf(file, "        \"\"\"Faire une prédiction avec un modèle\"\"\"\n");
    fprintf(file, "        weights = self.get_model_weights(model_name)\n");
    fprintf(file, "        biases = self.get_model_biases(model_name)\n");
    fprintf(file, "        \n");
    fprintf(file, "        x = input_data.copy()\n");
    fprintf(file, "        \n");
    fprintf(file, "        for w, b in zip(weights, biases):\n");
    fprintf(file, "            x = np.dot(x, w.T) + b\n");
    fprintf(file, "            # Appliquer l'activation (ReLU par défaut)\n");
    fprintf(file, "            x = np.maximum(0, x)\n");
    fprintf(file, "        \n");
    fprintf(file, "        return x\n\n");
    
    fprintf(file, "    def get_model_info(self, model_name: str) -> Dict:\n");
    fprintf(file, "        \"\"\"Obtenir les informations d'un modèle\"\"\"\n");
    fprintf(file, "        model_data = self.load_model_h5(model_name)\n");
    fprintf(file, "        return model_data['metadata']\n\n");
    
    fprintf(file, "    def list_models(self) -> List[str]:\n");
    fprintf(file, "        \"\"\"Lister tous les modèles disponibles\"\"\"\n");
    fprintf(file, "        return self.available_models.copy()\n\n");
    
    fprintf(file, "# Exemple d'utilisation\n");
    fprintf(file, "if __name__ == '__main__':\n");
    fprintf(file, "    loader = NeuralNetworkLoader()\n");
    fprintf(file, "    \n");
    fprintf(file, "    print('Modèles disponibles:')\n");
    fprintf(file, "    for model in loader.list_models():\n");
    fprintf(file, "        info = loader.get_model_info(model)\n");
    fprintf(file, "        print(f'  {model}: Précision={info[\"accuracy\"]:.3f}, Perte={info[\"loss\"]:.3f}')\n");
    fprintf(file, "    \n");
    fprintf(file, "    # Exemple de prédiction\n");
    fprintf(file, "    if loader.available_models:\n");
    fprintf(file, "        best_model = loader.available_models[0]\n");
    fprintf(file, "        print(f'\\nUtilisation du modèle: {best_model}')\n");
    fprintf(file, "        \n");
    fprintf(file, "        # Créer des données d'exemple (à adapter selon votre cas)\n");
    fprintf(file, "        sample_input = np.random.randn(1, 10)  # Exemple: 10 features\n");
    fprintf(file, "        prediction = loader.predict(best_model, sample_input)\n");
    fprintf(file, "        print(f'Prédiction: {prediction}')\n");
    
    fclose(file);
    
    printf("Interface Python exportée vers: %s\n", output_file);
    return 0;
} 