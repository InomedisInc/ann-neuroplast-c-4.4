#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Python pour charger les modèles sauvegardés
Généré automatiquement par model_saver
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional

class NeuralNetworkLoader:
    """Classe pour charger et utiliser les modèles sauvegardés"""

    def __init__(self, model_directory: str = './saved_models'):
        self.model_directory = model_directory
        self.available_models = [
            'model_1',
            'model_2',
            'model_3',
            'model_12',
            'model_11',
            'model_6',
            'model_7',
            'model_8',
            'model_9',
            'model_10',
        ]

    def load_model_h5(self, model_name: str) -> Dict:
        """Charger un modèle au format H5 (JSON)"""
        filepath = f'{self.model_directory}/{model_name}.h5'
        with open(filepath, 'r') as f:
            return json.load(f)

    def get_model_weights(self, model_name: str) -> List[np.ndarray]:
        """Extraire les poids d'un modèle"""
        model_data = self.load_model_h5(model_name)
        weights = []
        
        for layer in model_data['parameters']['layers']:
            layer_weights = np.array(layer['weights'], dtype=np.float32)
            weights.append(layer_weights)
        
        return weights

    def get_model_biases(self, model_name: str) -> List[np.ndarray]:
        """Extraire les biais d'un modèle"""
        model_data = self.load_model_h5(model_name)
        biases = []
        
        for layer in model_data['parameters']['layers']:
            layer_biases = np.array(layer['biases'], dtype=np.float32)
            biases.append(layer_biases)
        
        return biases

    def predict(self, model_name: str, input_data: np.ndarray) -> np.ndarray:
        """Faire une prédiction avec un modèle"""
        weights = self.get_model_weights(model_name)
        biases = self.get_model_biases(model_name)
        
        x = input_data.copy()
        
        for w, b in zip(weights, biases):
            x = np.dot(x, w.T) + b
            # Appliquer l'activation (ReLU par défaut)
            x = np.maximum(0, x)
        
        return x

    def get_model_info(self, model_name: str) -> Dict:
        """Obtenir les informations d'un modèle"""
        model_data = self.load_model_h5(model_name)
        return model_data['metadata']

    def list_models(self) -> List[str]:
        """Lister tous les modèles disponibles"""
        return self.available_models.copy()

# Exemple d'utilisation
if __name__ == '__main__':
    loader = NeuralNetworkLoader()
    
    print('Modèles disponibles:')
    for model in loader.list_models():
        info = loader.get_model_info(model)
        print(f'  {model}: Précision={info["accuracy"]:.3f}, Perte={info["loss"]:.3f}')
    
    # Exemple de prédiction
    if loader.available_models:
        best_model = loader.available_models[0]
        print(f'\nUtilisation du modèle: {best_model}')
        
        # Créer des données d'exemple (à adapter selon votre cas)
        sample_input = np.random.randn(1, 10)  # Exemple: 10 features
        prediction = loader.predict(best_model, sample_input)
        print(f'Prédiction: {prediction}')
