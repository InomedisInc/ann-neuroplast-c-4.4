#!/usr/bin/env python3
"""
NEUROPLAST-ANN Model Loader v4.3
=================================
Programme Python avec TensorFlow pour tester le chargement des modèles
sauvegardés par NEUROPLAST-ANN framework.

Fonctionnalités:
- Chargement des modèles .h5 et .pth
- Conversion vers TensorFlow
- Tests de prédiction
- Analyse des performances
- Visualisation des architectures
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from colorama import init, Fore, Style, Back
import h5py
import warnings

# Initialiser colorama pour les couleurs dans le terminal
init(autoreset=True)

class NeuroplastModelLoader:
    """Classe pour charger et tester les modèles NEUROPLAST-ANN"""
    
    def __init__(self, base_directory="../"):
        """
        Initialiser le chargeur de modèles
        
        Args:
            base_directory (str): Répertoire de base contenant les modèles sauvegardés
        """
        self.base_directory = Path(base_directory)
        self.model_directories = []
        self.loaded_models = {}
        self.model_info = {}
        
        print(f"{Fore.CYAN}🚀 NEUROPLAST-ANN Model Loader v4.3{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        
        self._discover_model_directories()
    
    def _discover_model_directories(self):
        """Découvrir automatiquement les répertoires de modèles"""
        pattern = "best_models_neuroplast_*"
        self.model_directories = list(self.base_directory.glob(pattern))
        
        if self.model_directories:
            print(f"{Fore.GREEN}✅ Répertoires de modèles trouvés:{Style.RESET_ALL}")
            for i, dir_path in enumerate(self.model_directories, 1):
                dataset_name = dir_path.name.replace("best_models_neuroplast_", "")
                print(f"   {i}. {Fore.YELLOW}{dataset_name}{Style.RESET_ALL} → {dir_path}")
        else:
            print(f"{Fore.RED}❌ Aucun répertoire de modèles trouvé{Style.RESET_ALL}")
            print(f"   Recherche dans: {self.base_directory}")
            print(f"   Pattern: {pattern}")
    
    def load_model_info(self, model_dir):
        """Charger les informations des modèles depuis best_models_info.json"""
        info_file = model_dir / "best_models_info.json"
        
        if info_file.exists():
            try:
                with open(info_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"{Fore.RED}❌ Erreur lecture {info_file}: {e}{Style.RESET_ALL}")
                return None
        else:
            print(f"{Fore.YELLOW}⚠️ Fichier info non trouvé: {info_file}{Style.RESET_ALL}")
            return None
    
    def load_h5_model(self, h5_file):
        """Charger un modèle .h5 et le convertir en TensorFlow"""
        try:
            print(f"{Fore.BLUE}📂 Chargement du modèle H5: {h5_file.name}{Style.RESET_ALL}")
            
            # Lire le fichier H5 créé par NEUROPLAST-ANN
            with h5py.File(h5_file, 'r') as f:
                # Afficher la structure du fichier
                print(f"   Structure H5:")
                self._print_h5_structure(f, "   ")
                
                # Extraire les informations du modèle
                if 'model_info' in f:
                    model_info = {}
                    for key in f['model_info'].keys():
                        if isinstance(f['model_info'][key][()], bytes):
                            model_info[key] = f['model_info'][key][()].decode('utf-8')
                        else:
                            model_info[key] = f['model_info'][key][()]
                    
                    print(f"   📊 Info modèle: {model_info}")
                
                # Extraire l'architecture
                if 'architecture' in f:
                    layers = []
                    for layer_name in f['architecture'].keys():
                        layer_group = f['architecture'][layer_name]
                        layer_info = {
                            'name': layer_name,
                            'input_size': layer_group['input_size'][()],
                            'output_size': layer_group['output_size'][()],
                            'activation': layer_group['activation'][()].decode('utf-8') if isinstance(layer_group['activation'][()], bytes) else layer_group['activation'][()]
                        }
                        
                        # Extraire les poids si disponibles
                        if 'weights' in layer_group:
                            layer_info['weights'] = layer_group['weights'][:]
                        if 'biases' in layer_group:
                            layer_info['biases'] = layer_group['biases'][:]
                        
                        layers.append(layer_info)
                    
                    # Créer un modèle TensorFlow équivalent
                    tf_model = self._create_tensorflow_model(layers)
                    return tf_model, model_info, layers
                
        except Exception as e:
            print(f"{Fore.RED}❌ Erreur chargement H5: {e}{Style.RESET_ALL}")
            return None, None, None
    
    def _print_h5_structure(self, group, indent=""):
        """Afficher la structure d'un fichier H5"""
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                print(f"{indent}📁 {key}/")
                self._print_h5_structure(item, indent + "  ")
            else:
                print(f"{indent}📄 {key}: {item.shape} {item.dtype}")
    
    def _create_tensorflow_model(self, layers):
        """Créer un modèle TensorFlow à partir des couches NEUROPLAST-ANN"""
        try:
            model = tf.keras.Sequential()
            
            for i, layer_info in enumerate(layers):
                input_size = layer_info['input_size']
                output_size = layer_info['output_size']
                activation = layer_info['activation'].lower()
                
                # Mapper les activations NEUROPLAST-ANN vers TensorFlow
                activation_map = {
                    'relu': 'relu',
                    'sigmoid': 'sigmoid',
                    'tanh': 'tanh',
                    'gelu': 'gelu',
                    'leaky_relu': tf.keras.layers.LeakyReLU(),
                    'elu': 'elu',
                    'mish': 'mish',
                    'swish': 'swish',
                    'neuroplast': 'relu'  # Fallback pour neuroplast
                }
                
                tf_activation = activation_map.get(activation, 'relu')
                
                if i == 0:
                    # Première couche avec input_shape
                    layer = tf.keras.layers.Dense(
                        output_size,
                        activation=tf_activation,
                        input_shape=(input_size,),
                        name=f"dense_{i+1}"
                    )
                else:
                    layer = tf.keras.layers.Dense(
                        output_size,
                        activation=tf_activation,
                        name=f"dense_{i+1}"
                    )
                
                model.add(layer)
                
                # Appliquer les poids si disponibles
                if 'weights' in layer_info and 'biases' in layer_info:
                    weights = layer_info['weights']
                    biases = layer_info['biases']
                    
                    # Reshape si nécessaire
                    if len(weights.shape) == 1:
                        weights = weights.reshape(input_size, output_size)
                    
                    layer.set_weights([weights, biases])
            
            # Compiler le modèle
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy' if layers[-1]['output_size'] == 1 else 'categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"{Fore.GREEN}✅ Modèle TensorFlow créé avec succès{Style.RESET_ALL}")
            return model
            
        except Exception as e:
            print(f"{Fore.RED}❌ Erreur création modèle TensorFlow: {e}{Style.RESET_ALL}")
            return None
    
    def test_model_prediction(self, model, dataset_name):
        """Tester les prédictions du modèle avec des données synthétiques"""
        try:
            print(f"{Fore.BLUE}🧪 Test de prédiction pour {dataset_name}{Style.RESET_ALL}")
            
            # Générer des données de test synthétiques
            input_shape = model.input_shape[1]
            test_data = np.random.randn(10, input_shape)
            
            # Faire des prédictions
            predictions = model.predict(test_data, verbose=0)
            
            print(f"   📊 Forme des données d'entrée: {test_data.shape}")
            print(f"   📊 Forme des prédictions: {predictions.shape}")
            print(f"   📊 Échantillon de prédictions:")
            
            for i, pred in enumerate(predictions[:5]):
                if len(pred) == 1:
                    print(f"      Échantillon {i+1}: {pred[0]:.4f}")
                else:
                    print(f"      Échantillon {i+1}: {pred}")
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Erreur test prédiction: {e}{Style.RESET_ALL}")
            return False
    
    def analyze_model_architecture(self, model, dataset_name):
        """Analyser et afficher l'architecture du modèle"""
        print(f"{Fore.BLUE}🏗️ Architecture du modèle {dataset_name}{Style.RESET_ALL}")
        
        # Résumé du modèle
        print("   📋 Résumé:")
        model.summary(print_fn=lambda x: print(f"      {x}"))
        
        # Statistiques
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        print(f"   📊 Paramètres totaux: {total_params:,}")
        print(f"   📊 Paramètres entraînables: {trainable_params:,}")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'layers': len(model.layers)
        }
    
    def visualize_model(self, model, dataset_name, save_path=None):
        """Visualiser l'architecture du modèle"""
        try:
            if save_path is None:
                save_path = f"model_architecture_{dataset_name}.png"
            
            tf.keras.utils.plot_model(
                model,
                to_file=save_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=96
            )
            
            print(f"{Fore.GREEN}✅ Diagramme sauvegardé: {save_path}{Style.RESET_ALL}")
            return save_path
            
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ Impossible de créer le diagramme: {e}{Style.RESET_ALL}")
            return None
    
    def load_and_test_all_models(self):
        """Charger et tester tous les modèles trouvés"""
        results = {}
        
        for model_dir in self.model_directories:
            dataset_name = model_dir.name.replace("best_models_neuroplast_", "")
            
            print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}🔍 ANALYSE DU DATASET: {dataset_name.upper()}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
            
            # Charger les informations des modèles
            model_info = self.load_model_info(model_dir)
            
            # Chercher les fichiers de modèles
            h5_files = list(model_dir.glob("*.h5"))
            
            if h5_files:
                print(f"{Fore.GREEN}📁 {len(h5_files)} modèles H5 trouvés{Style.RESET_ALL}")
                
                # Tester le premier modèle (meilleur)
                best_model_file = model_dir / "model_1.h5"
                if best_model_file.exists():
                    model, info, layers = self.load_h5_model(best_model_file)
                    
                    if model is not None:
                        # Analyser l'architecture
                        arch_stats = self.analyze_model_architecture(model, dataset_name)
                        
                        # Tester les prédictions
                        pred_success = self.test_model_prediction(model, dataset_name)
                        
                        # Visualiser le modèle
                        viz_path = self.visualize_model(model, dataset_name)
                        
                        # Stocker les résultats
                        results[dataset_name] = {
                            'model': model,
                            'info': info,
                            'layers': layers,
                            'architecture_stats': arch_stats,
                            'prediction_test': pred_success,
                            'visualization': viz_path,
                            'model_info_json': model_info
                        }
                        
                        self.loaded_models[dataset_name] = model
                        self.model_info[dataset_name] = info
                    
                else:
                    print(f"{Fore.RED}❌ Fichier model_1.h5 non trouvé{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}⚠️ Aucun fichier .h5 trouvé dans {model_dir}{Style.RESET_ALL}")
        
        return results
    
    def generate_comparison_report(self, results):
        """Générer un rapport de comparaison des modèles"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 RAPPORT DE COMPARAISON DES MODÈLES{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        if not results:
            print(f"{Fore.RED}❌ Aucun modèle chargé pour la comparaison{Style.RESET_ALL}")
            return
        
        # Créer un DataFrame pour la comparaison
        comparison_data = []
        
        for dataset_name, result in results.items():
            if result['architecture_stats']:
                stats = result['architecture_stats']
                comparison_data.append({
                    'Dataset': dataset_name,
                    'Couches': stats['layers'],
                    'Paramètres Totaux': stats['total_params'],
                    'Paramètres Entraînables': stats['trainable_params'],
                    'Test Prédiction': '✅' if result['prediction_test'] else '❌',
                    'Visualisation': '✅' if result['visualization'] else '❌'
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(f"\n{Fore.GREEN}📋 Tableau de comparaison:{Style.RESET_ALL}")
            print(df.to_string(index=False))
            
            # Sauvegarder le rapport
            report_file = "model_comparison_report.csv"
            df.to_csv(report_file, index=False)
            print(f"\n{Fore.GREEN}✅ Rapport sauvegardé: {report_file}{Style.RESET_ALL}")
        
        # Statistiques globales
        total_models = len(results)
        successful_loads = sum(1 for r in results.values() if r['model'] is not None)
        
        print(f"\n{Fore.BLUE}📈 Statistiques globales:{Style.RESET_ALL}")
        print(f"   🎯 Modèles trouvés: {total_models}")
        print(f"   ✅ Chargements réussis: {successful_loads}")
        print(f"   📊 Taux de succès: {successful_loads/total_models*100:.1f}%" if total_models > 0 else "   📊 Taux de succès: 0%")

def main():
    """Fonction principale"""
    print(f"{Fore.CYAN}🚀 Démarrage du test de chargement des modèles NEUROPLAST-ANN{Style.RESET_ALL}")
    
    # Créer le chargeur de modèles
    loader = NeuroplastModelLoader()
    
    if not loader.model_directories:
        print(f"\n{Fore.RED}❌ Aucun modèle trouvé. Assurez-vous d'avoir exécuté NEUROPLAST-ANN avec --test-all{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Exemple: ./neuroplast-ann --config config/cancer_simple.yml --test-all{Style.RESET_ALL}")
        return
    
    # Charger et tester tous les modèles
    results = loader.load_and_test_all_models()
    
    # Générer le rapport de comparaison
    loader.generate_comparison_report(results)
    
    print(f"\n{Fore.GREEN}🎉 Test de chargement terminé avec succès !{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📁 Fichiers générés dans le répertoire courant{Style.RESET_ALL}")

if __name__ == "__main__":
    # Supprimer les warnings TensorFlow pour un affichage plus propre
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    main() 