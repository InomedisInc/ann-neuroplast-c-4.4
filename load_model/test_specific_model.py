#!/usr/bin/env python3
"""
Test Spécifique de Modèle NEUROPLAST-ANN
========================================
Script pour tester un modèle spécifique avec des données réelles
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from colorama import init, Fore, Style
import argparse

# Initialiser colorama
init(autoreset=True)

class ModelTester:
    """Classe pour tester un modèle spécifique avec des données réelles"""
    
    def __init__(self, model_path, dataset_path=None):
        """
        Initialiser le testeur de modèle
        
        Args:
            model_path (str): Chemin vers le modèle .h5
            dataset_path (str): Chemin vers le dataset de test
        """
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.model = None
        self.scaler = StandardScaler()
        
        print(f"{Fore.CYAN}🧪 Test Spécifique de Modèle NEUROPLAST-ANN{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    
    def load_tensorflow_model(self):
        """Charger le modèle TensorFlow depuis le fichier H5"""
        try:
            print(f"{Fore.BLUE}📂 Chargement du modèle: {self.model_path}{Style.RESET_ALL}")
            
            # Importer le model_loader pour utiliser la conversion
            sys.path.append('.')
            from model_loader import NeuroplastModelLoader
            
            loader = NeuroplastModelLoader()
            model, info, layers = loader.load_h5_model(self.model_path)
            
            if model is not None:
                self.model = model
                print(f"{Fore.GREEN}✅ Modèle chargé avec succès{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}❌ Échec du chargement du modèle{Style.RESET_ALL}")
                return False
                
        except Exception as e:
            print(f"{Fore.RED}❌ Erreur: {e}{Style.RESET_ALL}")
            return False
    
    def load_dataset(self):
        """Charger le dataset de test"""
        if not self.dataset_path or not self.dataset_path.exists():
            print(f"{Fore.YELLOW}⚠️ Pas de dataset spécifié, génération de données synthétiques{Style.RESET_ALL}")
            return self.generate_synthetic_data()
        
        try:
            print(f"{Fore.BLUE}📊 Chargement du dataset: {self.dataset_path}{Style.RESET_ALL}")
            
            # Charger selon l'extension
            if self.dataset_path.suffix.lower() == '.csv':
                df = pd.read_csv(self.dataset_path)
            elif self.dataset_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(self.dataset_path)
            else:
                raise ValueError(f"Format de fichier non supporté: {self.dataset_path.suffix}")
            
            print(f"   📋 Forme du dataset: {df.shape}")
            print(f"   📋 Colonnes: {list(df.columns)}")
            
            # Séparer features et target (dernière colonne = target)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
            # Normaliser les features
            X_scaled = self.scaler.fit_transform(X)
            
            # Diviser en train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"   📊 Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"{Fore.RED}❌ Erreur chargement dataset: {e}{Style.RESET_ALL}")
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self):
        """Générer des données synthétiques pour le test"""
        print(f"{Fore.BLUE}🎲 Génération de données synthétiques{Style.RESET_ALL}")
        
        if self.model is None:
            print(f"{Fore.RED}❌ Modèle non chargé{Style.RESET_ALL}")
            return None, None, None, None
        
        # Obtenir la forme d'entrée du modèle
        input_shape = self.model.input_shape[1]
        output_shape = self.model.output_shape[1] if len(self.model.output_shape) > 1 else 1
        
        # Générer des données
        n_samples = 1000
        X = np.random.randn(n_samples, input_shape)
        
        # Générer des labels selon le type de problème
        if output_shape == 1:
            # Classification binaire
            y = np.random.randint(0, 2, n_samples)
        else:
            # Classification multi-classe
            y = np.random.randint(0, output_shape, n_samples)
        
        # Diviser en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   📊 Données synthétiques générées: {X.shape}")
        print(f"   📊 Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, X_test, y_test):
        """Évaluer le modèle sur les données de test"""
        if self.model is None:
            print(f"{Fore.RED}❌ Modèle non chargé{Style.RESET_ALL}")
            return
        
        print(f"{Fore.BLUE}📈 Évaluation du modèle{Style.RESET_ALL}")
        
        try:
            # Prédictions
            y_pred_proba = self.model.predict(X_test, verbose=0)
            
            # Convertir en classes
            if y_pred_proba.shape[1] == 1:
                # Classification binaire
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            else:
                # Classification multi-classe
                y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Calculer les métriques
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"   🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Rapport de classification
            print(f"\n   📊 Rapport de classification:")
            report = classification_report(y_test, y_pred, output_dict=True)
            
            for class_name, metrics in report.items():
                if isinstance(metrics, dict):
                    precision = metrics.get('precision', 0)
                    recall = metrics.get('recall', 0)
                    f1 = metrics.get('f1-score', 0)
                    support = metrics.get('support', 0)
                    
                    print(f"      {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={support}")
            
            # Matrice de confusion
            self.plot_confusion_matrix(y_test, y_pred)
            
            # Distribution des prédictions
            self.plot_prediction_distribution(y_pred_proba, y_test)
            
            return {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': report
            }
            
        except Exception as e:
            print(f"{Fore.RED}❌ Erreur évaluation: {e}{Style.RESET_ALL}")
            return None
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Tracer la matrice de confusion"""
        try:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=np.unique(y_true), 
                       yticklabels=np.unique(y_true))
            
            plt.title('Matrice de Confusion')
            plt.xlabel('Prédictions')
            plt.ylabel('Vraies Valeurs')
            
            # Sauvegarder
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{Fore.GREEN}   ✅ Matrice de confusion sauvegardée: confusion_matrix.png{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.YELLOW}   ⚠️ Impossible de créer la matrice de confusion: {e}{Style.RESET_ALL}")
    
    def plot_prediction_distribution(self, y_pred_proba, y_true):
        """Tracer la distribution des prédictions"""
        try:
            plt.figure(figsize=(12, 4))
            
            # Sous-graphique 1: Distribution des probabilités
            plt.subplot(1, 2, 1)
            if y_pred_proba.shape[1] == 1:
                plt.hist(y_pred_proba.flatten(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                plt.xlabel('Probabilité de Classe Positive')
            else:
                for i in range(y_pred_proba.shape[1]):
                    plt.hist(y_pred_proba[:, i], bins=20, alpha=0.5, label=f'Classe {i}')
                plt.xlabel('Probabilités')
                plt.legend()
            
            plt.ylabel('Fréquence')
            plt.title('Distribution des Probabilités de Prédiction')
            plt.grid(True, alpha=0.3)
            
            # Sous-graphique 2: Distribution des vraies classes
            plt.subplot(1, 2, 2)
            unique, counts = np.unique(y_true, return_counts=True)
            plt.bar(unique, counts, color='lightcoral', alpha=0.7, edgecolor='black')
            plt.xlabel('Classes')
            plt.ylabel('Nombre d\'échantillons')
            plt.title('Distribution des Vraies Classes')
            plt.grid(True, alpha=0.3)
            
            # Sauvegarder
            plt.tight_layout()
            plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{Fore.GREEN}   ✅ Distribution sauvegardée: prediction_distribution.png{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.YELLOW}   ⚠️ Impossible de créer la distribution: {e}{Style.RESET_ALL}")
    
    def run_complete_test(self):
        """Exécuter le test complet"""
        print(f"{Fore.MAGENTA}🚀 Démarrage du test complet{Style.RESET_ALL}")
        
        # 1. Charger le modèle
        if not self.load_tensorflow_model():
            return False
        
        # 2. Charger les données
        X_train, X_test, y_train, y_test = self.load_dataset()
        if X_test is None:
            return False
        
        # 3. Évaluer le modèle
        results = self.evaluate_model(X_test, y_test)
        
        if results:
            print(f"\n{Fore.GREEN}🎉 Test terminé avec succès !{Style.RESET_ALL}")
            print(f"{Fore.CYAN}📁 Fichiers générés:{Style.RESET_ALL}")
            print(f"   - confusion_matrix.png")
            print(f"   - prediction_distribution.png")
            return True
        else:
            print(f"\n{Fore.RED}❌ Échec du test{Style.RESET_ALL}")
            return False

def main():
    """Fonction principale avec arguments en ligne de commande"""
    parser = argparse.ArgumentParser(description='Tester un modèle NEUROPLAST-ANN spécifique')
    parser.add_argument('model_path', help='Chemin vers le fichier .h5 du modèle')
    parser.add_argument('--dataset', '-d', help='Chemin vers le dataset de test (optionnel)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mode verbeux')
    
    args = parser.parse_args()
    
    # Vérifier que le modèle existe
    if not Path(args.model_path).exists():
        print(f"{Fore.RED}❌ Fichier modèle non trouvé: {args.model_path}{Style.RESET_ALL}")
        return
    
    # Créer le testeur et exécuter
    tester = ModelTester(args.model_path, args.dataset)
    success = tester.run_complete_test()
    
    if success:
        print(f"\n{Fore.GREEN}✅ Test réussi !{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}❌ Test échoué !{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 