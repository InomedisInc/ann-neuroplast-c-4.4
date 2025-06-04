# 🐍 NEUROPLAST-ANN Model Loader v4.3

Programme Python avec TensorFlow pour tester le chargement et l'évaluation des modèles sauvegardés par NEUROPLAST-ANN framework.

## 🎯 Fonctionnalités

- **🔄 Chargement automatique** des modèles .h5 sauvegardés par NEUROPLAST-ANN
- **🧠 Conversion TensorFlow** : Conversion des modèles C vers TensorFlow/Keras
- **📊 Tests de prédiction** avec données réelles ou synthétiques
- **📈 Analyse des performances** (accuracy, precision, recall, F1-score)
- **📋 Rapports détaillés** avec comparaison multi-modèles
- **📊 Visualisations** (matrices de confusion, distributions)
- **🎨 Interface colorée** pour un suivi facile

## 📦 Installation

### 1. Installer les dépendances

```bash
cd load_model
pip install -r requirements.txt
```

### 2. Vérifier l'installation TensorFlow

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## 🚀 Utilisation

### 📋 **Test de tous les modèles (Recommandé)**

```bash
# Tester automatiquement tous les modèles trouvés
python model_loader.py
```

**Sortie attendue :**
```
🚀 NEUROPLAST-ANN Model Loader v4.3
==================================================
✅ Répertoires de modèles trouvés:
   1. cancer → ../best_models_neuroplast_cancer
   2. chest_xray → ../best_models_neuroplast_chest_xray
   3. diabetes → ../best_models_neuroplast_diabetes

============================================================
🔍 ANALYSE DU DATASET: CANCER
============================================================
📂 Chargement du modèle H5: model_1.h5
   Structure H5:
   📁 model_info/
   📁 architecture/
   📄 dataset_name: (1,) <U6
   📊 Info modèle: {'method': 'neuroplast', 'optimizer': 'adamw', 'activation': 'relu'}
✅ Modèle TensorFlow créé avec succès
🏗️ Architecture du modèle cancer
   📋 Résumé:
      Model: "sequential"
      _________________________________________________________________
      Layer (type)                 Output Shape              Param #   
      =================================================================
      dense_1 (Dense)              (None, 256)               768       
      dense_2 (Dense)              (None, 128)               32896     
      dense_3 (Dense)              (None, 1)                 129       
      =================================================================
      Total params: 33,793
      Trainable params: 33,793
      Non-trainable params: 0
   📊 Paramètres totaux: 33,793
   📊 Paramètres entraînables: 33,793
🧪 Test de prédiction pour cancer
   📊 Forme des données d'entrée: (10, 2)
   📊 Forme des prédictions: (10, 1)
   📊 Échantillon de prédictions:
      Échantillon 1: 0.7234
      Échantillon 2: 0.4567
✅ Diagramme sauvegardé: model_architecture_cancer.png

======================================================================
📊 RAPPORT DE COMPARAISON DES MODÈLES
======================================================================
📋 Tableau de comparaison:
    Dataset  Couches  Paramètres Totaux  Paramètres Entraînables Test Prédiction Visualisation
     cancer        3             33,793                   33,793              ✅            ✅
  chest_xray        4            524,417                  524,417              ✅            ✅
    diabetes        3             45,321                   45,321              ✅            ✅

📈 Statistiques globales:
   🎯 Modèles trouvés: 3
   ✅ Chargements réussis: 3
   📊 Taux de succès: 100.0%

🎉 Test de chargement terminé avec succès !
📁 Fichiers générés dans le répertoire courant
```

### 🎯 **Test d'un modèle spécifique**

```bash
# Tester un modèle spécifique avec un dataset
python test_specific_model.py ../best_models_neuroplast_cancer/model_1.h5 --dataset ../datasets/Cancer.csv

# Tester avec données synthétiques
python test_specific_model.py ../best_models_neuroplast_diabetes/model_1.h5
```

**Exemple de sortie :**
```
🧪 Test Spécifique de Modèle NEUROPLAST-ANN
==================================================
📂 Chargement du modèle: ../best_models_neuroplast_cancer/model_1.h5
✅ Modèle chargé avec succès
📊 Chargement du dataset: ../datasets/Cancer.csv
   📋 Forme du dataset: (569, 31)
   📋 Colonnes: ['mean radius', 'mean texture', ..., 'diagnosis']
   📊 Train: (455, 30), Test: (114, 30)
📈 Évaluation du modèle
   🎯 Accuracy: 0.9649 (96.49%)

   📊 Rapport de classification:
      0: P=0.971, R=0.944, F1=0.957, Support=68
      1: P=0.956, R=0.978, F1=0.967, Support=46
   ✅ Matrice de confusion sauvegardée: confusion_matrix.png
   ✅ Distribution sauvegardée: prediction_distribution.png

🎉 Test terminé avec succès !
📁 Fichiers générés:
   - confusion_matrix.png
   - prediction_distribution.png
```

## 📁 Structure des Fichiers

```
load_model/
├── 📄 model_loader.py           # Programme principal (test tous modèles)
├── 📄 test_specific_model.py    # Test d'un modèle spécifique
├── 📄 requirements.txt         # Dépendances Python
├── 📄 README.md               # Cette documentation
└── 📊 Fichiers générés:
    ├── model_architecture_*.png      # Diagrammes d'architecture
    ├── model_comparison_report.csv   # Rapport de comparaison
    ├── confusion_matrix.png          # Matrice de confusion
    └── prediction_distribution.png   # Distribution des prédictions
```

## 🔧 Fonctionnalités Détaillées

### 📊 **Chargement des Modèles H5**

Le programme lit les fichiers `.h5` créés par NEUROPLAST-ANN et extrait :
- **Architecture** : Couches, tailles, activations
- **Poids et biais** : Paramètres entraînés
- **Métadonnées** : Méthode, optimiseur, dataset

### 🧠 **Conversion TensorFlow**

Conversion automatique des modèles C vers TensorFlow :
- **Mapping des activations** : relu, gelu, sigmoid, tanh, etc.
- **Reconstruction des couches** : Dense layers avec poids corrects
- **Compilation** : Optimiseur Adam, loss appropriée

### 📈 **Évaluation Complète**

- **Métriques** : Accuracy, Precision, Recall, F1-score
- **Visualisations** : Matrices de confusion, distributions
- **Rapports** : CSV avec comparaison multi-modèles

## 🎨 Exemples de Visualisations

### 📊 **Matrice de Confusion**
![Matrice de Confusion](https://via.placeholder.com/400x300/4CAF50/FFFFFF?text=Confusion+Matrix)

### 📈 **Distribution des Prédictions**
![Distribution](https://via.placeholder.com/600x200/2196F3/FFFFFF?text=Prediction+Distribution)

### 🏗️ **Architecture du Modèle**
![Architecture](https://via.placeholder.com/300x400/FF9800/FFFFFF?text=Model+Architecture)

## 🔍 Cas d'Usage

### 🩺 **Validation Médicale**
```bash
# Tester un modèle cancer avec nouvelles données
python test_specific_model.py ../best_models_neuroplast_cancer/model_1.h5 \
    --dataset new_cancer_data.csv
```

### 🖼️ **Images Médicales**
```bash
# Tester un modèle chest X-ray
python test_specific_model.py ../best_models_neuroplast_chest_xray/model_1.h5
```

### 📊 **Comparaison Multi-Modèles**
```bash
# Comparer tous les modèles disponibles
python model_loader.py
# → Génère model_comparison_report.csv
```

## 🐛 Dépannage

### ❌ **Erreur : "Aucun modèle trouvé"**
```bash
# Vérifier que NEUROPLAST-ANN a été exécuté
ls -la ../best_models_neuroplast_*/

# Exécuter NEUROPLAST-ANN pour créer des modèles
cd ..
./neuroplast-ann --config config/cancer_simple.yml --test-all
```

### ❌ **Erreur TensorFlow**
```bash
# Réinstaller TensorFlow
pip uninstall tensorflow
pip install tensorflow>=2.12.0
```

### ❌ **Erreur de format H5**
```bash
# Vérifier la structure du fichier H5
python -c "import h5py; f=h5py.File('model.h5', 'r'); print(list(f.keys()))"
```

## 🔗 Intégration avec NEUROPLAST-ANN

### 🔄 **Workflow Complet**

1. **Entraîner avec NEUROPLAST-ANN** :
   ```bash
   ./neuroplast-ann --config config/cancer_simple.yml --test-all
   ```

2. **Tester avec Python** :
   ```bash
   cd load_model
   python model_loader.py
   ```

3. **Analyser les résultats** :
   - Consulter `model_comparison_report.csv`
   - Examiner les visualisations générées

### 📊 **Formats Supportés**

- **Entrée** : Fichiers `.h5` de NEUROPLAST-ANN
- **Datasets** : CSV, Excel (.xlsx, .xls)
- **Sortie** : PNG (visualisations), CSV (rapports)

## 🎯 Avantages

- ✅ **Interopérabilité** : Pont entre C et Python/TensorFlow
- ✅ **Validation** : Vérification des modèles NEUROPLAST-ANN
- ✅ **Analyse** : Métriques détaillées et visualisations
- ✅ **Facilité** : Interface simple et automatisée
- ✅ **Extensibilité** : Code modulaire et personnalisable

---

**NEUROPLAST-ANN Model Loader v4.3** - Validation Python/TensorFlow pour modèles C 🐍🧠 