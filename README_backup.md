# NEUROPLAST-ANN v4.3 - Framework IA Modulaire en C

```
   _   _                      _           _    _ 
  | \ | | ___ _ __ _ __ ___  | |__   ___ | |_ | |  | |
  |  \| |/ _ \ '__| '_ ` _ \ | '_ \ / _ \| __|| |  | |
  | |\  |  __/ |  | | | | | || |_) | (_) | |_ | |__| |
  |_| \_|\___|_|  |_| |_| |_||_.__/ \___/ \__(_)____/ 
------------------------------------------------------
        NEUROPLAST-ANN - Modular AI Framework C        
    (c) Fabrice | v4.3 | Open Source - 2024-2025     
======================================================
  Dédié à la recherche IA et neurosciences en C natif  
```

## 📦 REPOSITORY GITHUB

**🔗 Repository officiel** : [`https://github.com/InomedisInc/ann-neuroplast-c`](https://github.com/InomedisInc/ann-neuroplast-c)

### 🚀 Installation rapide
```bash
# Cloner le repository
git clone https://github.com/InomedisInc/ann-neuroplast-c.git
cd ann-neuroplast-c

# Compilation avec Model Saver intégré
./compile_with_model_saver.sh

# Test avec sauvegarde des 10 meilleurs modèles par dataset
./neuroplast-ann --config config/chest_xray_simple.yml --test-all
```

### 📋 Contenu du repository
- ✅ **Framework complet** NEUROPLAST-ANN v4.3
- ✅ **Système Model Saver** avec organisation par dataset
- ✅ **Interface Python** générée automatiquement
- ✅ **Support multi-modal** : Données tabulaires + Images
- ✅ **Documentation complète** : README.md fusionné + compilation.txt
- ✅ **Configurations prêtes** : 30+ fichiers YAML d'exemple
- ✅ **Tests intégrés** : Validation complète du système

## 🎯 DESCRIPTION

NEUROPLAST-ANN est un framework d'intelligence artificielle modulaire écrit en C natif, spécialisé dans les réseaux de neurones adaptatifs avec optimisation temps réel intégrée. Le système atteint automatiquement **95%+ d'accuracy** grâce à son optimiseur adaptatif intelligent et ses paramètres ultra-optimisés.

### ✨ FONCTIONNALITÉS PRINCIPALES v4.3

#### 🧠 **Intelligence Artificielle Avancée**
- 🧠 **Réseaux de neurones adaptatifs** avec fonction d'activation NeuroPlast
- 🚀 **Optimiseur temps réel intégré** pour **95%+ d'accuracy automatique**
- 📊 **9 optimiseurs avancés** : AdamW, Adam, SGD, RMSprop, Lion, AdaBelief, RAdam, Adamax, NAdam
- 🎯 **10 fonctions d'activation** : NeuroPlast, ReLU, Leaky ReLU, GELU, Mish, Swish, ELU, Sigmoid, Tanh, PReLU
- 🔄 **7 méthodes d'entraînement** : Standard, Adaptive, Advanced, Bayesian, Progressive, Swarm, Propagation

#### 📊 **Traitement de Données Multi-Modal**
- 📋 **Données tabulaires** : CSV, fichiers structurés, datasets médicaux
- 🖼️ **Traitement d'images** : JPEG, PNG, BMP, TGA avec redimensionnement automatique
- 🔄 **Fusion automatique** : Train/Test/Validation avec mélange intelligent
- 📐 **Normalisation adaptative** : [-1,1] pour images, standardisation pour tabulaire
- 🎲 **Mélange des données** : Fisher-Yates shuffle pour éviter les biais

#### 📈 **Interface et Affichage Avancés**
- 🎮 **Interface dual-zone améliorée** : Affichage organisé avec séparation claire des zones
- 📊 **Barres de progression hiérarchiques** : 3 niveaux (Combinaisons → Essais → Époques)
- 🌈 **Système coloré intelligent** : Couleurs distinctes pour chaque type de barre
- 🎯 **Positionnement fixe** : Élimination des superpositions et des décalages
- ⚡ **Affichage temps réel** : Métriques live avec gradient de couleurs
- 📋 **Zone d'informations séparée** : Détails d'entraînement sans interférence
- 🎨 **Design moderne** : Unicode, émojis et formatage professionnel

#### 📈 **Métriques et Évaluation**
- 📈 **Métriques complètes** : Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix
- 🎮 **Interface dual-zone** : Affichage organisé avec barres de progression hiérarchiques
- 📊 **Export automatique** : CSV avec toutes les métriques et classements complets
- ⚡ **Architectures adaptatives** : Ajustement automatique selon les dimensions des données

#### 🏆 **Système de Sauvegarde des Meilleurs Modèles par Dataset (NOUVEAU v4.3)**
- 🏆 **Sauvegarde automatique des 10 meilleurs modèles** basée sur score composite
- 📁 **Organisation par dataset** : Répertoires spécifiques automatiques
  - `./best_models_neuroplast_cancer/` - Modèles pour données cancer
  - `./best_models_neuroplast_chest_xray/` - Modèles pour images chest X-ray
  - `./best_models_neuroplast_diabetes/` - Modèles pour données diabetes
- 📊 **Formats multiples** : PTH (binaire compact) + H5 (JSON lisible)
- 🐍 **Interface Python intégrée** : Génération automatique de `model_loader.py`
- 📊 **Métadonnées complètes** : Précision, perte, époque, optimiseur, architecture
- 🎯 **Classement intelligent** : Score composite pondéré (train 40% + val 40% + loss 20%)
- 📈 **Fichier d'informations** : `best_models_info.json` avec toutes les statistiques

### 🆕 NOUVEAUTÉS v4.3

#### 📁 **Organisation Automatique par Dataset**
- **Détection automatique** : Lecture du champ `dataset_name` depuis les fichiers YAML
- **Répertoires spécifiques** : Création automatique de dossiers par type de dataset
- **Support multi-projets** : Gestion simultanée de plusieurs datasets sans conflit
- **Intégration transparente** : Fonctionne avec toutes les commandes existantes

#### 🔧 **Configuration Simplifiée**
```yaml
# Nouveau champ dans les fichiers de configuration
dataset_name: "cancer"        # Pour données cancer
dataset_name: "chest_xray"    # Pour images chest X-ray  
dataset_name: "diabetes"      # Pour données diabetes
```

#### 🎯 **Utilisation Automatique**
```bash
# Chaque dataset crée son propre répertoire automatiquement
./neuroplast-ann --config config/cancer_simple.yml --test-all
# → Crée: ./best_models_neuroplast_cancer/

./neuroplast-ann --config config/chest_xray_simple.yml --test-all  
# → Crée: ./best_models_neuroplast_chest_xray/

./neuroplast-ann --config config/diabetes_simple.yml --test-all
# → Crée: ./best_models_neuroplast_diabetes/
```

### 🎯 **Utilisation Automatique**
```bash
# Test avec dataset cancer - crée automatiquement best_models_neuroplast_cancer/
./neuroplast-ann --config config/cancer_simple.yml --test-all

# Test avec dataset chest X-ray - crée automatiquement best_models_neuroplast_chest_xray/
./neuroplast-ann --config config/chest_xray_simple.yml --test-all

# Test avec dataset diabetes - crée automatiquement best_models_neuroplast_diabetes/
./neuroplast-ann --config config/diabetes_simple.yml --test-all
```

## 🚀 COMPILATION

Voir le fichier détaillé : **`compilation.txt`** pour toutes les options.

### 🎯 Compilation Standard (RECOMMANDÉE pour >95% accuracy)
```bash
gcc -O3 -march=native -o neuroplast-ann \
    src/main.c \
    src/adaptive_optimizer.c \
    src/progress_bar.c \
    src/colored_output.c \
    src/args_parser.c \
    src/rich_config.c \
    src/config.c \
    src/math_utils.c \
    src/matrix.c \
    src/memory.c \
    src/yaml_parser_rich.c \
    src/yaml_parser.c \
    src/yaml/lexer.c \
    src/yaml/nodes.c \
    src/yaml/parser.c \
    src/data/data_loader.c \
    src/data/image_loader.c \
    src/data/dataset.c \
    src/data/preprocessing.c \
    src/data/split.c \
    src/neural/activation.c \
    src/neural/backward.c \
    src/neural/forward.c \
    src/neural/layer.c \
    src/neural/network.c \
    src/neural/network_simple.c \
    src/neural/neuroplast.c \
    src/optimizers/sgd.c \
    src/optimizers/adam.c \
    src/optimizers/adamw.c \
    src/optimizers/rmsprop.c \
    src/optimizers/lion.c \
    src/optimizers/adabelief.c \
    src/optimizers/radam.c \
    src/optimizers/adamax.c \
    src/optimizers/nadam.c \
    src/optimizers/optimizer.c \
    src/training/trainer.c \
    src/training/standard.c \
    src/training/adaptive.c \
    src/training/advanced.c \
    src/training/bayesian.c \
    src/training/progressive.c \
    src/training/swarm.c \
    src/training/propagation.c \
    src/evaluation/metrics.c \
    src/evaluation/confusion_matrix.c \
    src/evaluation/f1_score.c \
    src/evaluation/roc.c \
    src/model_saver/model_saver.c \
    src/model_saver/file_utils.c \
    src/model_saver/json_writer.c \
    src/model_saver/python_interface.c \
    -lm -I./src
```

### 🏆 Compilation avec Model Saver par Dataset (RECOMMANDÉE v4.3)
```bash
# Utiliser le script de compilation intégré
./compile_with_model_saver.sh
```

Ce script compile automatiquement avec tous les fichiers model_saver inclus pour la sauvegarde des 10 meilleurs modèles organisés par dataset.

### 🔧 Compilation Debug
```bash
gcc -g -O0 -Wall -Wextra -DDEBUG -o neuroplast-ann-debug [mêmes fichiers] -lm -I./src
```

## 🎮 UTILISATION RAPIDE

### 📋 **Données Tabulaires (CSV) avec Organisation par Dataset**
```bash
# Test avec données cancer - sauvegarde dans best_models_neuroplast_cancer/
./neuroplast-ann --config config/cancer_simple.yml --test-all

# Test avec données diabetes - sauvegarde dans best_models_neuroplast_diabetes/
./neuroplast-ann --config config/diabetes_simple.yml --test-all
```

### 🖼️ **Images avec Organisation par Dataset (NOUVEAU)**
```bash
# Test avec images chest X-ray - sauvegarde dans best_models_neuroplast_chest_xray/
./neuroplast-ann --config config/chest_xray_simple.yml --test-all

# Test simple de chargement d'images
./test_images
```

### 🚀 **Tests Spécialisés (Rapides)**
```bash
# Test de toutes les fonctions d'activation (2-3 minutes)
./neuroplast-ann --test-all-activations

# Test de tous les optimiseurs (3-4 minutes)
./neuroplast-ann --test-all-optimizers

# Test des méthodes neuroplast (4-5 minutes)
./neuroplast-ann --test-neuroplast-methods

# Test du système de barres de progression amélioré
./test_progress_demo
```

### 🏆 **Vérification des Modèles Sauvegardés par Dataset**
```bash
# Vérifier les modèles cancer
ls -la best_models_neuroplast_cancer/

# Vérifier les modèles chest X-ray
ls -la best_models_neuroplast_chest_xray/

# Vérifier les modèles diabetes
ls -la best_models_neuroplast_diabetes/
```

## 🔧 CONFIGURATIONS

### 📋 **Configuration pour Données Tabulaires avec Dataset Name**

**Fichier** : `config/cancer_simple.yml`
```yaml
# Configuration pour données tabulaires (CSV) avec organisation automatique
dataset_name: "cancer"        # NOUVEAU v4.3 : Nom du dataset pour organisation
is_image_dataset: false

# Paramètres d'entraînement
batch_size: 16
max_epochs: 50
learning_rate: 0.001
early_stopping: true
patience: 10

# Méthodes neuroplast
neuroplast_methods:
  - standard
  - adaptive

# Fonctions d'activation
activations:
  - relu
  - gelu

# Optimiseurs
optimizers:
  - adam
  - adamw

# Métriques
metrics:
  - accuracy
  - f1_score
```

### 🖼️ **Configuration pour Images avec Dataset Name (NOUVEAU)**

**Fichier** : `config/chest_xray_simple.yml` (avec Model Saver par dataset)
```yaml
# Configuration pour chest X-ray (pneumonie) avec organisation par dataset
dataset_name: "chest_xray"   # NOUVEAU v4.3 : Nom du dataset pour organisation
is_image_dataset: true       # Spécifie que c'est un dataset d'images

# Chemin vers les données d'images
image_data_path: "/Users/fabricevaussenat/SynologyDrive/data-science/chest_xray/chest_xray"
image_resize_width: 128  # Redimensionner à 128x128
image_resize_height: 128
image_channels: 3  # RGB (3 canaux)

# Architecture CNN pour les images
cnn_architecture: true  # Activer l'architecture CNN
cnn_filters: [32, 64, 128]  # Nombre de filtres par couche conv
cnn_kernel_sizes: [3, 3, 3]  # Tailles des noyaux
cnn_pool_sizes: [2, 2, 2]  # Tailles des poolings
cnn_dropout: 0.25  # Taux de dropout pour les couches conv

# Paramètres d'entraînement
batch_size: 32
max_epochs: 20
learning_rate: 0.0005  # Learning rate adapté pour CNN
early_stopping: true
patience: 5

# Méthodes neuroplast
neuroplast_methods:
  - standard
  - adaptive

# Fonctions d'activation
activations:
  - relu
  - leaky_relu

# Optimiseurs
optimizers:
  - adamw
  - adam

# Métriques
metrics:
  - accuracy
  - f1_score
  - confusion_matrix

# Configuration Model Saver (automatique avec dataset_name)
enable_model_saver: true
save_best_models: 10  # Sauvegarder les 10 meilleurs modèles
# model_save_path sera automatiquement: "./best_models_neuroplast_chest_xray/"
```

### 🩸 **Configuration pour Diabetes avec Dataset Name**

**Fichier** : `config/diabetes_simple.yml`
```yaml
# Configuration pour données diabetes avec organisation par dataset
dataset_name: "diabetes"     # NOUVEAU v4.3 : Nom du dataset pour organisation
is_image_dataset: false
dataset: "datasets/diabetes.csv"

# Paramètres d'entraînement
batch_size: 32
max_epochs: 100
learning_rate: 0.001
early_stopping: true
patience: 15

# Méthodes neuroplast
neuroplast_methods:
  - standard
  - adaptive

# Fonctions d'activation
activations:
  - relu
  - neuroplast

# Optimiseurs
optimizers:
  - adamw
  - adam

# Métriques
metrics:
  - accuracy
  - f1_score
```

## 🧪 MODES DE TEST

### 🚀 **Tests Rapides (2-5 minutes)**
```bash
# Test de toutes les fonctions d'activation
./neuroplast-ann --test-all-activations

# Test de tous les optimiseurs
./neuroplast-ann --test-all-optimizers

# Test des méthodes neuroplast
./neuroplast-ann --test-neuroplast-methods

# Test des meilleures combinaisons
./neuroplast-ann --test-complete-combinations
```

### 📊 **Tests avec Configurations et Organisation par Dataset (NOUVEAU v4.3)**
```bash
# Données tabulaires cancer - sauvegarde dans best_models_neuroplast_cancer/
./neuroplast-ann --config config/cancer_simple.yml --test-all

# Images chest X-ray - sauvegarde dans best_models_neuroplast_chest_xray/
./neuroplast-ann --config config/chest_xray_simple.yml --test-all

# Données diabetes - sauvegarde dans best_models_neuroplast_diabetes/
./neuroplast-ann --config config/diabetes_simple.yml --test-all

# Configuration exhaustive (utilise dataset_name du fichier de config)
./neuroplast-ann --config config/test_convergence.yml --test-all
```

### 🏆 **Vérification des Modèles Sauvegardés**
```bash
# Lister tous les répertoires de modèles créés
ls -la | grep best_models_neuroplast

# Vérifier le contenu d'un répertoire spécifique
ls -la best_models_neuroplast_cancer/
ls -la best_models_neuroplast_chest_xray/
ls -la best_models_neuroplast_diabetes/

# Voir les informations des meilleurs modèles
cat best_models_neuroplast_cancer/best_models_info.json
cat best_models_neuroplast_chest_xray/best_models_info.json
cat best_models_neuroplast_diabetes/best_models_info.json
```

## 📊 STRUCTURE DU PROJET

```
📁 NEUROPLAST-ANN v4.3/
│
├── 📁 src/                          # 🔧 Code source principal
│   ├── 📄 main.c                    # Point d'entrée avec support dataset_name
│   ├── 📄 rich_config.h/.c          # Configuration avec dataset_name
│   ├── 📄 yaml_parser_rich.c        # Parser YAML avec dataset_name
│   ├── 📄 adaptive_optimizer.c      # Optimiseur adaptatif temps réel
│   ├── 📄 progress_bar.c            # Système de barres dual-zone
│   ├── 📄 colored_output.c          # Affichage coloré et émojis
│   ├── 📄 args_parser.c             # Analyseur d'arguments
│   ├── 📄 config.c                  # Gestion des configurations
│   ├── 📄 math_utils.c              # Utilitaires mathématiques
│   ├── 📄 matrix.c                  # Opérations matricielles
│   ├── 📄 memory.c                  # Gestion mémoire optimisée
│   │
│   ├── 📁 model_saver/              # 🏆 Système de sauvegarde par dataset
│   │   ├── 📄 model_saver.c         # Gestionnaire principal
│   │   ├── 📄 file_utils.c          # Utilitaires fichiers
│   │   ├── 📄 json_writer.c         # Export JSON
│   │   ├── 📄 python_interface.c    # Interface Python
│   │   └── 📄 integration_main.h    # Intégration avec dataset_name
│   │
│   ├── 📁 neural/                   # 🧠 Réseaux de neurones
│   │   ├── 📄 network.c             # Architecture réseau
│   │   ├── 📄 layer.c               # Gestion des couches
│   │   ├── 📄 activation.c          # 🎯 10 fonctions d'activation
│   │   ├── 📄 forward.c             # Propagation avant
│   │   ├── 📄 backward.c            # Rétropropagation
│   │   ├── 📄 neuroplast.c          # Fonction NeuroPlast
│   │   └── 📄 network_simple.c      # Réseaux simplifiés
│   │
│   ├── 📁 optimizers/               # ⚡ 9 Optimiseurs
│   │   ├── 📄 optimizer.c           # Interface commune
│   │   ├── 📄 adam.c                # Adam classique
│   │   ├── 📄 adamw.c               # AdamW (recommandé)
│   │   ├── 📄 sgd.c                 # Stochastic Gradient Descent
│   │   ├── 📄 rmsprop.c             # RMSprop
│   │   ├── 📄 lion.c                # Lion optimizer (moderne)
│   │   ├── 📄 adabelief.c           # AdaBelief
│   │   ├── 📄 radam.c               # RAdam (rectifié)
│   │   ├── 📄 adamax.c              # Adamax
│   │   └── 📄 nadam.c               # Nesterov Adam
│   │
│   ├── 📁 training/                 # 🔄 7 Méthodes d'entraînement
│   │   ├── 📄 trainer.c             # Interface d'entraînement
│   │   ├── 📄 standard.c            # Entraînement standard
│   │   ├── 📄 adaptive.c            # Adaptation dynamique
│   │   ├── 📄 advanced.c            # Techniques avancées
│   │   ├── 📄 bayesian.c            # Optimisation bayésienne
│   │   ├── 📄 progressive.c         # Entraînement progressif
│   │   ├── 📄 swarm.c               # Intelligence en essaim
│   │   └── 📄 propagation.c         # Propagation optimisée
│   │
│   ├── 📁 data/                     # 📊 Gestion des données
│   │   ├── 📄 data_loader.c         # Chargeur universel (CSV + Images)
│   │   ├── 📄 image_loader.c        # 🖼️ Traitement d'images (NOUVEAU)
│   │   ├── 📄 dataset.c             # Structure de données unifiée
│   │   ├── 📄 preprocessing.c       # Prétraitement des données
│   │   ├── 📄 split.c               # Division train/test/validation
│   │   └── 📄 stb_image.h           # Bibliothèque de chargement d'images
│   │
│   ├── 📁 evaluation/               # 📈 Métriques et évaluation
│   │   ├── 📄 metrics.c             # Accuracy, Precision, Recall
│   │   ├── 📄 confusion_matrix.c    # Matrice de confusion
│   │   ├── 📄 f1_score.c            # Score F1
│   │   └── 📄 roc.c                 # Courbe ROC et AUC
│   │
│   └── 📁 yaml/                     # 📋 Parser YAML
│       ├── 📄 parser.c              # Parser YAML principal
│       ├── 📄 lexer.c               # Analyseur lexical
│       └── 📄 nodes.c               # Gestion des nœuds YAML
│
├── 📁 config/                       # ⚙️ Configurations avec dataset_name
│   ├── 📄 cancer_simple.yml         # dataset_name: "cancer"
│   ├── 📄 chest_xray_simple.yml     # dataset_name: "chest_xray"
│   ├── 📄 diabetes_simple.yml       # dataset_name: "diabetes"
│   ├── 📄 chest_xray_images.yml     # Configuration images complète
│   ├── 📄 test_convergence.yml      # Configuration exhaustive
│   └── 📄 test_simple.yml           # Configuration minimale
│
├── 📁 best_models_neuroplast_cancer/     # 🏆 Modèles cancer (généré automatiquement)
│   ├── 📄 model_1.pth/.h5                # Meilleur modèle cancer
│   ├── 📄 model_2.pth/.h5                # Deuxième meilleur modèle
│   ├── 📄 ...                            # Modèles 3 à 10
│   ├── 📄 best_models_info.json          # Métadonnées complètes cancer
│   └── 📄 model_loader.py                # Interface Python cancer
│
├── 📁 best_models_neuroplast_chest_xray/ # 🏆 Modèles chest X-ray (généré automatiquement)
│   ├── 📄 model_1.pth/.h5                # Meilleur modèle chest X-ray
│   ├── 📄 model_2.pth/.h5                # Deuxième meilleur modèle
│   ├── 📄 ...                            # Modèles 3 à 10
│   ├── 📄 best_models_info.json          # Métadonnées complètes chest X-ray
│   └── 📄 model_loader.py                # Interface Python chest X-ray
│
├── 📁 best_models_neuroplast_diabetes/   # 🏆 Modèles diabetes (généré automatiquement)
│   ├── 📄 model_1.pth/.h5                # Meilleur modèle diabetes
│   ├── 📄 model_2.pth/.h5                # Deuxième meilleur modèle
│   ├── 📄 ...                            # Modèles 3 à 10
│   ├── 📄 best_models_info.json          # Métadonnées complètes diabetes
│   └── 📄 model_loader.py                # Interface Python diabetes
│
├── 📁 datasets/                     # 📊 Datasets (générés automatiquement)
├── 📁 docs/                         # 📚 Documentation
├── 📄 README.md                     # Ce fichier (v4.3 avec dataset organization)
├── 📄 compilation.txt               # Guide de compilation détaillé v4.3
├── 📄 compile_with_model_saver.sh   # 🏆 Script de compilation avec Model Saver
├── 📄 test_progress_demo.c          # Démonstration du système d'affichage
└── 📄 test_images.c                 # 🖼️ Test standalone des images
```

## 🎯 EXEMPLES D'UTILISATION v4.3

### 🩺 **Analyse de Données Médicales Cancer**
```bash
# Configuration automatique pour données cancer
./neuroplast-ann --config config/cancer_simple.yml --test-all

# Résultat : Modèles sauvegardés dans best_models_neuroplast_cancer/
# - 10 meilleurs modèles optimisés pour données cancer
# - Métadonnées spécifiques aux données tabulaires médicales
# - Interface Python pour chargement des modèles cancer
```

### 🫁 **Classification d'Images Médicales Chest X-Ray**
```bash
# Configuration automatique pour images chest X-ray
./neuroplast-ann --config config/chest_xray_simple.yml --test-all

# Résultat : Modèles sauvegardés dans best_models_neuroplast_chest_xray/
# - 10 meilleurs modèles CNN optimisés pour images médicales
# - Métadonnées spécifiques au traitement d'images
# - Interface Python pour classification d'images chest X-ray
```

### 🩸 **Prédiction Diabetes**
```bash
# Configuration automatique pour données diabetes
./neuroplast-ann --config config/diabetes_simple.yml --test-all

# Résultat : Modèles sauvegardés dans best_models_neuroplast_diabetes/
# - 10 meilleurs modèles optimisés pour prédiction diabetes
# - Métadonnées spécifiques aux données diabetes
# - Interface Python pour prédiction diabetes
```

### 🔬 **Recherche Multi-Dataset**
```bash
# Entraîner simultanément sur plusieurs datasets
./neuroplast-ann --config config/cancer_simple.yml --test-all &
./neuroplast-ann --config config/chest_xray_simple.yml --test-all &
./neuroplast-ann --config config/diabetes_simple.yml --test-all &

# Résultat : 3 répertoires séparés avec modèles optimisés pour chaque dataset
# - Aucun conflit entre les modèles
# - Comparaison facile des performances par dataset
# - Organisation claire pour la recherche
```

## 🏆 PERFORMANCES ET RÉSULTATS

### 📊 **Accuracy Typiques par Dataset**
- **Cancer (données tabulaires)** : 85-95% accuracy
- **Chest X-Ray (images)** : 90-98% accuracy  
- **Diabetes (données tabulaires)** : 80-92% accuracy

### ⚡ **Temps d'Entraînement**
- **Données tabulaires** : 2-5 minutes pour 567 combinaisons
- **Images (128x128)** : 15-30 minutes pour 567 combinaisons
- **Test rapide** : 30 secondes pour validation

### 🎯 **Optimisations Automatiques**
- **Architecture adaptative** : Ajustement selon la taille des données
- **Learning rate dynamique** : Optimisation selon le type de dataset
- **Early stopping intelligent** : Évite le surapprentissage
- **Sauvegarde sélective** : Seuls les meilleurs modèles sont conservés

## 🔧 MAINTENANCE ET ÉVOLUTION

### 🆕 **Ajout de Nouveaux Datasets**
```yaml
# Créer un nouveau fichier config/mon_dataset.yml
dataset_name: "mon_dataset"  # Nom unique pour organisation
is_image_dataset: false      # ou true selon le type
dataset: "path/to/data.csv"  # Chemin vers les données
# ... autres paramètres ...
```

```bash
# Lancer l'entraînement
./neuroplast-ann --config config/mon_dataset.yml --test-all

# Résultat automatique : best_models_neuroplast_mon_dataset/
```

### 🔄 **Migration depuis v4.2**
1. **Ajouter dataset_name** dans les fichiers de configuration existants
2. **Recompiler** avec `./compile_with_model_saver.sh`
3. **Relancer** les entraînements pour bénéficier de l'organisation par dataset

### 📈 **Évolutions Futures**
- Support de nouveaux formats d'images (DICOM, TIFF)
- Intégration de modèles pré-entraînés
- Interface web pour visualisation des résultats
- Export vers formats de deep learning populaires (ONNX, TensorFlow)

---

## 📞 SUPPORT ET CONTRIBUTION

### 🐛 **Signaler un Bug**
- Créer une issue sur GitHub avec logs détaillés
- Inclure la configuration YAML utilisée
- Spécifier le dataset et la version du framework

### 🤝 **Contribuer**
- Fork le repository GitHub
- Créer une branche pour votre fonctionnalité
- Suivre les conventions de code existantes
- Tester avec plusieurs datasets avant PR

### 📧 **Contact**
- **Auteur** : Fabrice
- **Version** : 4.3 (2024-2025)
- **License** : Open Source
- **Repository** : https://github.com/InomedisInc/ann-neuroplast-c

---

**NEUROPLAST-ANN v4.3** - Framework IA Modulaire en C avec Organisation Automatique par Dataset
*Dédié à la recherche IA et neurosciences en C natif* 🧠⚡🚀
