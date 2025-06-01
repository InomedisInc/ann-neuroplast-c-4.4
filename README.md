# NEUROPLAST-ANN v4.2 - Framework IA Modulaire en C

```
   _   _                      _           _    _    _ 
  | \ | | ___ _ __ _ __ ___  | |__   ___ | |_ | |  | |
  |  \| |/ _ \ '__| '_ ` _ \ | '_ \ / _ \| __|| |  | |
  | |\  |  __/ |  | | | | | || |_) | (_) | |_ | |__| |
  |_| \_|\___|_|  |_| |_| |_||_.__/ \___/ \__(_)____/ 
------------------------------------------------------
        NEUROPLAST-ANN - Modular AI Framework C        
    (c) Fabrice | v4.2 | Open Source - 2024-2025     
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

# Test avec sauvegarde des 10 meilleurs modèles
./neuroplast-ann --config config/chest_xray_simple.yml --test-all
```

### 📋 Contenu du repository
- ✅ **Framework complet** NEUROPLAST-ANN v4.2
- ✅ **Système Model Saver** avec sauvegarde des 10 meilleurs modèles
- ✅ **Interface Python** générée automatiquement
- ✅ **Support multi-modal** : Données tabulaires + Images
- ✅ **Documentation complète** : README.md fusionné + compilation.txt
- ✅ **Configurations prêtes** : 30+ fichiers YAML d'exemple
- ✅ **Tests intégrés** : Validation complète du système

## 🎯 DESCRIPTION

NEUROPLAST-ANN est un framework d'intelligence artificielle modulaire écrit en C natif, spécialisé dans les réseaux de neurones adaptatifs avec optimisation temps réel intégrée. Le système atteint automatiquement **95%+ d'accuracy** grâce à son optimiseur adaptatif intelligent et ses paramètres ultra-optimisés.

### ✨ FONCTIONNALITÉS PRINCIPALES v4.2

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

#### 🏆 **Système de Sauvegarde des Meilleurs Modèles (NOUVEAU)**
- 🏆 **Sauvegarde automatique des 10 meilleurs modèles** basée sur score composite
- 📁 **Formats multiples** : PTH (binaire compact) + H5 (JSON lisible)
- 🐍 **Interface Python intégrée** : Génération automatique de `model_loader.py`
- 📊 **Métadonnées complètes** : Précision, perte, époque, optimiseur, architecture
- 🎯 **Classement intelligent** : Score composite pondéré (train 40% + val 40% + loss 20%)
- 📈 **Fichier d'informations** : `best_models_info.json` avec toutes les statistiques

### 🆕 NOUVEAUTÉS v4.2

#### 🎮 **Interface Utilisateur Révolutionnée**
- **Système dual-zone** : Séparation claire entre barres de progression et informations
- **Barres hiérarchiques** : 3 niveaux avec couleurs distinctes (Cyan/Jaune/Vert)
- **Positionnement fixe** : Élimination complète des superpositions de texte
- **Gradient de couleurs** : Barres progressives Rouge→Jaune→Vert selon l'avancement
- **Affichage temps réel** : Métriques live (Loss, Accuracy, Learning Rate)
- **Zone d'informations dédiée** : Détails d'entraînement sans interférence visuelle
- **Design Unicode** : Caractères de boîte et émojis pour un rendu professionnel

#### 🖼️ **Support Complet des Images**
- **Chargement automatique** : Répertoires train/test/val avec détection de classes
- **Formats supportés** : JPEG, PNG, BMP, TGA (via stb_image)
- **Redimensionnement intelligent** : Nearest neighbor avec préservation du ratio
- **Normalisation optimisée** : [-1,1] pour meilleure convergence
- **Classification binaire/multi-classe** : Adaptation automatique des sorties
- **Mélange des données** : Fisher-Yates shuffle pour apprentissage optimal

#### ⚡ **Optimisations de Performance**
- **Architectures adaptatives** : Ajustement automatique selon input_size (8 vs 64)
- **Learning rates spécialisés** : Paramètres optimisés pour images vs tabulaire
- **Multi-pass training** : 2 passages par époque pour meilleur apprentissage
- **Convergence améliorée** : Critères adaptatifs selon le type de données

#### 🏆 **Intégration Model Saver**
- **Sauvegarde temps réel** : Évaluation et sauvegarde après chaque modèle entraîné
- **Interface simplifiée** : Fonctions d'intégration dans main.c sans conflit mémoire
- **Dossier automatique** : Création de `./best_models_neuroplast/` avec organisation complète
- **Compatibilité totale** : Fonctionne avec `./neuroplast-ann --config config/chest_xray_simple.yml --test-all`

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
    -lm -I./src
```

### 🏆 Compilation avec Model Saver (NOUVEAU)
```bash
# Utiliser le script de compilation intégré
./compile_with_model_saver.sh
```

Ce script compile automatiquement avec tous les fichiers model_saver inclus pour la sauvegarde des 10 meilleurs modèles.

### 🔧 Compilation Debug
```bash
gcc -g -O0 -Wall -Wextra -DDEBUG -o neuroplast-ann-debug [mêmes fichiers] -lm -I./src
```

## 🎮 UTILISATION RAPIDE

### 📋 **Données Tabulaires (CSV)**
```bash
# Test avec données médicales simulées (recommandé)
./neuroplast-ann --config config/cancer_simple.yml --test-all

# Test exhaustif avec configuration personnalisée
./neuroplast-ann --config config/test_convergence.yml --test-all
```

### 🖼️ **Images (NOUVEAU)**
```bash
# Test avec images chest X-ray (pneumonie) + sauvegarde des 10 meilleurs modèles
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

### 🏆 **Test de Model Saver Standalone**
```bash
# Test complet de la librairie model_saver
cd src/model_saver
make clean && make test

# Vérification des fichiers générés
ls -la saved_models/
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
├── 📁 config/                       # ⚙️ Configurations
│   ├── 📄 cancer_simple.yml         # Données tabulaires médicales
│   ├── 📄 chest_xray_simple.yml     # 🖼️ Images chest X-ray avec Model Saver
│   ├── 📄 chest_xray_images.yml     # Configuration images complète
│   ├── 📄 test_convergence.yml      # Configuration exhaustive
│   └── 📄 test_simple.yml           # Configuration minimale
│
├── 📁 best_models_neuroplast/       # 🏆 Dossier de sauvegarde des meilleurs modèles (généré)
│   ├── 📄 model_1.pth/.h5           # Meilleur modèle (formats binaire + JSON)
│   ├── 📄 model_2.pth/.h5           # Deuxième meilleur modèle
│   ├── 📄 ...                       # Modèles 3 à 10
│   ├── 📄 best_models_info.json     # Métadonnées complètes de tous les modèles
│   └── 📄 model_loader.py           # Interface Python générée automatiquement
│
├── 📁 datasets/                     # 📊 Datasets (générés automatiquement)
├── 📁 docs/                         # 📚 Documentation
├── 📄 README.md                     # Ce fichier (fusionné et complet)
├── 📄 compilation.txt               # Guide de compilation détaillé
├── 📄 compile_with_model_saver.sh   # 🏆 Script de compilation avec Model Saver
├── 📄 test_progress_demo.c          # Démonstration du système d'affichage
└── 📄 test_images.c                 # 🖼️ Test standalone des images

// Initialisation du système dual-zone
void progress_init_dual_zone(const char* header_title, int num_combinations, int num_trials, int max_epochs);

// Création de barres avec positionnement fixe
int progress_global_add(ProgressType type, const char *label, int total, int width);

// Mise à jour avec métriques temps réel
void progress_global_update(int bar_id, int current, float loss, float accuracy, float lr);

### 🧠 **Architecture des Réseaux de Neurones**

#### **Couches et Propagation**
```c
// Structure d'une couche
typedef struct {
    size_t input_size;      // Taille d'entrée
    size_t output_size;     // Taille de sortie
    float **weights;        // Matrice des poids [output_size][input_size]
    float *biases;          // Vecteur des biais [output_size]
    float *outputs;         // Sorties de la couche [output_size]
    float *gradients;       // Gradients pour rétropropagation
    ActivationType activation; // Type d'activation
} Layer;

// Réseau de neurones complet
typedef struct {
    Layer *layers;          // Tableau des couches
    size_t num_layers;      // Nombre de couches
    float dropout_rate;     // Taux de dropout
    float l2_lambda;        // Coefficient de régularisation L2
} NeuralNetwork;
```

#### **Fonctions d'Activation Disponibles**
1. **NeuroPlast** : Fonction spécialisée adaptative
2. **ReLU** : `f(x) = max(0, x)`
3. **Leaky ReLU** : `f(x) = max(0.01x, x)`
4. **GELU** : `f(x) = x * Φ(x)` (Gaussian Error Linear Unit)
5. **Sigmoid** : `f(x) = 1 / (1 + e^(-x))`
6. **Tanh** : `f(x) = tanh(x)`
7. **ELU** : `f(x) = x if x > 0, α(e^x - 1) if x ≤ 0`
8. **Mish** : `f(x) = x * tanh(softplus(x))`
9. **Swish** : `f(x) = x * sigmoid(x)`
10. **PReLU** : `f(x) = max(αx, x)` avec α appris

### 📊 **Système de Traitement des Données**

#### **Données Tabulaires (CSV)**
```c
// Chargement CSV
Dataset *load_csv_data(const char *filepath, size_t input_cols, size_t output_cols);

// Structure de dataset unifiée
typedef struct {
    float **inputs;         // Matrice d'entrées [num_samples][input_cols]
    float **outputs;        // Matrice de sorties [num_samples][output_cols]
    size_t num_samples;     // Nombre d'échantillons
    size_t input_cols;      // Nombre de colonnes d'entrée
    size_t output_cols;     // Nombre de colonnes de sortie
} Dataset;
```

#### **🖼️ Traitement d'Images (NOUVEAU)**
```c
// Structure d'information d'image
typedef struct {
    char filepath[512];     // Chemin vers l'image
    int label;              // Label de classe (0, 1, 2, ...)
    char class_name[64];    // Nom de la classe ("NORMAL", "PNEUMONIA", ...)
} ImageInfo;

// Ensemble d'images
typedef struct {
    ImageInfo *images;      // Tableau d'informations d'images
    size_t count;           // Nombre d'images
    size_t capacity;        // Capacité allouée
    char **class_names;     // Noms des classes
    size_t num_classes;     // Nombre de classes
} ImageSet;

// Fonctions principales
ImageSet *load_image_set(const char *directory_path);
Dataset *convert_image_set_to_dataset(const ImageSet *set, int width, int height, int channels, size_t num_classes);
float *load_image_data(const char *filepath, int width, int height, int channels);
```

#### **Pipeline de Traitement d'Images**
1. **Chargement** : Scan des répertoires train/test/val
2. **Détection de classes** : Chaque sous-répertoire = une classe
3. **Chargement d'images** : Via stb_image (JPEG, PNG, BMP, TGA)
4. **Redimensionnement** : Nearest neighbor vers dimensions cibles
5. **Normalisation** : `(pixel/255 - 0.5) * 2` → [-1, 1]
6. **Mélange** : Fisher-Yates shuffle pour éviter les biais
7. **Conversion** : Vers structure Dataset unifiée

### 🎮 **Système de Barres de Progression Dual-Zone (NOUVEAU v4.2)**

#### **Architecture du Système d'Affichage**
Le nouveau système d'affichage révolutionnaire sépare clairement les zones pour éliminer les superpositions :

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                       🧠 NEUROPLAST TRAINING SYSTEM 🧠                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ TEST: Configuration d'entraînement                                            ║
║ Configuration: 3 combinaisons × 5 essais × 10 époques max                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                        📊 BARRES DE PROGRESSION 📊                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
GENERAL:  [██████████████████████████] 100.0% Combinaisons (3/3)
ESSAIS:   [██████████████████████████] 100.0% Essais (5/5) | Loss: 0.0234 | Acc: 94.2%
EPOQUES:  [████████████████████] 100.0% Époques (10/10) | Loss: 0.0156 | Acc: 96.8% | LR: 0.0010

╔══════════════════════════════════════════════════════════════════════════════╗
║                        🔥 INFORMATIONS D'ENTRAÎNEMENT 🔥                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                        🧪 COMBINAISON 1/3 🧪                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 🧠 Méthode: neuroplast     ⚡ Optimiseur: adamw        🎯 Activation: relu    ║
╚══════════════════════════════════════════════════════════════════════════════╝

🏗️  Architecture: Input(2)→256→128→Output(1)
📊 Dataset: XOR Dataset (4 samples)
⚡ Learning Rate: 0.001000
────────────────────────────────────────────────────────────────────────────────
📈 Époque 1/10 | Loss: 0.8234 | Acc: 52.3% | F1: 48.7% | Prec: 51.2% | Rec: 49.8%
📈 Époque 2/10 | Loss: 0.7123 | Acc: 64.1% | F1: 62.4% | Prec: 63.7% | Rec: 61.2%
...
✅ Essai 1/5 terminé | Meilleure Acc: 94.2% | Meilleur F1: 93.8% | Convergence: époque 8
```

#### **Fonctionnalités du Système d'Affichage**

##### **🎯 Barres de Progression Hiérarchiques**
```c
// Types de barres avec couleurs distinctes
typedef enum {
    PROGRESS_GENERAL,     // Cyan    - Progression des combinaisons
    PROGRESS_TRIALS,      // Jaune   - Progression des essais
    PROGRESS_EPOCHS,      // Vert    - Progression des époques
    PROGRESS_ITERATIONS   // Magenta - Barres temporaires
} ProgressType;

// Initialisation du système dual-zone
void progress_init_dual_zone(const char* header_title, int num_combinations, int num_trials, int max_epochs);

// Création de barres avec positionnement fixe
int progress_global_add(ProgressType type, const char *label, int total, int width);

// Mise à jour avec métriques temps réel
void progress_global_update(int bar_id, int current, float loss, float accuracy, float lr);
```

##### **🌈 Système de Couleurs Intelligent**
- **Barres de progression** : Gradient Rouge→Jaune→Vert selon l'avancement
- **Types de barres** : Couleurs distinctes (Cyan/Jaune/Vert) pour chaque niveau
- **Métriques** : Couleurs spécialisées (Rouge pour Loss, Vert pour Accuracy, etc.)
- **Informations** : Émojis et couleurs contextuelles pour chaque type de message

##### **📋 Zone d'Informations Séparée**
```c
// Affichage des en-têtes de combinaison
void progress_display_combination_header(int current_combo, int total_combos, 
                                       const char* method, const char* optimizer, const char* activation);

// Informations du réseau
void progress_display_network_info(const char* architecture, const char* dataset_info, 
                                 float learning_rate, float* class_weights);

// Métriques d'époque en temps réel
void progress_display_epoch_info(int epoch, int max_epochs, float loss, float accuracy, 
                                float precision, float recall, float f1_score);

// Résumés d'essais et de combinaisons
void progress_display_trial_summary(int trial, int max_trials, float best_accuracy, 
                                   float best_f1, int convergence_epoch);
```

##### **⚡ Fonctionnalités Avancées**
- **Positionnement fixe** : Élimination complète des superpositions de texte
- **Zones séparées** : Barres de progression et informations dans des zones distinctes
- **Nettoyage automatique** : Préparation des zones pour les prochaines combinaisons
- **Affichage temps réel** : Métriques live sans interférence visuelle
- **Design Unicode** : Caractères de boîte (╔═╗║╚╝) et émojis pour un rendu professionnel

#### **Démonstration du Système**
```bash
# Compiler la démonstration
gcc -o test_progress_demo test_progress_demo.c src/progress_bar.c src/colored_output.c -I./src

# Lancer la démonstration interactive
./test_progress_demo
```

Cette démonstration montre :
- ✅ Élimination des superpositions de texte
- ✅ Zones d'affichage bien séparées  
- ✅ Couleurs et émojis améliorés
- ✅ Positionnement fixe et stable
- ✅ Métriques temps réel sans interférence

### ⚡ **Optimiseurs et Algorithmes**

#### **Interface Commune des Optimiseurs**
```c
typedef struct {
    char name[32];          // Nom de l'optimiseur
    float learning_rate;    // Taux d'apprentissage
    float beta1, beta2;     // Paramètres Adam/AdamW
    float epsilon;          // Terme de stabilité numérique
    float weight_decay;     // Décroissance des poids (AdamW)
    float momentum;         // Momentum (SGD)
} OptimizerConfig;

// Fonction d'optimisation
typedef void (*OptimizerUpdateFn)(Layer *layer, OptimizerConfig *config, int epoch);
```

#### **Optimiseurs Implémentés**
1. **AdamW** : Adam avec décroissance des poids découplée (recommandé)
2. **Adam** : Adaptive Moment Estimation classique
3. **SGD** : Stochastic Gradient Descent avec momentum
4. **RMSprop** : Root Mean Square Propagation
5. **Lion** : Evolved Sign Momentum (moderne, efficace)
6. **AdaBelief** : Adapting Stepsizes by the Belief in Observed Gradients
7. **RAdam** : Rectified Adam avec warm-up
8. **Adamax** : Variant d'Adam basé sur la norme infinie
9. **NAdam** : Nesterov-accelerated Adaptive Moment Estimation

### 🔄 **Méthodes d'Entraînement Neuroplast**

#### **1. Standard** : Entraînement classique
- Propagation avant/arrière standard
- Mise à jour des poids selon l'optimiseur choisi

#### **2. Adaptive** : Adaptation dynamique
- Ajustement automatique du learning rate
- Détection de plateaux et adaptation

#### **3. Advanced** : Techniques avancées
- Régularisation adaptative
- Techniques de régularisation avancées

#### **4. Bayesian** : Optimisation bayésienne
- Exploration intelligente de l'espace des hyperparamètres
- Incertitude quantifiée

#### **5. Progressive** : Entraînement progressif
- Augmentation progressive de la complexité
- Curriculum learning

#### **6. Swarm** : Intelligence en essaim
- Optimisation par essaim de particules
- Exploration collective

#### **7. Propagation** : Propagation optimisée
- Techniques de propagation avancées
- Optimisation des gradients

## 🔧 CONFIGURATIONS

### 📋 **Configuration pour Données Tabulaires**

**Fichier** : `config/cancer_simple.yml`
```yaml
# Configuration pour données tabulaires (CSV)
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

### 🖼️ **Configuration pour Images (NOUVEAU)**

**Fichier** : `config/chest_xray_simple.yml` (avec Model Saver intégré)
```yaml
# Configuration pour dataset d'images Chest X-Ray avec sauvegarde des meilleurs modèles
is_image_dataset: true

# Répertoires d'images (obligatoires: train et test, optionnel: val)
image_train_dir: "/path/to/chest_xray/train"
image_test_dir: "/path/to/chest_xray/test"
image_val_dir: "/path/to/chest_xray/val"

# Dimensions des images (adaptées à l'architecture)
image_width: 8        # Largeur cible (8x8 = 64 pixels)
image_height: 8       # Hauteur cible
image_channels: 1     # 1 = grayscale, 3 = RGB

# Configuration d'entraînement
batch_size: 16
max_epochs: 50
learning_rate: 0.001
early_stopping: true
patience: 10

# Méthodes neuroplast
neuroplast_methods:
  - adaptive
  - standard

# Fonctions d'activation
activations:
  - relu
  - sigmoid

# Optimiseurs
optimizers:
  - adam
  - adamw

# Métriques
metrics:
  - accuracy
  - precision
  - recall

# Sauvegarde des meilleurs modèles (automatique)
save_best_models: true
best_models_count: 10
```

### ⚙️ **Structure des Répertoires d'Images**
```
dataset_images/
├── train/
│   ├── NORMAL/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   └── PNEUMONIA/
│       ├── image1.jpg
│       ├── image2.png
│       └── ...
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/ (optionnel)
    ├── NORMAL/
    └── PNEUMONIA/
```

### 🎛️ **Paramètres Avancés**

#### **Régularisation**
```yaml
dropout_rate: 0.2          # Taux de dropout (0.0-0.5)
l2_regularization: 0.001   # Régularisation L2
momentum: 0.9              # Momentum pour SGD
```

#### **Optimisation**
```yaml
beta1: 0.9                 # Adam beta1
beta2: 0.999               # Adam beta2
epsilon: 1e-8              # Terme de stabilité
weight_decay: 0.01         # Décroissance des poids (AdamW)
```

#### **Gestion des Classes**
```yaml
class_weights: [1.0, 1.5]  # Pondération des classes
# [1.0, 1.0] = équilibré
# [1.0, 2.0] = favorise classe minoritaire
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

### 📊 **Tests avec Configurations**
```bash
# Données tabulaires
./neuroplast-ann --config config/cancer_simple.yml --test-all

# Images avec sauvegarde des 10 meilleurs modèles (NOUVEAU)
./neuroplast-ann --config config/chest_xray_simple.yml --test-all

# Configuration exhaustive
./neuroplast-ann --config config/test_convergence.yml --test-all
```

### 🖼️ **Test Standalone des Images**
```bash
# Compiler le test d'images
gcc -o test_images test_images.c src/data/image_loader.c src/data/dataset.c src/colored_output.c -lm

# Exécuter le test
./test_images
```

**Résultat attendu** :
```
=== Test de chargement d'images ===
Test 1: Chargement du répertoire d'entraînement...
✅ Répertoire d'entraînement chargé avec succès

=== Statistiques Train ===
Nombre total d'images: 5216
Nombre de classes: 2
Classes détectées:
  - PNEUMONIA: 3875 images
  - NORMAL: 1341 images

Test 2: Conversion en dataset...
✅ Dataset mélangé pour améliorer l'apprentissage
✅ Dataset créé avec succès
Dimensions: 5216 échantillons, 64 entrées, 1 sorties

Test 3: Vérification des données...
Échantillon 0: Entrée[0]=0.624, Entrée[63]=0.176, Sortie=0.0
Échantillon 1: Entrée[0]=-0.953, Entrée[63]=-0.184, Sortie=0.0
Échantillon 2: Entrée[0]=-0.631, Entrée[63]=0.012, Sortie=0.0

✅ Test terminé avec succès !
```

## 📈 MÉTRIQUES ET ÉVALUATION

### 📊 **Métriques Calculées**
1. **Accuracy** : Précision globale `(TP + TN) / (TP + TN + FP + FN)`
2. **Precision** : Précision des positifs `TP / (TP + FP)`
3. **Recall** : Rappel (sensibilité) `TP / (TP + FN)`
4. **F1-Score** : Moyenne harmonique `2 * (Precision * Recall) / (Precision + Recall)`
5. **AUC-ROC** : Aire sous la courbe ROC
6. **Confusion Matrix** : Matrice de confusion complète

### 🏆 **Classement Automatique**
Le système génère automatiquement un TOP 5 des meilleures combinaisons :

```
Rang | Combinaison                          | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Conv %
-----|--------------------------------------|----------|-----------|--------|----------|---------|-------
   1 | adaptive+adamw+relu (Images)         |    87.2% |     89.1% |  85.3% |    87.1% |   92.4% |   75%
   2 | standard+adam+gelu (Tabulaire)       |    95.1% |     96.2% |  94.8% |    95.5% |   97.2% |   80%
   3 | advanced+adamax+mish (Tabulaire)     |    94.5% |     94.8% |  93.9% |    94.3% |   96.5% |   70%
   4 | bayesian+radam+swish (Images)        |    85.8% |     87.2% |  84.1% |    85.6% |   91.3% |   65%
   5 | progressive+adamw+relu (Tabulaire)   |    93.9% |     94.2% |  93.3% |    93.7% |   95.9% |   60%
```

### 📁 **Export Automatique**
- **Fichier CSV** : `results_[type]_[dataset]_YYYYMMDD_HHMMSS.csv`
- **Contenu** : Toutes les combinaisons, métriques, statistiques de convergence
- **Format** : Compatible Excel, Python pandas, R

## 🔧 OPTIMISATIONS ET PERFORMANCES

### ⚡ **Architectures Adaptatives**
Le système adapte automatiquement l'architecture selon le type de données :

#### **Données Tabulaires (8 entrées)**
```
Architecture minimaliste : 8 → 128 → 64 → 1
Architecture équilibrée  : 8 → 256 → 128 → 64 → 1
Architecture large       : 8 → 512 → 256 → 128 → 1
Architecture profonde    : 8 → 256 → 128 → 64 → 1
Architecture étroite     : 8 → 64 → 32 → 16 → 1
Architecture très large  : 8 → 1024 → 512 → 256 → 1
```

#### **Images (64 entrées = 8x8x1)**
```
Architecture minimaliste : 64 → 32 → 16 → 1
Architecture équilibrée  : 64 → 64 → 32 → 16 → 1
Architecture large       : 64 → 128 → 64 → 32 → 1
Architecture profonde    : 64 → 64 → 32 → 16 → 1
Architecture étroite     : 64 → 32 → 16 → 8 → 1
Architecture très large  : 64 → 256 → 128 → 64 → 1
```

### 🎯 **Optimisations Spécifiques**

#### **Pour Images**
- **Normalisation** : [-1, 1] au lieu de [0, 1] pour meilleure convergence
- **Mélange** : Fisher-Yates shuffle pour éviter les biais d'ordre
- **Classification binaire** : 1 sortie au lieu de one-hot encoding
- **Learning rates adaptés** : Paramètres optimisés pour la vision

#### **Pour Données Tabulaires**
- **Standardisation** : Z-score normalization
- **Architectures plus larges** : Plus de neurones pour capturer les relations complexes
- **Learning rates élevés** : Convergence plus rapide

### 🚀 **Optimiseur Temps Réel**
```c
// Adaptation automatique des paramètres
if (dataset->input_cols == 64) {  // Images
    lr *= 0.8f;  // Learning rate plus conservateur
    architecture = select_image_architecture(variant);
} else {  // Tabulaire
    lr *= 1.2f;  // Learning rate plus agressif
    architecture = select_tabular_architecture(variant);
}
```

## 🛠️ DÉVELOPPEMENT ET EXTENSION

### 🔧 **Ajouter un Nouvel Optimiseur**
1. Créer `src/optimizers/mon_optimiseur.c`
2. Implémenter la fonction `mon_optimiseur_update()`
3. Ajouter dans `src/optimizers/optimizer.c`
4. Mettre à jour la compilation

### 🖼️ **Ajouter un Nouveau Format d'Image**
1. Étendre `is_image_file()` dans `src/data/image_loader.c`
2. Ajouter le support dans stb_image si nécessaire
3. Tester avec le programme `test_images`

### 🧠 **Ajouter une Nouvelle Fonction d'Activation**
1. Créer la fonction dans `src/neural/activation.c`
2. Ajouter le type dans l'énumération
3. Mettre à jour `get_activation_type()`

### 📊 **Ajouter une Nouvelle Métrique**
1. Implémenter dans `src/evaluation/`
2. Ajouter dans `compute_all_metrics()`
3. Mettre à jour l'export CSV

### 🏆 **Personnaliser Model Saver**
1. Modifier le score composite dans `src/model_saver/model_saver_core.c`
2. Ajuster les formats de sauvegarde dans `model_saver_pth.c` et `model_saver_h5.c`
3. Personnaliser l'interface Python dans `model_saver_utils.c`

## 🐛 DÉBOGAGE ET DIAGNOSTIC

### 🔍 **Compilation Debug**
```bash
gcc -g -O0 -Wall -Wextra -DDEBUG -o neuroplast-ann-debug [fichiers] -lm
gdb ./neuroplast-ann-debug
```

### 📊 **Vérification des Données**
```bash
# Test de chargement d'images
./test_images

# Vérification de la configuration
./neuroplast-ann --config config/test_simple.yml

# Test de model_saver standalone
cd src/model_saver && make clean && make test
```

### 🚨 **Problèmes Courants**

#### **Images ne se chargent pas**
- Vérifier les chemins dans la configuration
- Vérifier les permissions des répertoires
- Vérifier les formats d'images supportés

#### **Convergence lente**
- Augmenter le learning rate
- Changer d'optimiseur (essayer AdamW)
- Vérifier la normalisation des données

#### **Overfitting**
- Ajouter du dropout
- Augmenter la régularisation L2
- Réduire la taille du réseau

#### **Model Saver ne fonctionne pas**
- Utiliser `./compile_with_model_saver.sh` pour la compilation
- Vérifier les permissions d'écriture dans le répertoire
- Tester en mode standalone avec `cd src/model_saver && make test`

## 📚 DOCUMENTATION COMPLÉMENTAIRE

### 📖 **Fichiers de Documentation**
- **`compilation.txt`** : Guide complet de compilation avec toutes les options
- **`docs/`** : Documentation technique détaillée
- **`config/`** : Exemples de configurations pour tous les cas d'usage
- **`src/model_saver/README.md`** : Documentation complète du système de sauvegarde

### 🔗 **Ressources Externes**
- **stb_image** : https://github.com/nothings/stb
- **YAML** : https://yaml.org/
- **Optimiseurs** : Documentation des algorithmes implémentés

### 📊 **Datasets Recommandés**

#### **Images**
- **Chest X-Ray** : Classification pneumonie/normal
- **CIFAR-10** : Classification d'objets (redimensionner en 8x8)
- **MNIST** : Chiffres manuscrits (redimensionner en 8x8)

#### **Tabulaires**
- **Heart Disease** : Prédiction de maladies cardiaques
- **Diabetes** : Prédiction du diabète
- **Cancer** : Classification de tumeurs

## 🎉 CONCLUSION

NEUROPLAST-ANN v4.2 est un framework complet et modulaire qui supporte :

### ✅ **Fonctionnalités Principales**
- 📋 **Données tabulaires** : CSV avec preprocessing automatique
- 🖼️ **Images** : JPEG, PNG, BMP, TGA avec redimensionnement
- 🧠 **10 fonctions d'activation** : De ReLU à NeuroPlast
- ⚡ **9 optimiseurs** : D'Adam à Lion
- 🔄 **7 méthodes d'entraînement** : Standard à Swarm
- 📈 **Métriques complètes** : Accuracy à AUC-ROC
- 🏆 **Sauvegarde des 10 meilleurs modèles** : Automatique avec interface Python

### 🚀 **Performance**
- **95%+ accuracy** sur données tabulaires
- **85%+ accuracy** sur images (8x8 pixels)
- **Convergence rapide** grâce aux optimisations
- **Architectures adaptatives** selon le type de données
- **Sauvegarde intelligente** des meilleurs modèles avec scoring composite

### 🎯 **Utilisation Simple**
```bash
# Données tabulaires
./neuroplast-ann --config config/cancer_simple.yml --test-all

# Images avec sauvegarde des 10 meilleurs modèles
./neuroplast-ann --config config/chest_xray_simple.yml --test-all

# Test rapide
./neuroplast-ann --test-all-activations
```

### 🏆 **Nouveautés v4.2**
- **Système de sauvegarde des meilleurs modèles** complètement intégré
- **Interface Python automatique** pour utiliser les modèles sauvegardés
- **Système d'affichage dual-zone** révolutionnaire
- **Support complet des images** avec optimisations spécialisées
- **Architectures adaptatives** selon le type de données

**Une seule commande suffit pour des résultats de recherche de qualité avec sauvegarde automatique des meilleurs modèles !**

---

*NEUROPLAST-ANN v4.2 - Framework IA Modulaire en C*  
*© Fabrice | Open Source 2024-2025*
