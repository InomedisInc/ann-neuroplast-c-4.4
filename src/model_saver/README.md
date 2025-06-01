# Model Saver Library

Une librairie C complète pour sauvegarder et gérer les 10 meilleurs modèles de réseaux de neurones, avec interface Python intégrée.

## 🚀 Fonctionnalités

- **Gestion automatique du top 10** : Garde automatiquement les 10 meilleurs modèles basés sur un score composite
- **Formats multiples** : Sauvegarde en format PTH (binaire) et H5 (JSON-like)
- **Interface Python** : Génération automatique d'une interface Python pour utiliser les modèles sauvegardés
- **Métadonnées complètes** : Sauvegarde toutes les informations d'entraînement
- **Sans dépendances externes** : Implémentation pure C sans librairies tierces
- **Compatible multiplateforme** : Fonctionne sur Linux, macOS et Windows

## 📁 Structure des fichiers

```
src/model_saver/
├── model_saver.h           # Header principal
├── model_saver.c           # Fonctions de base
├── model_saver_core.c      # Gestion des modèles
├── model_saver_pth.c       # Format PTH (binaire)
├── model_saver_h5.c        # Format H5 (JSON-like)
├── model_saver_utils.c     # Utilitaires et interface Python
├── example_usage.c         # Exemple d'utilisation
├── Makefile               # Compilation
└── README.md              # Cette documentation
```

## 🛠️ Compilation

### Compilation rapide
```bash
cd src/model_saver
make all
```

### Options de compilation
```bash
make lib      # Librairie dynamique (.so)
make static   # Librairie statique (.a)
make example  # Exemple d'utilisation
make test     # Compile et exécute l'exemple
make clean    # Nettoie les fichiers générés
make help     # Affiche l'aide
```

### Installation système (optionnel)
```bash
make install  # Installe dans /usr/local/lib
```

## 📖 Utilisation

### 1. Initialisation

```c
#include "model_saver.h"

// Créer un gestionnaire de sauvegarde
ModelSaver *saver = model_saver_create("./saved_models");
```

### 2. Ajouter des modèles candidats

```c
// Pendant l'entraînement, ajouter des modèles candidats
int result = model_saver_add_candidate(
    saver,
    network,           // Votre réseau de neurones
    trainer,           // Votre trainer
    accuracy,          // Précision sur le jeu d'entraînement
    loss,              // Perte sur le jeu d'entraînement
    val_accuracy,      // Précision sur le jeu de validation
    val_loss,          // Perte sur le jeu de validation
    epoch              // Numéro d'époque
);

// result = 1 : modèle ajouté au top 10
// result = 0 : modèle pas assez bon
// result = -1 : erreur
```

### 3. Afficher le classement

```c
model_saver_print_rankings(saver);
```

### 4. Sauvegarder les modèles

```c
// Sauvegarder en format PTH uniquement
model_saver_save_all(saver, FORMAT_PTH);

// Sauvegarder en format H5 uniquement
model_saver_save_all(saver, FORMAT_H5);

// Sauvegarder dans les deux formats
model_saver_save_all(saver, FORMAT_BOTH);
```

### 5. Exporter l'interface Python

```c
model_saver_export_python_interface(saver, "./saved_models/model_loader.py");
```

### 6. Charger un modèle

```c
ModelMetadata metadata;
NeuralNetwork *loaded_network = model_saver_load_model("./saved_models/model_1.h5", &metadata);

if (loaded_network) {
    printf("Modèle chargé: %s\n", metadata.model_name);
    printf("Précision: %.3f\n", metadata.accuracy);
    
    // Utiliser le modèle...
    
    // Nettoyer
    network_free(loaded_network);
    free(metadata.layer_sizes);
}
```

### 7. Nettoyage

```c
model_saver_free(saver);
```

## 🐍 Interface Python

La librairie génère automatiquement une interface Python complète :

```python
from model_loader import NeuralNetworkLoader

# Initialiser le chargeur
loader = NeuralNetworkLoader("./saved_models")

# Lister les modèles disponibles
models = loader.list_models()
print("Modèles disponibles:", models)

# Obtenir les informations d'un modèle
info = loader.get_model_info("model_1")
print(f"Précision: {info['accuracy']:.3f}")

# Charger les poids et biais
weights = loader.get_model_weights("model_1")
biases = loader.get_model_biases("model_1")

# Faire une prédiction
import numpy as np
input_data = np.random.randn(1, 10)  # Exemple
prediction = loader.predict("model_1", input_data)
print("Prédiction:", prediction)
```

## 📊 Formats de sauvegarde

### Format PTH (binaire)
- **Avantages** : Compact, rapide à charger
- **Inconvénients** : Non lisible par l'humain
- **Usage** : Production, modèles volumineux

### Format H5 (JSON-like)
- **Avantages** : Lisible, compatible Python, débuggage facile
- **Inconvénients** : Plus volumineux
- **Usage** : Développement, partage, analyse

### Exemple de fichier H5
```json
{
  "format": "NEURH5",
  "version": 1,
  "metadata": {
    "model_name": "model_1",
    "accuracy": 0.945000,
    "loss": 0.123000,
    "validation_accuracy": 0.920000,
    "validation_loss": 0.145000,
    "epoch": 150,
    "optimizer": "adam",
    "learning_rate": 0.001000
  },
  "architecture": {
    "layer_sizes": [784, 128, 64, 10],
    "activation_types": [1, 1, 2]
  },
  "parameters": {
    "layers": [
      {
        "layer_id": 0,
        "weights": [[0.12345, -0.67890, ...], ...],
        "biases": [0.1, -0.2, 0.3, ...],
        "neuroplast_params": {
          "alpha": 1.0,
          "beta": 0.0,
          "gamma": 1.0,
          "delta": 1.0
        }
      }
    ]
  }
}
```

## 🎯 Score composite

Le score utilisé pour classer les modèles combine :
- **40%** Précision d'entraînement
- **40%** Précision de validation  
- **10%** Inverse de la perte d'entraînement
- **10%** Inverse de la perte de validation

```c
float score = (accuracy * 0.4f) + (val_accuracy * 0.4f) + 
              ((1.0f / (1.0f + loss)) * 0.1f) + 
              ((1.0f / (1.0f + val_loss)) * 0.1f);
```

## 🔧 Intégration dans votre projet

### 1. Copier les fichiers
```bash
cp -r src/model_saver/ votre_projet/
```

### 2. Modifier votre Makefile
```makefile
INCLUDES += -Imodel_saver
SOURCES += model_saver/*.c
```

### 3. Dans votre code d'entraînement
```c
#include "model_saver/model_saver.h"

// Initialiser une fois
ModelSaver *saver = model_saver_create("./models");

// Dans votre boucle d'entraînement
for (int epoch = 0; epoch < max_epochs; epoch++) {
    // ... entraînement ...
    
    // Évaluer le modèle
    float acc = evaluate_accuracy(network, train_data);
    float loss = evaluate_loss(network, train_data);
    float val_acc = evaluate_accuracy(network, val_data);
    float val_loss = evaluate_loss(network, val_data);
    
    // Ajouter le modèle candidat
    model_saver_add_candidate(saver, network, trainer, 
                             acc, loss, val_acc, val_loss, epoch);
}

// À la fin de l'entraînement
model_saver_print_rankings(saver);
model_saver_save_all(saver, FORMAT_BOTH);
model_saver_export_python_interface(saver, "./models/loader.py");
model_saver_free(saver);
```

## 🧪 Test de la librairie

```bash
cd src/model_saver
make test
```

Cela va :
1. Compiler la librairie et l'exemple
2. Simuler l'entraînement de 12 modèles
3. Sélectionner les 10 meilleurs
4. Les sauvegarder en formats PTH et H5
5. Générer l'interface Python
6. Tester le chargement d'un modèle

## 📋 Exemple de sortie

```
=== TOP 10 MODÈLES ===
Rang | Nom du modèle | Score | Précision | Perte | Val. Précision | Val. Perte | Époque
-----|---------------|-------|-----------|-------|----------------|------------|-------
   1 | model_8       | 0.876 |     0.945 | 0.123 |          0.920 |      0.145 |     80
   2 | model_5       | 0.864 |     0.932 | 0.134 |          0.915 |      0.152 |     50
   3 | model_12      | 0.851 |     0.928 | 0.142 |          0.908 |      0.158 |    120
   ...
```

## 🤝 Contribution

Cette librairie est conçue pour être :
- **Modulaire** : Facile à étendre avec de nouveaux formats
- **Robuste** : Gestion d'erreurs complète
- **Efficace** : Optimisée pour les performances
- **Portable** : Compatible avec différents systèmes

## 📝 Notes techniques

- **Gestion mémoire** : Toute la mémoire est automatiquement gérée
- **Thread-safety** : Non thread-safe, utiliser des mutex si nécessaire
- **Limites** : Maximum 10 modèles, noms de modèles < 64 caractères
- **Dépendances** : Aucune librairie externe requise

## 🐛 Dépannage

### Erreur de compilation
```bash
# Vérifier les dépendances
make clean
make all
```

### Erreur de chargement Python
```python
# Vérifier le chemin
import os
print(os.path.exists("./saved_models/model_loader.py"))
```

### Modèles non sauvegardés
- Vérifier les permissions du répertoire
- Vérifier l'espace disque disponible
- Vérifier que `model_saver_add_candidate` retourne 1

## 📄 Licence

Cette librairie fait partie du projet neuroplast-ann et suit la même licence. 