# Guide d'intégration Model Saver dans main.c

Ce guide vous montre comment intégrer la librairie `model_saver` dans votre programme principal `neuroplast-ann`.

## 🔧 Étapes d'intégration

### 1. Modification du Makefile principal

Ajoutez ces lignes à votre Makefile principal :

```makefile
# Ajouter model_saver aux sources
MODEL_SAVER_DIR = src/model_saver
MODEL_SAVER_SOURCES = $(MODEL_SAVER_DIR)/model_saver.c \
                     $(MODEL_SAVER_DIR)/model_saver_core.c \
                     $(MODEL_SAVER_DIR)/model_saver_pth.c \
                     $(MODEL_SAVER_DIR)/model_saver_h5.c \
                     $(MODEL_SAVER_DIR)/model_saver_utils.c \
                     $(MODEL_SAVER_DIR)/integration_main.c

# Ajouter aux includes
INCLUDES += -I$(MODEL_SAVER_DIR)

# Ajouter aux sources principales
SOURCES += $(MODEL_SAVER_SOURCES)
```

### 2. Modification de main.c

#### A. Inclure les headers nécessaires

```c
#include "model_saver/integration_main.h"
```

#### B. Initialisation (début de main)

```c
int main(int argc, char *argv[]) {
    // ... votre code d'initialisation existant ...
    
    // Initialiser le système de sauvegarde des modèles
    printf("🔧 Initialisation du système de sauvegarde...\n");
    if (init_model_saver("./best_models") != 0) {
        fprintf(stderr, "❌ Erreur: Impossible d'initialiser ModelSaver\n");
        return 1;
    }
    
    // ... reste de votre initialisation ...
}
```

#### C. Dans la boucle d'entraînement

```c
// Votre boucle d'entraînement existante
for (int epoch = 0; epoch < max_epochs; epoch++) {
    // ... votre code d'entraînement existant ...
    
    // Entraîner une époque
    trainer_train(trainer, train_dataset);
    
    // Évaluer et potentiellement sauvegarder le modèle
    // (Ceci remplace ou complète votre évaluation existante)
    int save_result = evaluate_and_save_model(network, trainer, 
                                             train_dataset, val_dataset, epoch);
    
    // Optionnel : affichage conditionnel selon le résultat
    if (save_result == 1) {
        // Modèle ajouté au top 10 - peut-être sauvegarder des infos supplémentaires
    }
    
    // ... reste de votre code d'époque ...
}
```

#### D. Finalisation (fin de main)

```c
    // ... votre code de fin d'entraînement existant ...
    
    // Finaliser la sauvegarde des modèles
    printf("\n🎯 Finalisation de l'entraînement...\n");
    int saved_count = finalize_model_saving(FORMAT_BOTH);
    if (saved_count > 0) {
        printf("✅ %d modèles sauvegardés avec succès\n", saved_count);
    }
    
    // ... votre code de nettoyage existant ...
    
    // Nettoyer le système de sauvegarde
    cleanup_model_saver();
    
    return 0;
}
```

### 3. Exemple complet d'intégration

```c
#include <stdio.h>
#include <stdlib.h>
// ... vos includes existants ...
#include "model_saver/integration_main.h"

int main(int argc, char *argv[]) {
    printf("🚀 Démarrage de neuroplast-ann avec sauvegarde automatique\n");
    
    // Votre code d'initialisation existant
    // ...
    
    // Initialiser le système de sauvegarde
    if (init_model_saver("./best_models") != 0) {
        return 1;
    }
    
    // Créer le réseau, trainer, datasets (votre code existant)
    NeuralNetwork *network = network_create(/* vos paramètres */);
    Trainer *trainer = trainer_create(/* vos paramètres */);
    Dataset *train_dataset = /* votre dataset d'entraînement */;
    Dataset *val_dataset = /* votre dataset de validation */;
    
    // Boucle d'entraînement
    int max_epochs = 1000; // ou votre valeur
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        // Entraîner
        trainer_train(trainer, train_dataset);
        
        // Évaluer et sauvegarder automatiquement
        evaluate_and_save_model(network, trainer, train_dataset, val_dataset, epoch);
        
        // Vos autres traitements d'époque...
    }
    
    // Finaliser
    finalize_model_saving(FORMAT_BOTH);
    
    // Nettoyage (votre code existant)
    network_free(network);
    trainer_free(trainer);
    // ... autres nettoyages ...
    
    cleanup_model_saver();
    
    printf("✅ Entraînement terminé avec sauvegarde automatique\n");
    return 0;
}
```

## 🎯 Fonctionnalités automatiques

Une fois intégré, le système :

1. **Évalue automatiquement** chaque modèle à chaque époque
2. **Sélectionne automatiquement** les 10 meilleurs modèles
3. **Sauvegarde automatiquement** en formats PTH et H5
4. **Génère automatiquement** l'interface Python
5. **Affiche le classement** en temps réel

## 📊 Personnalisation

### Changer le répertoire de sauvegarde

```c
init_model_saver("./mon_dossier_modeles");
```

### Changer le format de sauvegarde

```c
finalize_model_saving(FORMAT_PTH);   // Seulement PTH
finalize_model_saving(FORMAT_H5);    // Seulement H5
finalize_model_saving(FORMAT_BOTH);  // Les deux (recommandé)
```

### Évaluation personnalisée

Si vous avez vos propres fonctions d'évaluation, modifiez `evaluate_and_save_model` dans `integration_main.c` :

```c
// Remplacer ces lignes dans integration_main.c
float train_accuracy = votre_fonction_evaluation(network, train_data);
float train_loss = votre_fonction_perte(network, train_data);
float val_accuracy = votre_fonction_evaluation(network, val_data);
float val_loss = votre_fonction_perte(network, val_data);
```

## 🐍 Utilisation des modèles sauvegardés

Après l'entraînement, utilisez l'interface Python générée :

```python
from best_models.model_loader import NeuralNetworkLoader

loader = NeuralNetworkLoader("./best_models")
models = loader.list_models()
print("Meilleurs modèles:", models)

# Utiliser le meilleur modèle
best_model = models[0]
prediction = loader.predict(best_model, your_input_data)
```

## 🔍 Vérification de l'intégration

Pour vérifier que l'intégration fonctionne :

1. **Compilation** : Votre programme doit compiler sans erreur
2. **Exécution** : Vous devriez voir les messages de ModelSaver
3. **Fichiers** : Le dossier `best_models/` doit être créé
4. **Sauvegarde** : Les fichiers `.pth` et `.h5` doivent apparaître
5. **Python** : Le fichier `model_loader.py` doit être généré

## ⚠️ Points d'attention

1. **Mémoire** : Le système garde 10 copies complètes des modèles en mémoire
2. **Performance** : L'évaluation à chaque époque peut ralentir l'entraînement
3. **Espace disque** : Prévoyez l'espace pour 20 fichiers de modèles (10 PTH + 10 H5)
4. **Permissions** : Assurez-vous que le programme peut créer des dossiers et fichiers

## 🚀 Optimisations possibles

### Évaluation moins fréquente

```c
// Évaluer seulement toutes les 10 époques
if (epoch % 10 == 0) {
    evaluate_and_save_model(network, trainer, train_dataset, val_dataset, epoch);
}
```

### Sauvegarde conditionnelle

```c
// Sauvegarder seulement si le modèle est dans le top 10
int result = evaluate_and_save_model(/* ... */);
if (result == 1) {
    printf("🏆 Nouveau modèle dans le top 10!\n");
    // Actions supplémentaires si nécessaire
}
```

## 📝 Exemple de sortie

```
🚀 Démarrage de neuroplast-ann avec sauvegarde automatique
✅ ModelSaver initialisé: ./best_models
🏆 Époque 15: Modèle ajouté au top 10! (Acc: 0.892, Val: 0.876)
📊 Époque 16: Modèle pas dans le top 10 (Acc: 0.845, Val: 0.831)
🏆 Époque 23: Modèle ajouté au top 10! (Acc: 0.934, Val: 0.921)
...
🎯 === FINALISATION DE L'ENTRAÎNEMENT ===
=== TOP 10 MODÈLES ===
Rang | Nom du modèle | Score | Précision | Perte | Val. Précision | Val. Perte | Époque
-----|---------------|-------|-----------|-------|----------------|------------|-------
   1 | model_23      | 0.934 |     0.934 | 0.066 |          0.921 |      0.079 |     23
...
💾 Sauvegarde des modèles...
✅ 20 fichiers sauvegardés
🐍 Interface Python exportée: ./best_models/model_loader.py
🧹 ModelSaver nettoyé
✅ Entraînement terminé avec sauvegarde automatique
```

Cette intégration vous donnera un système de sauvegarde automatique robuste et facile à utiliser ! 