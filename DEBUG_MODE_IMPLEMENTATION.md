# Mode Debug - Implémentation NEUROPLAST-ANN v4.3

## 📋 Résumé de l'implémentation

Le système de mode debug a été intégré dans NEUROPLAST-ANN v4.3 pour permettre l'affichage conditionnel des messages de débogage selon la configuration.

## 🔧 Modifications apportées

### 1. Structure de configuration (`src/rich_config.h`)
- **Ajout du champ** : `int debug_mode;` dans la structure `RichConfig`
- **Position** : Après les champs existants dans la section de configuration générale

### 2. Parser YAML (`src/yaml_parser_rich.c`)
- **Initialisation par défaut** : `cfg->debug_mode = 0;` (debug masqué par défaut)
- **Parsing du champ** : Support de `debug_mode: true/false/yes/no/1/0`
- **Fonction mise à jour** : Ajout dans `parse_yaml_rich_config()` et `read_dataset_dimensions()`

### 3. Macro de debug (`src/main.c`)
```c
#define DEBUG_PRINTF(config, ...) do { \
    if ((config) && (config)->debug_mode) { \
        printf(__VA_ARGS__); \
    } \
} while(0)
```

### 4. Fonction compute_all_metrics mise à jour
- **Signature modifiée** : `AllMetrics compute_all_metrics(NeuralNetwork *network, Dataset *dataset, const RichConfig *config)`
- **Messages de debug conditionnels** :
  - `🔍 Debug Métriques: Scores [...]` → Affiché seulement si debug_mode = true
  - `Matrice: TP=... FP=... FN=... TN=...` → Affiché seulement si debug_mode = true

### 5. Affichage de la configuration (`src/main.c`)
- **Ajout dans print_rich_config()** :
  ```c
  printf("Debug mode   : %s\n", cfg->debug_mode ? "🔍 Activé" : "🔇 Masqué");
  ```

## 📁 Fichiers de configuration mis à jour

Tous les fichiers YAML ont été mis à jour avec `debug_mode: false` par défaut :

| Fichier | Debug Mode | Utilisation |
|---------|------------|-------------|
| `config/cancer_simple.yml` | `false` | Tests cancer (simple) |
| `config/cancer_tabular.yml` | `false` | Tests cancer (analyse complète) |
| `config/chest_xray_images.yml` | `false` | Images radiographies |
| `config/chest_xray_simple.yml` | `false` | Images radiographies (simple) |
| `config/diabetes_simple.yml` | `false` | Tests diabète (simple) |
| `config/diabetes_tabular.yml` | `false` | Tests diabète (analyse complète) |
| `config/heart_disease_simple.yml` | `false` | Tests maladie cardiaque (simple) |
| `config/heart_disease_tabular.yml` | `false` | Tests maladie cardiaque (complète) |

## 📝 Fichiers d'exemple créés

### Configuration avec debug activé
**Fichier** : `config/example_debug_enabled.yml`
```yaml
debug_mode: true  # 🔍 MODE DEBUG ACTIVÉ - Affiche les messages de debug
```

### Configuration avec debug désactivé
**Fichier** : `config/example_debug_disabled.yml`
```yaml
debug_mode: false  # 🔇 MODE PRODUCTION - Messages de debug masqués
```

## 🎯 Utilisation

### Mode Debug Activé
```bash
./neuroplast-ann --config config/example_debug_enabled.yml --test-neuroplast-methods
```
**Résultat** : Affiche `Debug mode   : 🔍 Activé` + messages de debug lors du calcul des métriques

### Mode Debug Désactivé
```bash
./neuroplast-ann --config config/example_debug_disabled.yml --test-neuroplast-methods
```
**Résultat** : Affiche `Debug mode   : 🔇 Masqué` + aucun message de debug

### Configuration YAML
```yaml
# Configuration d'entraînement
batch_size: 32
max_epochs: 100
learning_rate: 0.005
debug_mode: true   # true = affichage des messages, false = masqué
```

## 🔍 Messages de debug disponibles

Quand `debug_mode: true`, les messages suivants sont affichés :

1. **Analyse des métriques** :
   ```
   🔍 Debug Métriques: Scores [min, max] | Pred[0:count, 1:count] | True[0:count, 1:count] | Seuil: threshold
   ```

2. **Matrice de confusion** :
   ```
   Matrice: TP=xx FP=xx FN=xx TN=xx
   ```

## ✅ Tests de validation

1. **Compilation réussie** : `./compile_with_model_saver.sh` ✅
2. **Affichage conditionnel** : Vérifié avec les deux modes ✅
3. **Parsing YAML** : Tous les formats supportés (true/false/yes/no/1/0) ✅
4. **Compatibilité** : Aucun impact sur les fonctionnalités existantes ✅

## 📊 Statut de l'implémentation

- ✅ **Structure de données** : debug_mode ajouté à RichConfig
- ✅ **Parser YAML** : Support complet du champ debug_mode
- ✅ **Macro DEBUG_PRINTF** : Créée et fonctionnelle
- ✅ **Messages conditionnels** : Implémentés dans compute_all_metrics
- ✅ **Affichage configuration** : Statut debug visible
- ✅ **Fichiers de config** : Tous mis à jour avec debug_mode: false
- ✅ **Exemples** : Configurations test créées
- ✅ **Tests** : Fonctionnement validé

## 🎯 Avantages

1. **Mode Production** : Debug masqué par défaut pour de meilleures performances
2. **Mode Développement** : Debug activable facilement via YAML
3. **Flexibilité** : Contrôle fin des messages selon les besoins
4. **Performance** : Aucun overhead quand debug_mode = false
5. **Simplicité** : Configuration via un seul paramètre boolean

L'implémentation est **complète et fonctionnelle** ! 🎉 