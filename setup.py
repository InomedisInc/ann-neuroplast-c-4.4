#!/usr/bin/env python3
"""
Script d'installation pour NEUROPLAST-ANN Model Loader
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Installer les dépendances"""
    print("🔧 Installation des dépendances...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dépendances installées avec succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur installation: {e}")
        return False

def test_installation():
    """Tester l'installation"""
    print("🧪 Test de l'installation...")
    
    try:
        import tensorflow as tf
        import numpy as np
        import pandas as pd
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        import h5py
        import colorama
        
        print(f"✅ TensorFlow: {tf.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ Pandas: {pd.__version__}")
        print(f"✅ Matplotlib: {matplotlib.__version__}")
        print(f"✅ Scikit-learn: {sklearn.__version__}")
        print("✅ Toutes les dépendances sont installées")
        return True
        
    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        return False

def create_test_script():
    """Créer un script de test rapide"""
    test_script = """#!/usr/bin/env python3
# Test rapide du Model Loader
import sys
sys.path.append('.')

try:
    from model_loader import NeuroplastModelLoader
    print("✅ Model Loader importé avec succès")
    
    loader = NeuroplastModelLoader()
    print("✅ Loader initialisé")
    
    if loader.model_directories:
        print(f"✅ {len(loader.model_directories)} répertoires de modèles trouvés")
    else:
        print("⚠️ Aucun modèle trouvé - exécutez d'abord NEUROPLAST-ANN")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
"""
    
    with open("test_installation.py", "w") as f:
        f.write(test_script)
    
    print("✅ Script de test créé: test_installation.py")

def main():
    """Fonction principale"""
    print("🚀 Installation NEUROPLAST-ANN Model Loader v4.3")
    print("=" * 50)
    
    # Vérifier que nous sommes dans le bon répertoire
    if not Path("requirements.txt").exists():
        print("❌ Fichier requirements.txt non trouvé")
        print("   Assurez-vous d'être dans le répertoire load_model/")
        return
    
    # Installer les dépendances
    if not install_requirements():
        print("❌ Échec de l'installation")
        return
    
    # Tester l'installation
    if not test_installation():
        print("❌ Échec du test")
        return
    
    # Créer le script de test
    create_test_script()
    
    print("\n🎉 Installation terminée avec succès !")
    print("\n📋 Prochaines étapes:")
    print("1. Exécutez NEUROPLAST-ANN pour créer des modèles:")
    print("   cd ..")
    print("   ./neuroplast-ann --config config/cancer_simple.yml --test-all")
    print("\n2. Testez le chargement des modèles:")
    print("   python3 model_loader.py")
    print("\n3. Ou testez un modèle spécifique:")
    print("   python3 test_specific_model.py ../best_models_neuroplast_cancer/model_1.h5")

if __name__ == "__main__":
    main() 