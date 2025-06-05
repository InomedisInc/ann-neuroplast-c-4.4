#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test simple pour valider les noms des colonnes dans les fichiers CSV
int main() {
    printf("🔍 VALIDATION DES NOMS DE COLONNES DANS LES FICHIERS CSV\n");
    printf("=========================================================\n\n");
    
    // Test 1: Heart Disease CSV
    printf("💖 TEST HEART DISEASE CSV:\n");
    printf("---------------------------\n");
    
    FILE *heart_file = fopen("datasets/heart_disease.csv", "r");
    if (heart_file) {
        char line[2048];
        if (fgets(line, sizeof(line), heart_file)) {
            printf("✅ En-tête trouvé: %s", line);
            
            // Vérifier si les colonnes principales sont présentes
            if (strstr(line, "HeartDiseaseorAttack") && 
                strstr(line, "HighBP") && 
                strstr(line, "BMI") &&
                strstr(line, "Age")) {
                printf("✅ Colonnes principales validées!\n");
            } else {
                printf("❌ Colonnes principales manquantes!\n");
            }
        }
        fclose(heart_file);
    } else {
        printf("❌ Fichier heart_disease.csv introuvable\n");
    }
    
    printf("\n");
    
    // Test 2: Diabetes CSV
    printf("🍃 TEST DIABETES CSV:\n");
    printf("---------------------\n");
    
    FILE *diabetes_file = fopen("datasets/diabetes.csv", "r");
    if (diabetes_file) {
        char line[1024];
        if (fgets(line, sizeof(line), diabetes_file)) {
            printf("✅ En-tête trouvé: %s", line);
            
            // Vérifier si les colonnes principales sont présentes
            if (strstr(line, "Outcome") && 
                strstr(line, "Pregnancies") && 
                strstr(line, "Glucose") &&
                strstr(line, "BMI")) {
                printf("✅ Colonnes principales validées!\n");
            } else {
                printf("❌ Colonnes principales manquantes!\n");
            }
        }
        fclose(diabetes_file);
    } else {
        printf("❌ Fichier diabetes.csv introuvable\n");
    }
    
    printf("\n🎯 Tests de validation des colonnes terminés!\n");
    return 0;
} 