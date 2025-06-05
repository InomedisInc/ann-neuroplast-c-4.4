#include "roc.h"
#include <stdlib.h>
#include <math.h>

// Tri, TPR, FPR, AUC : implémentation très simple, à affiner selon besoin.
int cmp_score(const void *a, const void *b) {
    float diff = ((float*)b)[1] - ((float*)a)[1];
    return (diff > 0) - (diff < 0);
}

float compute_auc(const float *y_true, const float *y_score, int n) {
    if (!y_true || !y_score || n <= 0) {
        return 0.5f; // AUC par défaut pour cas invalides
    }
    
    // 🔧 CORRECTION 1: Vérifier qu'on a des données valides
    int valid_samples = 0;
    for (int i = 0; i < n; i++) {
        if (!isnan(y_score[i]) && !isinf(y_score[i])) {
            valid_samples++;
        }
    }
    
    if (valid_samples < 2) {
        return 0.5f; // AUC par défaut si pas assez de données valides
    }
    
    float arr[n][2];
    int valid_count = 0;
    
    // 🔧 CORRECTION 2: Filtrer les valeurs invalides
    for (int i = 0; i < n; ++i) {
        if (!isnan(y_score[i]) && !isinf(y_score[i])) {
            arr[valid_count][0] = y_true[i];
            arr[valid_count][1] = y_score[i];
            valid_count++;
        }
    }
    
    if (valid_count < 2) {
        return 0.5f; // AUC par défaut
    }
    
    qsort(arr, valid_count, sizeof(arr[0]), cmp_score);
    
    // 🔧 CORRECTION 3: Compter les positifs et négatifs avec vérification
    int P = 0, N = 0;
    for (int i = 0; i < valid_count; ++i) {
        if (arr[i][0] > 0.5f) P++; 
        else N++;
    }
    
    // 🔧 CORRECTION 4: Gérer les cas où toutes les classes sont identiques
    if (P == 0 || N == 0) {
        return 0.5f; // AUC par défaut quand une seule classe présente
    }
    
    // 🔧 CORRECTION 5: Calcul AUC avec vérification de division par zéro
    float tp = 0, fp = 0, prev_fp = 0, prev_tp = 0, auc = 0;
    for (int i = 0; i < valid_count; ++i) {
        if (arr[i][0] > 0.5f) tp++; 
        else fp++;
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0f;
        prev_fp = fp; 
        prev_tp = tp;
    }
    
    float final_auc = auc / (P * N);
    
    // 🔧 CORRECTION 6: Validation du résultat final
    if (isnan(final_auc) || isinf(final_auc) || final_auc < 0.0f || final_auc > 1.0f) {
        return 0.5f; // AUC par défaut pour résultats invalides
    }
    
    return final_auc;
}