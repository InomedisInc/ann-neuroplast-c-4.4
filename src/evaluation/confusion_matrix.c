#include "confusion_matrix.h"

void compute_confusion_matrix(const int *y_true, const int *y_pred, int n, int *TP, int *TN, int *FP, int *FN) {
    *TP = *TN = *FP = *FN = 0;
    for (int i = 0; i < n; ++i) {
        if (y_true[i] == 1 && y_pred[i] == 1) (*TP)++;
        if (y_true[i] == 0 && y_pred[i] == 0) (*TN)++;
        if (y_true[i] == 0 && y_pred[i] == 1) (*FP)++;
        if (y_true[i] == 1 && y_pred[i] == 0) (*FN)++;
    }
}