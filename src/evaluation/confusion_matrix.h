#ifndef CONFUSION_MATRIX_H
#define CONFUSION_MATRIX_H

void compute_confusion_matrix(const int *y_true, const int *y_pred, int n, int *TP, int *TN, int *FP, int *FN);

#endif