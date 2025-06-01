#include "roc.h"
#include <stdlib.h>

// Tri, TPR, FPR, AUC : implémentation très simple, à affiner selon besoin.
int cmp_score(const void *a, const void *b) {
    float diff = ((float*)b)[1] - ((float*)a)[1];
    return (diff > 0) - (diff < 0);
}

float compute_auc(const float *y_true, const float *y_score, int n) {
    float arr[n][2];
    for (int i = 0; i < n; ++i) {
        arr[i][0] = y_true[i];
        arr[i][1] = y_score[i];
    }
    qsort(arr, n, sizeof(arr[0]), cmp_score);
    int P = 0, N = 0;
    for (int i = 0; i < n; ++i)
        if (arr[i][0] > 0.5f) P++; else N++;
    float tp = 0, fp = 0, prev_fp = 0, prev_tp = 0, auc = 0;
    for (int i = 0; i < n; ++i) {
        if (arr[i][0] > 0.5f) tp++; else fp++;
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0f;
        prev_fp = fp; prev_tp = tp;
    }
    return auc / (P * N);
}