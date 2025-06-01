#include "metrics.h"
#include <math.h>

float accuracy(float *y_true, float *y_pred, int n) {
    int correct = 0;
    for (int i = 0; i < n; ++i)
        if ((y_true[i] >= 0.5f && y_pred[i] >= 0.5f) ||
            (y_true[i] <  0.5f && y_pred[i] <  0.5f)) correct++;
    return (float)correct / n;
}

float mse(float *y_true, float *y_pred, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
        sum += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    return sum / n;
}