#ifndef METRICS_H
#define METRICS_H

// Scores basiques pour classification/régression
float accuracy(float *y_true, float *y_pred, int n);
float mse(float *y_true, float *y_pred, int n);

#endif