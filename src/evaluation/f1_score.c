#include "f1_score.h"

float compute_f1_score(int TP, int FP, int FN) {
    float precision = TP + FP > 0 ? (float)TP / (TP + FP) : 0;
    float recall    = TP + FN > 0 ? (float)TP / (TP + FN) : 0;
    return (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;
}