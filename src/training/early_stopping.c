#include "early_stopping.h"

void early_stopping_init(EarlyStopping *es, int patience, float min_delta) {
    es->patience = patience;
    es->min_delta = min_delta;
    es->wait = 0;
    es->best_metric = -1e10f;
    es->stopped_epoch = 0;
}

int early_stopping_check(EarlyStopping *es, float metric, int epoch) {
    if (metric > es->best_metric + es->min_delta) {
        es->best_metric = metric;
        es->wait = 0;
    } else {
        es->wait++;
        if (es->wait >= es->patience) {
            es->stopped_epoch = epoch;
            return 1; // Stop
        }
    }
    return 0; // Continue
}