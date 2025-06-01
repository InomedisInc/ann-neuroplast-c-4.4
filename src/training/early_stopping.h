#ifndef EARLY_STOPPING_H
#define EARLY_STOPPING_H

#include "trainer.h"

typedef struct {
    int patience;
    float min_delta;
    int wait;
    float best_metric;
    int stopped_epoch;
} EarlyStopping;

void early_stopping_init(EarlyStopping *es, int patience, float min_delta);
int early_stopping_check(EarlyStopping *es, float metric, int epoch);

#endif