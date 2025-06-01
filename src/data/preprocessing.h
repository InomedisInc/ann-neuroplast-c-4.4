#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "dataset.h"

void normalize_dataset(Dataset *d, float new_min, float new_max);
void shuffle_dataset(Dataset *d);

#endif