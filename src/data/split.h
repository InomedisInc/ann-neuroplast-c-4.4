#ifndef SPLIT_H
#define SPLIT_H

#include "dataset.h"

void split_dataset(const Dataset *src, float ratio, Dataset **train, Dataset **test);

#endif