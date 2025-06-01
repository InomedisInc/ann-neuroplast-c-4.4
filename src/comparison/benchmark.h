#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "../data/dataset.h"
#include "../training/trainer.h"

float benchmark_trainer(Trainer *trainer, Dataset *dataset);

#endif