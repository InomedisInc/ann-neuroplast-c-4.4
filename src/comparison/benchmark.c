#include "benchmark.h"
#include <time.h>

float benchmark_trainer(Trainer *trainer, Dataset *dataset) {
    clock_t start = clock();
    trainer_train(trainer, dataset);
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    return seconds;
}