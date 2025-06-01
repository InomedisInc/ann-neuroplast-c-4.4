#include "compare.h"

float compare_models(float acc1, float acc2) {
    return (acc1 > acc2) ? acc1 : acc2;
}