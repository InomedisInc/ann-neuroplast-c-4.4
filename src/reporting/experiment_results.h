#ifndef EXPERIMENT_RESULTS_H
#define EXPERIMENT_RESULTS_H

#define MAX_EXPERIMENTS 1024
#define MAX_STR 128

typedef struct {
    char method[MAX_STR];
    char optimizer[MAX_STR];
    char activation[MAX_STR];
    float train_acc;
    float test_acc;
} ExperimentResult;

typedef struct {
    ExperimentResult results[MAX_EXPERIMENTS];
    int count;
} ExperimentTable;

// reporting/experiment_results.h
void experiment_table_init(ExperimentTable *table);
void experiment_table_add(ExperimentTable *table, const char *method, const char *optimizer, const char *activation, float train_acc, float test_acc);
void experiment_table_save_csv(ExperimentTable *table, const char *filename);
void experiment_table_print(ExperimentTable *table);


#endif