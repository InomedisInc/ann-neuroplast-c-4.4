#include "experiment_results.h"
#include <stdio.h>
#include <string.h>

void experiment_table_init(ExperimentTable *table) {
    table->count = 0;
}

void experiment_table_add(ExperimentTable *table, const char *method, const char *optimizer, const char *activation, float train_acc, float test_acc) {
    if (table->count >= MAX_EXPERIMENTS) return;
    
    ExperimentResult *result = &table->results[table->count];
    strncpy(result->method, method, MAX_STR-1);
    strncpy(result->optimizer, optimizer, MAX_STR-1);
    strncpy(result->activation, activation, MAX_STR-1);
    result->train_acc = train_acc;
    result->test_acc = test_acc;
    
    table->count++;
}

void experiment_table_save_csv(ExperimentTable *table, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) return;
    
    fprintf(f, "Method,Optimizer,Activation,Train Accuracy,Test Accuracy\n");
    for (int i = 0; i < table->count; i++) {
        ExperimentResult *r = &table->results[i];
        fprintf(f, "%s,%s,%s,%.4f,%.4f\n",
                r->method, r->optimizer, r->activation,
                r->train_acc, r->test_acc);
    }
    fclose(f);
}

void experiment_table_print(ExperimentTable *table) {
    printf("\n=== Résultats des expériences ===\n");
    printf("%-20s %-15s %-15s %-12s %-12s\n", 
           "Méthode", "Optimizer", "Activation", "Train Acc", "Test Acc");
    printf("--------------------------------------------------------------------\n");
    
    for (int i = 0; i < table->count; i++) {
        ExperimentResult *r = &table->results[i];
        printf("%-20s %-15s %-15s %10.4f%% %10.4f%%\n",
               r->method, r->optimizer, r->activation,
               r->train_acc * 100, r->test_acc * 100);
    }
    printf("--------------------------------------------------------------------\n");
}