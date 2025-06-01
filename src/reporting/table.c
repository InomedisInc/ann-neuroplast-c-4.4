#include "table.h"
#include <stdio.h>

void print_table_header(const char **col_names, int n) {
    for (int i = 0; i < n; ++i)
        printf("| %-12s ", col_names[i]);
    printf("|\n");
    for (int i = 0; i < n; ++i)
        printf("---------------");
    printf("\n");
}

void print_table_row(const char **values, int n) {
    for (int i = 0; i < n; ++i)
        printf("| %-12s ", values[i]);
    printf("|\n");
}