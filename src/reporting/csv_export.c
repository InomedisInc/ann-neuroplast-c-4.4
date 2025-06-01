#include "csv_export.h"
#include <stdio.h>

void export_to_csv(const char *filename, const char **header, const char ***rows, int n_rows, int n_cols) {
    FILE *f = fopen(filename, "w");
    if (!f) return;
    for (int i = 0; i < n_cols; ++i) {
        fprintf(f, "%s%s", header[i], (i < n_cols - 1) ? "," : "\n");
    }
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            fprintf(f, "%s%s", rows[i][j], (j < n_cols - 1) ? "," : "\n");
        }
    }
    fclose(f);
}