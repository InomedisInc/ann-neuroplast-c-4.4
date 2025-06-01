#include "json_export.h"
#include <stdio.h>

void export_to_json(const char *filename, const char **header, const char ***rows, int n_rows, int n_cols) {
    FILE *f = fopen(filename, "w");
    if (!f) return;
    fprintf(f, "[\n");
    for (int i = 0; i < n_rows; ++i) {
        fprintf(f, "  {\n");
        for (int j = 0; j < n_cols; ++j) {
            fprintf(f, "    \"%s\": \"%s\"%s\n", header[j], rows[i][j], (j < n_cols - 1) ? "," : "");
        }
        fprintf(f, "  }%s\n", (i < n_rows - 1) ? "," : "");
    }
    fprintf(f, "]\n");
    fclose(f);
}