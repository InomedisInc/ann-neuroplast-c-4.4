#ifndef CSV_EXPORT_H
#define CSV_EXPORT_H

void export_to_csv(const char *filename, const char **header, const char ***rows, int n_rows, int n_cols);

#endif