#ifndef JSON_EXPORT_H
#define JSON_EXPORT_H

void export_to_json(const char *filename, const char **header, const char ***rows, int n_rows, int n_cols);

#endif