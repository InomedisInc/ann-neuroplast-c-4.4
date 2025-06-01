#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stddef.h>
#include <stdbool.h>
#include "dataset.h"
#include "../rich_config.h"
#include "image_loader.h"

// Charge un fichier CSV directement
Dataset *load_csv_data(const char *filepath, size_t input_cols, size_t output_cols);

// Charge un dataset à partir d'un fichier de configuration YAML
Dataset *load_dataset_from_yaml(const char *yaml_path);

// Charge un dataset d'images ou tabulaire selon la configuration
Dataset *load_dataset_from_config(const RichConfig *config);

// Fusionne deux datasets
Dataset *merge_datasets(const Dataset *d1, const Dataset *d2);

#endif