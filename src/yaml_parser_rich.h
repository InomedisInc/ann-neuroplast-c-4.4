#ifndef YAML_PARSER_RICH_H
#define YAML_PARSER_RICH_H

#include "rich_config.h"

#define MAX_LINE 1024

// Fonctions de parsing
int parse_yaml_rich(const char *filename, RichConfig *cfg);
int parse_key_value(const char *line, char *key, char *value);

#endif 