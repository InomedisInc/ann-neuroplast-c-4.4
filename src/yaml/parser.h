#ifndef YAML_PARSER_H
#define YAML_PARSER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct YamlNode {
    char *key;
    char *value;
    struct YamlNode **children;
    int num_children;
    bool is_array;
} YamlNode;

// Main parser function
YamlNode* yaml_parse_file(const char *filename);

// Node manipulation
YamlNode* yaml_create_node(const char *key, const char *value);
void yaml_add_child(YamlNode *parent, YamlNode *child);
void yaml_free_node(YamlNode *node);

// Value extraction
const char* yaml_get_value(YamlNode *node, const char *key);
YamlNode* yaml_get_node(YamlNode *node, const char *key);
int yaml_get_array_size(YamlNode *node, const char *key);
const char* yaml_get_array_element(YamlNode *node, const char *key, int index);

#endif 