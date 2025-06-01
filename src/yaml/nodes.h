#ifndef YAML_NODES_H
#define YAML_NODES_H

#include "parser.h"

// Node utility functions
bool yaml_node_is_array(YamlNode *node);
bool yaml_node_has_key(YamlNode *node, const char *key);
YamlNode* yaml_node_get_child(YamlNode *node, const char *key);
void yaml_node_print(YamlNode *node, int depth);

#endif 