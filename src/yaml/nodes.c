#include "nodes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool yaml_node_is_array(YamlNode *node) {
    return node && node->is_array;
}

bool yaml_node_has_key(YamlNode *node, const char *key) {
    if (!node || !key) return false;
    
    for (int i = 0; i < node->num_children; i++) {
        if (node->children[i] && node->children[i]->key && strcmp(node->children[i]->key, key) == 0) {
            return true;
        }
    }
    
    return false;
}

YamlNode* yaml_node_get_child(YamlNode *node, const char *key) {
    if (!node || !key) return NULL;
    
    for (int i = 0; i < node->num_children; i++) {
        if (node->children[i] && node->children[i]->key && strcmp(node->children[i]->key, key) == 0) {
            return node->children[i];
        }
    }
    
    return NULL;
}

void yaml_node_print(YamlNode *node, int depth) {
    if (!node) return;
    
    for (int i = 0; i < depth; i++) {
        printf("  ");
    }
    
    if (node->key) {
        printf("%s: ", node->key);
    }
    
    if (node->value) {
        printf("%s\n", node->value);
    } else {
        printf("\n");
    }
    
    for (int i = 0; i < node->num_children; i++) {
        yaml_node_print(node->children[i], depth + 1);
    }
} 