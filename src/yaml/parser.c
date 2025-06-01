#include "parser.h"
#include "lexer.h"
#include "nodes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Simple implementation for our basic needs
YamlNode* yaml_parse_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    YamlNode *root = yaml_create_node("root", NULL);
    char line[1024];
    
    while (fgets(line, sizeof(line), file)) {
        // Remove newline
        line[strcspn(line, "\n")] = 0;
        
        // Skip empty lines and comments
        if (line[0] == 0 || line[0] == '#') continue;
        
        // Split key and value
        char *colon = strchr(line, ':');
        if (colon) {
            // Split into key-value
            *colon = 0;
            char *key = line;
            char *value = colon + 1;
            
            // Trim spaces
            while (*key && (*key == ' ' || *key == '\t')) key++;
            while (*value && (*value == ' ' || *value == '\t')) value++;
            
            if (strlen(key) > 0) {
                // Create node and add to root
                YamlNode *node = yaml_create_node(key, value);
                yaml_add_child(root, node);
            }
        }
    }
    
    fclose(file);
    return root;
}

YamlNode* yaml_create_node(const char *key, const char *value) {
    YamlNode *node = malloc(sizeof(YamlNode));
    if (!node) return NULL;
    
    node->key = key ? strdup(key) : NULL;
    node->value = value ? strdup(value) : NULL;
    node->children = NULL;
    node->num_children = 0;
    node->is_array = false;
    
    return node;
}

void yaml_add_child(YamlNode *parent, YamlNode *child) {
    if (!parent || !child) return;
    
    parent->children = realloc(parent->children, (parent->num_children + 1) * sizeof(YamlNode*));
    if (!parent->children) {
        free(child->key);
        free(child->value);
        free(child);
        return;
    }
    
    parent->children[parent->num_children++] = child;
}

void yaml_free_node(YamlNode *node) {
    if (!node) return;
    
    free(node->key);
    free(node->value);
    
    for (int i = 0; i < node->num_children; i++) {
        yaml_free_node(node->children[i]);
    }
    
    free(node->children);
    free(node);
}

const char* yaml_get_value(YamlNode *node, const char *key) {
    if (!node || !key) return NULL;
    
    // Check direct children
    for (int i = 0; i < node->num_children; i++) {
        YamlNode *child = node->children[i];
        if (child && child->key && strcmp(child->key, key) == 0) {
            return child->value;
        }
    }
    
    return NULL;
}

YamlNode* yaml_get_node(YamlNode *node, const char *key) {
    if (!node || !key) return NULL;
    
    // Check direct children
    for (int i = 0; i < node->num_children; i++) {
        YamlNode *child = node->children[i];
        if (child && child->key && strcmp(child->key, key) == 0) {
            return child;
        }
    }
    
    return NULL;
}

int yaml_get_array_size(YamlNode *node, const char *key) {
    YamlNode *array_node = yaml_get_node(node, key);
    if (!array_node || !array_node->is_array) return 0;
    
    return array_node->num_children;
}

const char* yaml_get_array_element(YamlNode *node, const char *key, int index) {
    YamlNode *array_node = yaml_get_node(node, key);
    if (!array_node || !array_node->is_array || index < 0 || index >= array_node->num_children) {
        return NULL;
    }
    
    return array_node->children[index]->value;
} 