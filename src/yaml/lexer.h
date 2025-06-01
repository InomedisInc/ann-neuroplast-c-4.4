#ifndef YAML_LEXER_H
#define YAML_LEXER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef enum {
    TOKEN_KEY,
    TOKEN_VALUE,
    TOKEN_COLON,
    TOKEN_ARRAY_START,
    TOKEN_ARRAY_ITEM,
    TOKEN_ARRAY_END,
    TOKEN_INDENT,
    TOKEN_DEDENT,
    TOKEN_EOL,
    TOKEN_EOF
} TokenType;

typedef struct {
    TokenType type;
    char *value;
    int line;
    int indent_level;
} Token;

typedef struct {
    FILE *file;
    char *buffer;
    size_t buffer_size;
    size_t buffer_pos;
    int line;
    int column;
    int indent_level;
    int last_indent_level;
} Lexer;

// Lexer functions
Lexer* lexer_create(const char *filename);
void lexer_free(Lexer *lexer);
Token* lexer_next_token(Lexer *lexer);

#endif 