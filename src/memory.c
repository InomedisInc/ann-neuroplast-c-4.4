#include "memory.h"
#include <stdio.h>
#include <stdlib.h>

// Allocation mémoire sécurisée
void *mem_alloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Erreur d'allocation mémoire de taille %zu\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Allocation mémoire initialisée à zéro sécurisée
void *mem_calloc(size_t num, size_t size) {
    void *ptr = calloc(num, size);
    if (!ptr) {
        fprintf(stderr, "Erreur d'allocation mémoire (calloc) de taille %zu\n", num * size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Libération sécurisée
void mem_free(void *ptr) {
    if (ptr) {
        free(ptr);
        ptr = NULL;
    }
}