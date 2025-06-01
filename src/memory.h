#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>

// Fonctions d'allocation et libération optimisées
void *mem_alloc(size_t size);
void mem_free(void *ptr);
void *mem_calloc(size_t num, size_t size);

#endif /* MEMORY_H */