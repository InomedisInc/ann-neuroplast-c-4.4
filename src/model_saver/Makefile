# Makefile pour la librairie Model Saver
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -fPIC
INCLUDES = -I../neural -I../training -I../data -I../optimizers -I..
LDFLAGS = -shared

# Répertoires
SRCDIR = .
OBJDIR = obj
LIBDIR = lib

# Fichiers sources de model_saver
SOURCES = model_saver.c model_saver_core.c model_saver_pth.c model_saver_h5.c model_saver_utils.c
OBJECTS = $(SOURCES:%.c=$(OBJDIR)/%.o)

# Dépendances externes nécessaires
DEPS_SOURCES = ../neural/layer.c ../neural/network.c ../neural/neuroplast.c ../neural/activation.c ../memory.c ../matrix.c ../colored_output.c
DEPS_OBJECTS = $(OBJDIR)/layer.o $(OBJDIR)/network.o $(OBJDIR)/neuroplast.o $(OBJDIR)/activation.o $(OBJDIR)/memory.o $(OBJDIR)/matrix.o $(OBJDIR)/colored_output.o

# Tous les objets
ALL_OBJECTS = $(OBJECTS) $(DEPS_OBJECTS)

# Nom de la librairie
LIBNAME = libmodel_saver.so
STATIC_LIBNAME = libmodel_saver.a

# Exemple
EXAMPLE_SRC = example_usage.c
EXAMPLE_BIN = model_saver_example

# Cibles principales
.PHONY: all clean lib static example install

all: lib static example

# Créer les répertoires nécessaires
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(LIBDIR):
	mkdir -p $(LIBDIR)

# Compiler les objets de model_saver
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Compiler les dépendances spécifiques
$(OBJDIR)/layer.o: ../neural/layer.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/network.o: ../neural/network.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/neuroplast.o: ../neural/neuroplast.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/activation.o: ../neural/activation.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/memory.o: ../memory.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/matrix.o: ../matrix.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/colored_output.o: ../colored_output.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Librairie dynamique
lib: $(LIBDIR)/$(LIBNAME)

$(LIBDIR)/$(LIBNAME): $(ALL_OBJECTS) | $(LIBDIR)
	$(CC) $(LDFLAGS) -o $@ $(ALL_OBJECTS)
	@echo "Librairie dynamique créée: $@"

# Librairie statique
static: $(LIBDIR)/$(STATIC_LIBNAME)

$(LIBDIR)/$(STATIC_LIBNAME): $(ALL_OBJECTS) | $(LIBDIR)
	ar rcs $@ $(ALL_OBJECTS)
	@echo "Librairie statique créée: $@"

# Exemple d'utilisation
example: $(EXAMPLE_BIN)

$(EXAMPLE_BIN): $(EXAMPLE_SRC) $(ALL_OBJECTS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(ALL_OBJECTS)
	@echo "Exemple compilé: $@"

# Installation (optionnel)
install: lib static
	@echo "Installation de la librairie..."
	sudo cp $(LIBDIR)/$(LIBNAME) /usr/local/lib/
	sudo cp $(LIBDIR)/$(STATIC_LIBNAME) /usr/local/lib/
	sudo cp model_saver.h /usr/local/include/
	sudo ldconfig 2>/dev/null || true
	@echo "Installation terminée"

# Tests
test: example
	@echo "Exécution de l'exemple..."
	./$(EXAMPLE_BIN)

# Nettoyage
clean:
	rm -rf $(OBJDIR) $(LIBDIR) $(EXAMPLE_BIN)
	rm -rf saved_models
	@echo "Nettoyage terminé"

# Documentation
doc:
	@echo "=== DOCUMENTATION MODEL_SAVER ==="
	@echo ""
	@echo "COMPILATION:"
	@echo "  make all      - Compile tout (librairies + exemple)"
	@echo "  make lib      - Compile la librairie dynamique"
	@echo "  make static   - Compile la librairie statique"
	@echo "  make example  - Compile l'exemple"
	@echo ""
	@echo "UTILISATION:"
	@echo "  make test     - Exécute l'exemple"
	@echo "  make install  - Installe la librairie système"
	@echo "  make clean    - Nettoie les fichiers générés"
	@echo ""
	@echo "FICHIERS GÉNÉRÉS:"
	@echo "  lib/libmodel_saver.so    - Librairie dynamique"
	@echo "  lib/libmodel_saver.a     - Librairie statique"
	@echo "  model_saver_example      - Exemple d'utilisation"
	@echo "  saved_models/            - Modèles sauvegardés"
	@echo ""
	@echo "INTÉGRATION DANS VOTRE PROJET:"
	@echo "  #include \"model_saver.h\""
	@echo "  gcc -o mon_prog mon_prog.c -lmodel_saver"

# Aide
help: doc 