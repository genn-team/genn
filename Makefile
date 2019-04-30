# Include common makefile
include src/genn/MakefileCommon

# List of backends
BACKENDS		:=single_threaded_cpu
ifdef CUDA_PATH
	BACKENDS	+=cuda
endif

# Build list of libraries
BACKEND_LIBS		:=$(BACKENDS:%=$(LIBRARY_DIRECTORY)/libgenn_%_backend$(GENN_PREFIX).$(LIBRARY_EXTENSION))

# Default install location
PREFIX 			?= /usr/local

.PHONY: all clean install uninstall libgenn $(BACKENDS)

all: libgenn $(BACKENDS)

install: $(LIBGENN) $(BACKEND_LIBS)
	@# Make install directories
	@mkdir -p $(PREFIX)/bin
	@mkdir -p $(PREFIX)/include/genn
	@mkdir -p $(PREFIX)/lib
	@mkdir -p $(PREFIX)/src/genn/generator
	@# Deploy libraries and headers
	@cp -f $(LIBRARY_DIRECTORY)/libgenn*.a $(PREFIX)/lib
	@cp -rf $(GENN_DIR)/include/genn/* $(PREFIX)/include/genn/
	@# Deploy minimal set of Makefiles for building generator
	@cp -r $(GENN_DIR)/src/genn/MakefileCommon $(PREFIX)/src/genn
	@# Deploy genn_generator source and shell scripts
	@cp -r $(GENN_DIR)/src/genn/generator/Makefile* $(PREFIX)/src/genn/generator
	@cp -r $(GENN_DIR)/src/genn/generator/generator.cc $(PREFIX)/src/genn/generator
	@cp -r $(GENN_DIR)/bin/genn-buildmodel.sh $(PREFIX)/bin
	@cp -r $(GENN_DIR)/bin/genn-create-user-project.sh $(PREFIX)/bin

uninstall:
	@# Delete sources
	@rm -rf $(PREFIX)/src/genn
	@# Delete installed libraries
	@rm -rf $(PREFIX)/lib/libgenn*.a
	@# Delete installed headers
	@rm -rf $(PREFIX)/include/genn
	# Delete installed executables
	@rm -f $(PREFIX)/bin/genn-buildmodel.sh
	@rm -f $(PREFIX)/bin/genn-create-user-project.sh

libgenn:
	$(MAKE) -C src/genn/genn

single_threaded_cpu:
	$(MAKE) -C src/genn/backends/single_threaded_cpu

cuda:
	$(MAKE) -C src/genn/backends/cuda

clean:
	@# Delete all objects, dependencies and coverage files if object directory exists
	@if [ -d "${OBJECT_DIRECTORY}" ]; then find $(OBJECT_DIRECTORY) -type f \( -name "*.o" -o -name "*.d" -o -name "*.gcda" -o -name "*.gcdo" \) -delete; fi;

	@# Delete libGeNN
	@rm -f $(LIBGENN)
	@rm -f $(BACKEND_LIBS)
