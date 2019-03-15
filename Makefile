# Include common makefile
include src/MakefileCommon

# List of backends
BACKENDS		:=single_threaded_cpu
ifdef CUDA_PATH
	BACKENDS	+=cuda
endif

# Build list of libraries
BACKEND_LIBS		:=$(BACKENDS:%=$(LIBRARY_DIRECTORY)/libgenn_%_backend$(GENN_PREFIX).$(LIBRARY_EXTENSION))

# Default install location
PREFIX 			:= /usr/local

.PHONY: all clean install uninstall libgenn $(BACKENDS)

all: libgenn $(BACKENDS)

install: $(LIBGENN) $(BACKEND_LIBS)
	@# Make install directories
	@mkdir -p $(PREFIX)/bin
	@mkdir -p $(PREFIX)/include
	@mkdir -p $(PREFIX)/lib
	@mkdir -p $(PREFIX)/src/genn
	@mkdir -p $(PREFIX)/src/genn_generator
	@# Deploy libraries and headers
	@cp -f $(LIBRARY_DIRECTORY)/*.a $(PREFIX)/lib
	@cp -rf $(GENN_DIR)/include/* $(PREFIX)/include
	@# Deploy minimal set of Makefiles for building generator
	@cp -r $(GENN_DIR)/src/genn/MakefileCommon $(PREFIX)/src/genn
	@# Deploy genn_generator source and shell scriptlibgenn
	@cp -r $(GENN_DIR)/src/genn_generator/Makefile* $(PREFIX)/src/genn_generator
	@cp -r $(GENN_DIR)/src/genn_generator/generator.cc $(PREFIX)/src/genn_generator
	@cp -r $(GENN_DIR)/bin/genn-buildmodel.sh $(PREFIX)/bin

uninstall:
	@# Delete sources
	@rm -rf $(PREFIX)/src/genn_generator
	@rm -rf $(PREFIX)/src/genn
	@# Delete installed libraries
	@rm -rf $(PREFIX)/lib/libgenn*.a
	@# Delete installed headers
	@rm -rf $(PREFIX)/include/genn
	@rm -rf $(PREFIX)/include/genn_*_backend
	# Delete installed executables
	@rm -f $(PREFIX)/bin/genn-buildmodel.sh

libgenn:
	$(MAKE) -C src/genn

single_threaded_cpu:
	$(MAKE) -C src/genn_single_threaded_cpu_backend

cuda:
	$(MAKE) -C src/genn_cuda_backend

clean:
	@# Delete all objects, dependencies and coverage files if object directory exists
	@if [ -d "${OBJECT_DIRECTORY}" ]; then find $(OBJECT_DIRECTORY) -type f \( -name "*.o" -o -name "*.d" -o -name "*.gcda" -o -name "*.gcdo" \) -delete; fi;

	@# Delete libGeNN
	@# Delete libGeNN
	@rm -f $(LIBGENN)
