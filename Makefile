# Include common makefile
include src/genn/MakefileCommon

# List of backends
BACKENDS		:=single_threaded_cpu
ifdef CUDA_PATH
	BACKENDS	+=cuda
endif

# Build list of libraries
BACKEND_LIBS		:=$(BACKENDS:%=$(GENN_DIR)/lib/libgenn_%_backend$(GENN_PREFIX).a)

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
	@cp -f $(GENN_DIR)/lib/*.a $(PREFIX)/lib
	@cp -rf $(GENN_DIR)/include/* $(PREFIX)/include
	@# Deploy minimal set of Makefiles for building generator
	@cp -r $(GENN_DIR)/src/genn/MakefileCommon $(PREFIX)/src/genn
	@# Deploy genn_generator source and shell script
	@cp -r $(GENN_DIR)/src/genn_generator/Makefile* $(PREFIX)/src/genn_generator
	@cp -r $(GENN_DIR)/src/genn_generator/generator.cc $(PREFIX)/src/genn_generator
	@cp -r $(GENN_DIR)/bin/genn-buildmodel.sh $(PREFIX)/bin

uninstall:
	@# Delete sources
	@rm -rf $(PREFIX)/src/genn_generator
	@rm -rf $(PREFIX)/src/genn
	@rm -rf $(PREFIX)/lib/libgenn*.a
	@rm -rf $(PREFIX)/include/genn
	@rm -rf $(PREFIX)/include/genn_*_backend
	@rm -f $(PREFIX)/bin/genn-buildmodel.sh

libgenn:
	$(MAKE) -C src/genn

single_threaded_cpu:
	$(MAKE) -C src/genn_single_threaded_cpu_backend

cuda:
	$(MAKE) -C src/genn_cuda_backend
	
clean:
	rm -rf $(OBJECT_DIRECTORY)/* $(LIBGENN)