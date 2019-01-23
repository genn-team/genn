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
# 
# LIBGENN				:=$(GENN_DIR)/lib/libgenn$(GENN_PREFIX).a
# INSTALL_BACKENDS		:=$(INSTALL_BACKENDS:%=$(GENN_DIR)/lib/lib%$(GENN_PREFIX)_backend.a)
# .PHONY: install uninstall
# 
# install: $(LIBGENN) $(INSTALL_BACKENDS)
# 	@echo $(LIBGENN)
# 	@echo $(INSTALL_BACKENDS)
# 
# 
# $(GENN_DIR)/lib/libgenn$(GENN_PREFIX).a:
# 	$(MAKE) -C lib_genn
# 
# $(GENN_DIR)/lib/lib%$(GENN_PREFIX)_backend.a: $(LIBGENN)
# 	$(MAKE) -C backends/$*

.PHONY: all clean install uninstall

all: $(LIBGENN) $(BACKEND_LIBS)

install: $(LIBGENN) $(BACKEND_LIBS)
	@# Make install directories
	@mkdir -p $(PREFIX)/bin
	@mkdir -p $(PREFIX)/include
	@mkdir -p $(PREFIX)/lib
	@mkdir -p $(PREFIX)/src
	@# Deploy libraries and headers
	@cp -f $(GENN_DIR)/lib/*.a $(PREFIX)/lib
	@cp -rf $(GENN_DIR)/include/* $(PREFIX)/include
	@# Deploy genn_generator source and shell script
	@cp -r $(GENN_DIR)/src/genn_generator/* $(PREFIX)/src
	@cp -r $(GENN_DIR)/bin/genn-buildmodel.sh $(PREFIX)/bin

$(LIBGENN):
	$(MAKE) -C src/genn

$(GENN_DIR)/lib/libgenn_single_threaded_cpu_backend$(GENN_PREFIX).a:
	$(MAKE) -C src/genn_single_threaded_cpu_backend

$(GENN_DIR)/lib/libgenn_cuda_backend$(GENN_PREFIX).a:
	$(MAKE) -C src/genn_cuda_backend
	
clean:
	rm -rf $(OBJECT_DIRECTORY)/* $(LIBGENN)