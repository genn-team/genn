# Include common makefile
include MakefileCommon

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

#install: $(LIBGENN) $(BACKEND_LIBS)
#	mkdir -p 

$(LIBGENN):
	$(MAKE) -f MakefileLibGeNN

$(GENN_DIR)/lib/libgenn_single_threaded_cpu_backend$(GENN_PREFIX).a:
	$(MAKE) -f MakefileBackendSingleThreadedCPU

$(GENN_DIR)/lib/libgenn_cuda_backend$(GENN_PREFIX).a:
	$(MAKE) -f MakefileBackendCUDA
	
clean:
	rm -rf $(OBJECT_DIRECTORY)/* $(LIBGENN)