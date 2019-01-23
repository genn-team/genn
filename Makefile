# Include common makefile
include MakefileCommon

# List of backends
BACKENDS		:=single_threaded_cpu
ifdef CUDA_PATH
	BACKENDS	+=cuda
endif

# Build list of libraries
BACKEND_LIBS		:=$(BACKENDS:%=$(GENN_DIR)/lib/lib%_backend$(GENN_PREFIX).a)
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

.PHONY: all clean

all: $(LIBGENN) $(BACKEND_LIBS)

$(LIBGENN):
	$(MAKE) -f MakefileLibGeNN

$(GENN_DIR)/lib/libsingle_threaded_cpu_backend$(GENN_PREFIX).a:
	$(MAKE) -f MakefileBackendSingleThreadedCPU

$(GENN_DIR)/lib/libcuda_backend$(GENN_PREFIX).a:
	$(MAKE) -f MakefileBackendCUDA
	
clean:
	rm -rf $(OBJECT_DIRECTORY)/* $(LIBGENN)