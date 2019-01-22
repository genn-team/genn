# Include common makefile
include MakefileCommon

# List of backends to install
INSTALL_BACKENDS		:=single_threaded_cpu
ifdef CUDA_PATH
	INSTALL_BACKENDS	+=cuda
endif

LIBGENN				:=$(GENN_DIR)/lib/libgenn$(GENN_PREFIX).a
INSTALL_BACKENDS		:=$(INSTALL_BACKENDS:%=$(GENN_DIR)/lib/lib%$(GENN_PREFIX)_backend.a)
.PHONY: install uninstall

install: $(LIBGENN) $(INSTALL_BACKENDS)
	@echo $(LIBGENN)
	@echo $(INSTALL_BACKENDS)


$(GENN_DIR)/lib/libgenn$(GENN_PREFIX).a:
	$(MAKE) -C lib_genn

$(GENN_DIR)/lib/lib%$(GENN_PREFIX)_backend.a: $(LIBGENN)
	$(MAKE) -C backends/$*