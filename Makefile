# Include common makefile
include src/genn/MakefileCommon

# List of backends
BACKENDS		:=single_threaded_cpu_backend
ifdef CUDA_PATH
	BACKENDS	+=cuda_backend
endif

ifdef HIP_PATH
	BACKENDS	+=hip_backend
endif

ifdef OPENCL_PATH
	BACKENDS	+=opencl_backend
endif

# Build list of libraries
BACKEND_LIBS		:=$(BACKENDS:%=$(LIBRARY_DIRECTORY)/libgenn_%$(GENN_PREFIX).$(LIBRARY_EXTENSION))

# Default install location
PREFIX 			?= /usr/local

.PHONY: all clean genn $(BACKENDS)

all: genn $(BACKENDS)

genn:
	$(MAKE) -C src/genn/genn

single_threaded_cpu_backend: genn
	$(MAKE) -C src/genn/backends/single_threaded_cpu

cuda_backend: genn
	$(MAKE) -C src/genn/backends/cuda

hip_backend: genn
	$(MAKE) -C src/genn/backends/hip

opencl_backend: genn
	$(MAKE) -C src/genn/backends/opencl

clean:
	@# Delete all objects, dependencies and coverage files if object directory exists
	@if [ -d "${OBJECT_DIRECTORY}" ]; then find $(OBJECT_DIRECTORY) -type f \( -name "*.o" -o -name "*.d" -o -name "*.gcda" -o -name "*.gcdo" \) -delete; fi;

	@# Delete libGeNN
	@rm -f $(LIBGENN)
	@rm -f $(BACKEND_LIBS)

GENN_VER := $(shell cat version.txt)
.PHONY docker-build:
docker-build:
	@docker build --build-arg GENN_VER=$(GENN_VER) -t genn:latest .
