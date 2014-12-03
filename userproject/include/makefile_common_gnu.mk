##--------------------------------------------------------------------------
##   Author: Thomas Nowotny
##
##   Institute: Center for Computational Neuroscience and Robotics
##              University of Sussex
##            	Falmer, Brighton BN1 9QJ, UK
##
##   email to:  T.Nowotny@sussex.ac.uk
##
##   initial version: 2010-02-07
##
##--------------------------------------------------------------------------


# Makefile include for all GeNN projects
# This is a UNIX Makefile, to be used by the GNU make build system
#-----------------------------------------------------------------

# OS name (Linux or Darwin) and architecture (32 bit or 64 bit).
OS_SIZE		:=$(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_UPPER 	:=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OS_LOWER 	:=$(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
DARWIN  	:=$(strip $(findstring DARWIN, $(OS_UPPER)))

# Global C++ and CUDA compiler settings.
CUDA_PATH	?=/usr/local/cuda
NVCC		:=$(CUDA_PATH)/bin/nvcc
NVCCFLAGS	+=
CXXFLAGS	+= 
ifeq ($(DARWIN),DARWIN)
   CXX		= clang++
endif

# Global include flags and link flags.
INCLUDE_FLAGS	+=-I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -I$(GENN_PATH)/lib/include -I$(GENN_PATH)/userproject/include
ifeq ($(DARWIN),DARWIN)
  LINK_FLAGS	+=-L$(CUDA_PATH)/lib -lcudart -stdlib=libstdc++ -lc++ 
else
  ifeq ($(OS_SIZE),32)
    LINK_FLAGS	+=-L$(CUDA_PATH)/lib -lcudart 
  else
    LINK_FLAGS	+=-L$(CUDA_PATH)/lib64 -lcudart 
  endif
endif

# An auto-generated file containing your cuda device's compute capability.
include sm_version.mk

# Enumerate all source and object files (if they have not already been listed).
SOURCES		?=$(wildcard *.cc *.cpp *.cu)
OBJECTS_FLOAT		:=$(foreach obj,$(basename $(SOURCES)),$(obj)_FLOAT.o)
OBJECTS_DOUBLE		:=$(foreach obj,$(basename $(SOURCES)),$(obj)_DOUBLE.o)

# Target rules.
.PHONY: all
all: release

%_FLOAT.o: %.cc
	$(CXX) -Dscalar=float $(CXXFLAGS) $(INCLUDE_FLAGS) $< -o $@ -c

%_DOUBLE.o: %.cc
	$(CXX) -Dscalar=double $(CXXFLAGS) $(INCLUDE_FLAGS) $< -o $@ -c

%_FLOAT.o: %.cpp
	$(CXX) -Dscalar=float $(CXXFLAGS) $(INCLUDE_FLAGS) $< -o $@ -c

%_DOUBLE.o: %.cpp
	$(CXX) -Dscalar=double $(CXXFLAGS) $(INCLUDE_FLAGS) $< -o $@ -c

%_FLOAT.o: %.cu
	$(NVCC) -Dscalar=float $(NVCCFLAGS) $(INCLUDE_FLAGS) $(GENCODE_FLAGS) $< -o $@ -c

%_DOUBLE.o: %.cu
	$(NVCC) -Dscalar=double $(NVCCFLAGS) $(INCLUDE_FLAGS) $(GENCODE_FLAGS) $< -o $@ -c

$(EXECUTABLE_FLOAT): $(OBJECTS_FLOAT)
	$(CXX) $(CXXFLAGS) $(OBJECTS_FLOAT) -o $@ $(LINK_FLAGS)

$(EXECUTABLE_DOUBLE): $(OBJECTS_DOUBLE)
	$(CXX) $(CXXFLAGS) $(OBJECTS_DOUBLE) -o $@ $(LINK_FLAGS)


.PHONY: release
release: CXXFLAGS +=-O3 -ffast-math
release: NVCCFLAGS +=--compiler-options "-O3 -ffast-math"
release: $(EXECUTABLE_FLOAT) $(EXECUTABLE_DOUBLE)

.PHONY: debug
debug: CXXFLAGS +=-g
debug: NVCCFLAGS +=-g -G
debug: $(EXECUTABLE_FLOAT) $(EXECUTABLE_DOUBLE)

.PHONY: clean
clean:
	rm -rf $(EXECUTABLE_FLOAT) $(EXECUTABLE_DOUBLE) *.o

.PHONY: purge
purge: clean
	rm -rf *_CODE sm_version.mk
