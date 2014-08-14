##--------------------------------------------------------------------------
##   Author: Thomas Nowotny
##
##   Institute: Center for Computational Neuroscience and Robotics
##              University of Sussex
##              Falmer, Brighton BN1 9QJ, UK
##
##   email to:  T.Nowotny@sussex.ac.uk
##
##   initial version: 2010-02-07
##
##--------------------------------------------------------------------------


# Makefile for the GeNN "generateALL" binary
# This is a UNIX Makefile, to be used by the GNU make build system
#-----------------------------------------------------------------

# OS name (Linux or Darwin) and architecture (32 bit or 64 bit).
OS_SIZE 	:=$(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_LOWER 	:=$(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
OS_UPPER 	:=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
DARWIN  	:=$(strip $(findstring DARWIN, $(OS_UPPER)))

# Global C++ / CUDA compiler settings and CUDA SDK directory.
CUDA_PATH	?=/usr/local/cuda
NVCC		?=$(CUDA_PATH)/bin/nvcc
NVCCFLAGS	+=
CXX		?=g++
CXXFLAGS	+=-DNVCC=\"$(NVCC)\"

# Global include flags and link flags.
INCLUDE_FLAGS	:=-I$(CUDA_PATH)/include -I$(GeNNPATH)/lib/include
ifeq ($(DARWIN),DARWIN)
  LINK_FLAGS	+=-Xlinker -L$(CUDA_PATH)/lib -lcudart -lstdc++ -lc++
else
  ifeq ($(OS_SIZE),32)
    LINK_FLAGS	+=-L$(CUDA_PATH)/lib -lcudart
  else
    LINK_FLAGS	+=-L$(CUDA_PATH)/lib64 -lcudart
  endif
endif

# Target rules.
.PHONY: all
all: bin/generateALL

.PHONY: debug
debug: CXXFLAGS +=-g
debug: all

bin/generateALL: src/generateALL.cc src/generateRunner.cc src/generateCPU.cc src/generateKernels.cc src/currentModel.cc src/CodeHelper.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $(LINK_FLAGS) -o bin/generateALL src/generateALL.cc

.PHONY: clean
clean:
	rm -rf bin/generateALL
