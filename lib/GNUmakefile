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

SRC_PATH                :=$(GENN_PATH)/lib/src
INC_PATH                :=$(GENN_PATH)/lib/include
OBJ_PATH                :=$(GENN_PATH)/lib/obj

# OS name (Linux or Darwin) and architecture (32 bit or 64 bit)
OS_SIZE                 :=$(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")
OS_UPPER                :=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OS_LOWER                :=$(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
DARWIN                  :=$(strip $(findstring DARWIN, $(OS_UPPER)))

# Global CUDA compiler settings
ifndef CPU_ONLY
    CUDA_PATH           ?=/usr/local/cuda
    NVCC                :=$(CUDA_PATH)/bin/nvcc
endif

# Global C++ compiler settings
ifeq ($(DARWIN),DARWIN)
    CXX                 :=clang++
endif
ifndef CPU_ONLY
    CXXFLAGS            :=-std=c++0x -DNVCC=\"$(NVCC)\"
else
    CXXFLAGS            :=-std=c++0x -DCPU_ONLY
endif

# Global include and link flags
ifndef CPU_ONLY
    INCLUDE_FLAGS       :=-I"$(GENN_INC)" -I"$(CUDA_PATH)/include" -include "$(MODEL)"
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS      :=-L"$(GENN_PATH)/lib/lib" -L"$(CUDA_PATH)/lib" -lcuda -lcudart -lstdc++ -lc++
    else
        ifeq ($(OS_SIZE),32)
            LINK_FLAGS  :=-L"$(GENN_PATH)/lib/lib" -L"$(CUDA_PATH)/lib" -lcuda -lcudart
        else
            LINK_FLAGS  :=-L"$(GENN_PATH)/lib/lib" -L"$(CUDA_PATH)/lib64" -lcuda -lcudart
        endif
    endif
else
    INCLUDE_FLAGS       :=-I"$(GENN_INC)" -include "$(MODEL)"
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS      :=-L"$(OBJ_PATH)" -lstdc++ -lc++
    endif
endif

# Target rules
.PHONY: all
all: generateALL

generateALL: "$(OBJ_PATH)/generateALL.o" "$(OBJ_PATH)/libgenn.a"
	$(CXX) $(CXXFLAGS) $(LINK_FLAGS) $< -o $@

generateALL.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $< -o $@

.PHONY: debug
debug: CXXFLAGS +=-g
debug: all

.PHONY: clean
clean:
	rm -f generateALL
