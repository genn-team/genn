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

# OS name (Linux or Darwin) and architecture (32 bit or 64 bit)
OS_SIZE                 :=$(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")
OS_UPPER                :=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OS_LOWER                :=$(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
DARWIN                  :=$(strip $(findstring DARWIN,$(OS_UPPER)))

# Source, Include, Object and Library directories 
SRC_PATH                :=$(GENN_PATH)/lib/src
INC_PATH                :=$(GENN_PATH)/lib/include
OBJ_PATH                :=$(GENN_PATH)/lib/obj
LIB_PATH                :=$(GENN_PATH)/lib/lib
OUT_PATH                :=$(CURDIR)

# Global CUDA compiler settings
ifndef CPU_ONLY
    CUDA_PATH           ?=/usr/local/cuda
    NVCC                :="$(CUDA_PATH)/bin/nvcc"
endif

# Global C++ compiler settings
ifeq ($(DARWIN),DARWIN)
    CXX                 :=clang++
endif
ifndef CPU_ONLY
    CXXFLAGS            :=-std=c++0x -MMD -MP -DNVCC=\"$(NVCC)\"
else
    CXXFLAGS            :=-std=c++0x -MMD -MP -DCPU_ONLY
endif

# Global include and link flags
ARFLAGS                 :=-rcs
ifndef CPU_ONLY
    INCLUDE_FLAGS       :=-I"$(INC_PATH)" -I"$(CUDA_PATH)/include"
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS      :=-L"$(LIB_PATH)" -L"$(CUDA_PATH)/lib" -lgenn -lcuda -lcudart -lstdc++ -lc++
    else
        ifeq ($(OS_SIZE),32)
            LINK_FLAGS  :=-L"$(LIB_PATH)" -L"$(CUDA_PATH)/lib" -lgenn -lcuda -lcudart
        else
            LINK_FLAGS  :=-L"$(LIB_PATH)" -L"$(CUDA_PATH)/lib64" -lgenn -lcuda -lcudart
        endif
    endif
else
    INCLUDE_FLAGS       :=-I"$(INC_PATH)"
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS      :=-L"$(LIB_PATH)" -lgenn -lstdc++ -lc++
    endif
endif

# generateALL and libgenn.a object targets
GENERATE_OBJ            :=$(addprefix $(OBJ_PATH)/,generateALL.o generateRunner.o generateCPU.o generateKernels.o)
LIBGENN_OBJ             :=$(addprefix $(OBJ_PATH)/,global.o modelSpec.o neuronModels.o synapseModels.o postSynapseModels.o utils.o stringUtils.o sparseUtils.o hr_time.o)


# Target rules
.PHONY: all debug clean

all: $(OUT_PATH)/generateALL

$(LIB_PATH)/libgenn.a: $(LIBGENN_OBJ) | $(LIB_PATH)
	$(AR) $(ARFLAGS) $@ $(LIBGENN_OBJ)

$(OUT_PATH)/generateALL: $(GENERATE_OBJ) $(LIB_PATH)/libgenn.a
	$(CXX) $(CXXFLAGS) -o $@ $(GENERATE_OBJ) $(LINK_FLAGS)

$(OBJ_PATH)/generateALL.o: $(SRC_PATH)/generateALL.cc | $(OBJ_PATH)
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INCLUDE_FLAGS) -include "$(MODEL)"

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cc | $(OBJ_PATH)
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INCLUDE_FLAGS)

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp | $(OBJ_PATH)
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INCLUDE_FLAGS)

$(OBJ_PATH):
	mkdir -p $@

$(LIB_PATH):
	mkdir -p $@

debug: CXXFLAGS +=-g -O0
debug: all

clean:
	rm -rf $(OUT_PATH)/generateALL $(OBJ_PATH) $(LIB_PATH)

-include $(patsubst %.o,%.d,$(GENERATE_OBJ) $(LIBGENN_OBJ))
