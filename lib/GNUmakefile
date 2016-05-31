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


# Makefile for the GeNN "generateALL" and "libgenn.a" binaries
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
    CXXFLAGS            :=-std=c++11 -MMD -MP -DNVCC=\"$(NVCC)\"
else
    CXXFLAGS            :=-std=c++11 -MMD -MP -DCPU_ONLY
endif
ifdef DEBUG
    CXXFLAGS            +=-g -O0 -DDEBUG
endif
ARFLAGS                 :=-rcs

# Global include and link flags
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
    LINK_FLAGS          :=-L"$(LIB_PATH)" -lgenn
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS      :=-L"$(LIB_PATH)" -lgenn -lstdc++ -lc++
    endif
endif

# generateALL and libgenn.a object targets
GENERATEALL_OBJ         :=$(addprefix $(OUT_PATH)/,generateALL.o generateRunner.o generateCPU.o generateKernels.o)
LIBGENN_OBJ             :=$(addprefix $(OBJ_PATH)/,global.o modelSpec.o neuronModels.o synapseModels.o postSynapseModels.o utils.o stringUtils.o sparseUtils.o hr_time.o)


# Target rules
.PHONY: all clean clean_generateall clean_libgenn

all: $(OUT_PATH)/generateALL

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INCLUDE_FLAGS)

$(OUT_PATH)/%.o: $(SRC_PATH)/%.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INCLUDE_FLAGS)

$(LIB_PATH)/libgenn.a: $(OBJ_PATH) $(LIB_PATH) $(LIBGENN_OBJ)
	$(AR) $(ARFLAGS) $@ $(LIBGENN_OBJ)

$(OUT_PATH)/generateALL: $(GENERATEALL_OBJ) $(LIB_PATH)/libgenn.a
	$(CXX) $(CXXFLAGS) -o $@ $(GENERATEALL_OBJ) $(LINK_FLAGS)

$(OUT_PATH)/generateALL.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $(SRC_PATH)/generateALL.cc $(INCLUDE_FLAGS) -include "$(MODEL)"

$(OBJ_PATH):
	mkdir -p $@

$(LIB_PATH):
	mkdir -p $@

clean: clean_generateall clean_libgenn

clean_generateall:
	rm -f $(OUT_PATH)/generateALL $(GENERATEALL_OBJ) $(patsubst %.o,%.d,$(GENERATEALL_OBJ))

clean_libgenn:
	rm -rf $(OBJ_PATH) $(LIB_PATH)

-include $(patsubst %.o,%.d,$(GENERATEALL_OBJ) $(LIBGENN_OBJ))
