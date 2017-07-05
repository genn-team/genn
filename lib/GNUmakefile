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

# OS name (Linux or Darwin) and kernel architecture (32 bit or 64 bit)
OS_SIZE                  :=$(shell getconf LONG_BIT)
OS_UPPER                 :=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OS_LOWER                 :=$(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
DARWIN                   :=$(strip $(findstring DARWIN,$(OS_UPPER)))

# Source, Include, Object and Library directories
SRC_PATH                 :=$(GENN_PATH)/lib/src
INC_PATH                 :=$(GENN_PATH)/lib/include
GENERATEALL_PATH         :=$(CURDIR)
LIBGENN_PATH             :=$(GENN_PATH)/lib/lib
ifndef CPU_ONLY
    LIBGENN_OBJ_PATH     :=$(GENN_PATH)/lib/obj
else
    LIBGENN_OBJ_PATH     :=$(GENN_PATH)/lib/obj_CPU_ONLY
endif

# generateALL and libgenn.a targets
ifndef CPU_ONLY
    GENERATEALL          :=$(GENERATEALL_PATH)/generateALL
    LIBGENN              :=$(LIBGENN_PATH)/libgenn.a
else
    GENERATEALL          :=$(GENERATEALL_PATH)/generateALL_CPU_ONLY
    LIBGENN              :=$(LIBGENN_PATH)/libgenn_CPU_ONLY.a
endif
LIBGENN_OBJ              :=global.o modelSpec.o neuronGroup.o synapseGroup.o neuronModels.o synapseModels.o postSynapseModels.o utils.o codeStream.o codeGenUtils.o sparseUtils.o hr_time.o newNeuronModels.o newPostsynapticModels.o newWeightUpdateModels.o standardSubstitutions.o standardGeneratedSections.o
LIBGENN_OBJ              :=$(addprefix $(LIBGENN_OBJ_PATH)/,$(LIBGENN_OBJ))

# Global CUDA compiler settings
ifndef CPU_ONLY
    CUDA_PATH            ?=/usr/local/cuda
    NVCC                 :="$(CUDA_PATH)/bin/nvcc"
endif

# Global C++ compiler settings
ifeq ($(DARWIN),DARWIN)
    CXX                  :=clang++
endif
ifndef CPU_ONLY
    CXXFLAGS             :=-std=c++11 -Wall -Wextra -DNVCC=\"$(NVCC)\"
else
    CXXFLAGS             :=-std=c++11 -Wall -Wextra -DCPU_ONLY
endif
ifdef DEBUG
    CXXFLAGS             +=-g -O0 -DDEBUG
endif
ifdef MPI_ENABLE
    CXXFLAGS             +=-DMPI_ENABLE
endif

ARFLAGS                  :=-rcs

# Global include and link flags
ifndef CPU_ONLY
    INCLUDE_FLAGS        :=-I"$(INC_PATH)" -I"$(CUDA_PATH)/include"
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS       :=-rpath $(CUDA_PATH)/lib -L"$(LIBGENN_PATH)" -L"$(CUDA_PATH)/lib" -lgenn -lcuda -lcudart -lstdc++ -lc++
    else
        ifeq ($(OS_SIZE),32)
            LINK_FLAGS   :=-L"$(LIBGENN_PATH)" -L"$(CUDA_PATH)/lib" -lgenn -lcuda -lcudart
        else
            LINK_FLAGS   :=-L"$(LIBGENN_PATH)" -L"$(CUDA_PATH)/lib64" -lgenn -lcuda -lcudart
        endif
    endif
else
    INCLUDE_FLAGS        :=-I"$(INC_PATH)"
    LINK_FLAGS           :=-L"$(LIBGENN_PATH)" -lgenn_CPU_ONLY
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS       :=-L"$(LIBGENN_PATH)" -lgenn_CPU_ONLY -lstdc++ -lc++
    endif
endif

ifdef COVERAGE
    CXXFLAGS             +=-O0 --coverage
    LINK_FLAGS           +=--coverage
endif

# Target rules
.PHONY: all clean clean_generateall clean_libgenn always_compile

all: $(GENERATEALL)

$(GENERATEALL): $(LIBGENN) always_compile
	$(CXX) $(CXXFLAGS) -DMODEL=\"$(MODEL)\" -o $@ $(SRC_PATH)/generate*.cc $(INCLUDE_FLAGS) $(LINK_FLAGS)

$(LIBGENN): $(LIBGENN_OBJ_PATH) $(LIBGENN_OBJ) $(LIBGENN_PATH)
	$(AR) $(ARFLAGS) $@ $(LIBGENN_OBJ)

$(LIBGENN_OBJ_PATH)/%.o: $(SRC_PATH)/%.cc
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $< $(INCLUDE_FLAGS)

$(LIBGENN_OBJ_PATH):
	mkdir -p $@

$(LIBGENN_PATH):
	mkdir -p $@

clean: clean_generateall clean_libgenn

clean_generateall:
	rm -f $(GENERATEALL) $(GENERATEALL).d

clean_libgenn:
	rm -rf $(LIBGENN_OBJ_PATH) $(LIBGENN_PATH)

always_compile:

-include $(patsubst %.o,%.d,$(LIBGENN_OBJ))
