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


# Makefile include for all GeNN example projects
# This is a UNIX Makefile, to be used by the GNU make build system
#-----------------------------------------------------------------

# OS name (Linux or Darwin) and architecture (32 bit or 64 bit)
OS_SIZE                 :=$(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")
OS_UPPER                :=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OS_LOWER                :=$(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
DARWIN                  :=$(strip $(findstring DARWIN,$(OS_UPPER)))

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
    CXXFLAGS            +=-std=c++0x
else
    CXXFLAGS            +=-std=c++0x -DCPU_ONLY
endif

# Global include and link flags
ifndef CPU_ONLY
    INCLUDE_FLAGS       +=-I"$(GENN_PATH)/lib/include" -I"$(GENN_PATH)/userproject/include" -I"$(CUDA_PATH)/include"
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS      +=-L"$(GENN_PATH)/lib/lib" -L"$(CUDA_PATH)/lib" -lgenn -lcuda -lcudart -lstdc++ -lc++
    else
        ifeq ($(OS_SIZE),32)
            LINK_FLAGS  +=-L"$(GENN_PATH)/lib/lib" -L"$(CUDA_PATH)/lib" -lgenn -lcuda -lcudart
        else
            LINK_FLAGS  +=-L"$(GENN_PATH)/lib/lib" -L"$(CUDA_PATH)/lib64" -lgenn -lcuda -lcudart
        endif
    endif
else
    INCLUDE_FLAGS       +=-I"$(GENN_PATH)/lib/include" -I"$(GENN_PATH)/userproject/include"
    LINK_FLAGS          +=-L"$(GENN_PATH)/lib/lib" -lgenn
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS      +=-L"$(GENN_PATH)/lib/lib" -lgenn -lstdc++ -lc++
    endif
endif

# An auto-generated file containing your cuda device's compute capability
-include sm_version.mk

# Enumerate all object files (if they have not already been listed)
SIM_CODE                ?=*_CODE # Can be changed by passing SIM_CODE=something in the make command
SOURCES                 ?=$(wildcard *.cc *.cpp *.cu)
OBJECTS                 :=$(foreach obj,$(basename $(SOURCES)),$(obj).o) $(SIM_CODE)/runner.o


# Target rules
.PHONY: all release debug clean purge show

all: release

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS) $(LINK_FLAGS)

$(SIM_CODE)/runner.o:
	cd $(SIM_CODE) && make

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE_FLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE_FLAGS)

ifndef CPU_ONLY
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE_FLAGS)
endif

release: NVCCFLAGS      +=$(NVCC_OPTIMIZATIONFLAGS) -Xcompiler "$(OPTIMIZATIONFLAGS)"
release: CXXFLAGS       +=$(OPTIMIZATIONFLAGS)
release: $(EXECUTABLE)

debug: NVCCFLAGS        +=-g -G
debug: CXXFLAGS         +=-g
debug: $(EXECUTABLE)

clean:
	rm -rf $(EXECUTABLE) *.o *.dSYM/ generateALL
	cd $(SIM_CODE) && make clean

purge: clean
	rm -rf $(SIM_CODE) sm_version.mk

show:
	echo $(OBJECTS)
