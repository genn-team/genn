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

# OS name (Linux or Darwin) and kernel architecture (32 bit or 64 bit)
OS_SIZE                  :=$(shell getconf LONG_BIT)
OS_UPPER                :=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OS_LOWER                :=$(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
DARWIN                  :=$(strip $(findstring DARWIN,$(OS_UPPER)))

# Global CUDA compiler settings
ifndef CPU_ONLY
    CUDA_PATH           ?=/usr/local/cuda
    NVCC                :="$(CUDA_PATH)/bin/nvcc"
endif
ifdef DEBUG
    NVCCFLAGS           +=-g -G
else
    NVCCFLAGS           +=$(NVCC_OPTIMIZATIONFLAGS) -Xcompiler "$(OPTIMIZATIONFLAGS)"
endif

# Global C++ compiler settings
ifeq ($(DARWIN),DARWIN)
    CXX                 :=clang++
endif
ifndef CPU_ONLY
    CXXFLAGS            +=-std=c++11
else
    CXXFLAGS            +=-std=c++11 -DCPU_ONLY
endif
ifdef DEBUG
    CXXFLAGS            +=-g -O0 -DDEBUG
else
    CXXFLAGS            +=$(OPTIMIZATIONFLAGS)
endif

# Global include and link flags
ifndef CPU_ONLY
    INCLUDE_FLAGS       +=-I"$(GENN_PATH)/lib/include" -I"$(GENN_PATH)/userproject/include" -I"$(CUDA_PATH)/include"
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS      +=-rpath $(CUDA_PATH)/lib -L"$(GENN_PATH)/lib/lib" -L"$(CUDA_PATH)/lib" -lgenn -lcuda -lcudart -lstdc++ -lc++
    else
        ifeq ($(OS_SIZE),32)
            LINK_FLAGS  +=-L"$(GENN_PATH)/lib/lib" -L"$(CUDA_PATH)/lib" -lgenn -lcuda -lcudart
        else
            LINK_FLAGS  +=-L"$(GENN_PATH)/lib/lib" -L"$(CUDA_PATH)/lib64" -lgenn -lcuda -lcudart
        endif
    endif
else
    INCLUDE_FLAGS       +=-I"$(GENN_PATH)/lib/include" -I"$(GENN_PATH)/userproject/include"
    LINK_FLAGS          +=-L"$(GENN_PATH)/lib/lib" -lgenn_CPU_ONLY
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS      +=-L"$(GENN_PATH)/lib/lib" -lgenn_CPU_ONLY -lstdc++ -lc++
    endif
endif

# An auto-generated file containing your cuda device's compute capability
-include sm_version.mk

# Enumerate all object files (if they have not already been listed)
ifndef SIM_CODE
    $(warning SIM_CODE=<model>_CODE was not defined in the Makefile or make command.)
    $(warning Using wildcard SIM_CODE=*_CODE.)
    SIM_CODE            :=*_CODE
endif
SOURCES                 ?=$(wildcard *.cc *.cpp *.cu *.c)
OBJECTS                 :=$(foreach obj,$(basename $(SOURCES)),$(obj).o) $(SIM_CODE)/runner.o


# Target rules
.PHONY: all clean purge show

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS) $(LINK_FLAGS)

$(SIM_CODE)/runner.o:
	cd $(SIM_CODE) && make

%.o: %.c
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INCLUDE_FLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INCLUDE_FLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INCLUDE_FLAGS)

ifndef CPU_ONLY
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $< $(INCLUDE_FLAGS)
endif

clean:
	rm -rf $(EXECUTABLE) *.o *.dSYM/ generateALL
	cd $(SIM_CODE) && make clean

purge: clean
	rm -rf $(SIM_CODE) sm_version.mk

show:
	echo $(OBJECTS)
