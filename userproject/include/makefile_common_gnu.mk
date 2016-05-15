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
SOURCES                 ?=$(wildcard *.cc *.cpp *.cu)
OBJECTS                 :=$(foreach obj,$(basename $(SOURCES)),$(obj).o) *_CODE/runner.o


# Target rules
.PHONY: all
all: release

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LINK_FLAGS) -o $@

*_CODE/runner.o:
	cd *_CODE && make

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $< -o $@

ifndef CPU_ONLY
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FLAGS) -c $< -o $@
endif

.PHONY: release
release: NVCCFLAGS      +=$(NVCC_OPTIMIZATIONFLAGS) -Xcompiler "$(OPTIMIZATIONFLAGS)"
release: CXXFLAGS       +=$(OPTIMIZATIONFLAGS)
release: $(EXECUTABLE)

.PHONY: debug
debug: NVCCFLAGS        +=-g -G
debug: CXXFLAGS         +=-g
debug: $(EXECUTABLE)

.PHONY: clean
clean:
	rm -rf $(EXECUTABLE) *.o *.dSYM/ generateALL
	cd *_CODE && make clean

.PHONY: purge
purge: clean
	rm -rf *_CODE sm_version.mk

.PHONY: show
show:
	echo $(OBJECTS)
