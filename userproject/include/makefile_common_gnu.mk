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

# OS name (Linux or Darwin) and architecture (32 bit or 64 bit)
OS_SIZE			:=$(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")
OS_UPPER 		:=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OS_LOWER 		:=$(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
DARWIN			:=$(strip $(findstring DARWIN, $(OS_UPPER)))

# Global CUDA compiler settings
ifndef CPU_ONLY
    CUDA_PATH		?=/usr/local/cuda
    NVCC		:=$(CUDA_PATH)/bin/nvcc
endif

# Global C++ compiler settings
ifeq ($(DARWIN),DARWIN)
    CXX			:=clang++
endif
ifndef CPU_ONLY
    CXXFLAGS		+=-std=c++0x
else
    CXXFLAGS		+=-std=c++0x -DCPU_ONLY
endif

# Global include and link flags
ifndef CPU_ONLY
    INCLUDE_FLAGS	+=-I$(GENN_PATH)/lib/include -I$(GENN_PATH)/userproject/include -I$(CUDA_PATH)/include
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS	+=-Xlinker -lstdc++ -lc++ -L$(CUDA_PATH)/lib -lcuda -lcudart
    else
        ifeq ($(OS_SIZE),32)
            LINK_FLAGS	+=-L$(CUDA_PATH)/lib -lcuda -lcudart
        else
            LINK_FLAGS	+=-L$(CUDA_PATH)/lib64 -lcuda -lcudart
        endif
    endif
else
    INCLUDE_FLAGS	+=-I$(GENN_PATH)/lib/include -I$(GENN_PATH)/userproject/include
    ifeq ($(DARWIN),DARWIN)
        LINK_FLAGS	+=-Xlinker -lstdc++ -lc++
    endif
endif

# An auto-generated file containing your cuda device's compute capability
-include sm_version.mk

# Enumerate all source and object files (if they have not already been listed)
SOURCES		?=$(wildcard *.cc *.cpp *.cu)
OBJECTS		:=$(foreach obj,$(basename $(SOURCES)),$(obj).o)

# Target rules
.PHONY: all
all: release

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $< -o $@ -c

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $< -o $@ -c

ifndef CPU_ONLY
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FLAGS) $(GENCODE_FLAGS) $< -o $@ -c
endif

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LINK_FLAGS)

.PHONY: release
# use --compiler-options "-Wconversion" for silent type conversions
release: NVCCFLAGS	+=$(NVCC_OPTIMIZATIONFLAGS) --compiler-options "$(OPTIMIZATIONFLAGS)"
release: CXXFLAGS	+=$(OPTIMIZATIONFLAGS)
release: $(EXECUTABLE)

.PHONY: debug
debug: NVCCFLAGS	+=-g -G
debug: CXXFLAGS		+=-g
debug: $(EXECUTABLE)

.PHONY: clean
clean:
	rm -rf $(EXECUTABLE) *.o *.dSYM/ generateALL runner.cubin

.PHONY: purge
purge: clean
	rm -rf *_CODE sm_version.mk 

.PHONY: show
show:
	echo $(OBJECTS)
