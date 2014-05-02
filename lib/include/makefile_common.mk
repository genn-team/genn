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

##################################################################################
# Makefile include for all projects
##################################################################################

# OS name (Linux or Darwin) and architecture (32 bit or 64 bit).
OSUPPER 	:= $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER 	:= $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
DARWIN  	:= $(strip $(findstring DARWIN, $(OSUPPER)))
OS_SIZE 	:= $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH 	:= $(shell uname -m | sed -e "s/i386/i686/")

# C / C++ compiler, cuda compiler, include flags and link flags. Specify
# additional lib and include paths by defining LINK_FLAGS and INCLUDE_FLAGS in
# a project's main Makefile.  Declare cuda's install directory with CUDA_PATH.
CUDA_PATH        ?= /usr/local/cuda
COMPILER         ?= g++
NVCC             ?= $(CUDA_PATH)/bin/nvcc
INCLUDE_FLAGS    += -I$(CUDA_PATH)/include -I$(GeNNPATH)/lib/include -I. -include cuda_runtime.h
ifeq ($(DARWIN),DARWIN)
  LINK_FLAGS     += -Xlinker -L$(CUDA_PATH)/lib -lcudart 
else
  ifeq ($(OS_SIZE),32)
    LINK_FLAGS   += -L$(CUDA_PATH)/lib -lcudart 
  else
    LINK_FLAGS   += -L$(CUDA_PATH)/lib64 -lcudart 
  endif
endif

# Global compiler flags to be used by all projects. Declate CCFLAGS and NVCCFLAGS
# in a project's main Makefile to specify compiler flags on a per-project basis.
ifeq ($(DARWIN),DARWIN)
	CCFLAGS          += -arch i386# put your global compiler flags here
else
	CCFLAGS          += -O3 -ffast-math # put your global compiler flags here
endif
NVCCFLAGS        += --compiler-options "-O3 -ffast-math" # put your global nvcc flags here

# Get object targets from the files listed in SOURCES, also the GeNN code for each device.
# Define your own SOURCES variable in the project's Makefile to specify these source files.
OBJECTS          ?= $(foreach obj, $(SOURCES), $(obj).o)
GENN_CODE        ?= $(wildcard *_CODE_*)


#################################################################################
# Target rules
#################################################################################

.PHONY: all
all: release

.PHONY: $(GENN_CODE)
$(GENN_CODE):
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FLAGS) $(shell cat $@/sm_version) -o $@.o -c $@/runner.cc

%.cc.o: %.cc
	$(COMPILER) $(CCFLAGS) $(INCLUDE_FLAGS) -o $@ -c $<

%.cpp.o: %.cpp
	$(COMPILER) $(CCFLAGS) $(INCLUDE_FLAGS) -o $@ -c $<

$(EXECUTABLE): $(OBJECTS) $(GENN_CODE)
	$(NVCC) $(NVCCFLAGS) $(LINK_FLAGS) -o $@ $+

.PHONY: release
release: $(EXECUTABLE)
	mkdir -p $(ROOTDIR)/bin/$(OSLOWER)/release
	mv $(EXECUTABLE) $(ROOTDIR)/bin/$(OSLOWER)/release

.PHONY: debug
debug: CCFLAGS += -g
debug: NVCCFLAGS += -g -G
debug: $(EXECUTABLE)
	mkdir -p $(ROOTDIR)/bin/$(OSLOWER)/debug
	mv $(EXECUTABLE) $(ROOTDIR)/bin/$(OSLOWER)/debug

.PHONY: clean
clean:
	rm -rf $(ROOTDIR)/bin $(ROOTDIR)/*.o $(ROOTDIR)/*_CODE_*
