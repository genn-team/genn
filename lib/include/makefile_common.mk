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
ROOTDIR		?= $(CURDIR)
COMPILER	?= g++
CUDA_PATH	?= /usr/local/cuda
NVCC		?= $(CUDA_PATH)/bin/nvcc
INCLUDE_FLAGS	+= -I$(CUDA_PATH)/include -I$(GeNNPATH)/lib/include -I$(GeNNPATH)/userproject/include -I.
ifeq ($(DARWIN),DARWIN)
  LINK_FLAGS	+= -Xlinker -lstdc++ -lc++ -L$(CUDA_PATH)/lib -lcudart 
else
  ifeq ($(OS_SIZE),32)
    LINK_FLAGS	+= -L$(CUDA_PATH)/lib -lcudart
  else
    LINK_FLAGS	+= -L$(CUDA_PATH)/lib64 -lcudart
  endif
endif

# Global compiler flags to be used by all projects. Declate CCFLAGS and NVCCFLAGS
# in a project's main Makefile to specify compiler flags on a per-project basis.
ifeq ($(DARWIN),DARWIN)
  CCFLAGS       += # put your global compiler flags here
else
  CCFLAGS	+= # put your global C++ compiler flags here
endif
NVCCFLAGS       += # put your global NVCC flags here

# Get object targets from the files listed in SOURCES, also the GeNN code for each device.
# Define your own SOURCES variable in the project's Makefile to specify these source files.
USER_OBJECTS	?= $(patsubst %.cpp,%.o,$(patsubst %.cc,%.o,$(SOURCES)))
HOST_OBJECTS	?= $(patsubst %.cc,%.o,$(wildcard *_CODE_HOST/host.cc))
CUDA_OBJECTS	?= $(foreach obj,$(wildcard *_CODE_CUDA*/cuda*.cu),$(patsubst %.cu,%.o,$(obj)))


#################################################################################
#                                Target rules                                   #
#################################################################################

.PHONY: all
all: release

%.o: %.cc
	$(COMPILER) $(CCFLAGS) $(INCLUDE_FLAGS) -o $@ -c $<

%.o: %.cpp
	$(COMPILER) $(CCFLAGS) $(INCLUDE_FLAGS) -o $@ -c $<

$(HOST_OBJECTS):
	$(COMPILER) $(CCFLAGS) $(INCLUDE_FLAGS) -o $@ -c $(patsubst %.o,%.cc,$@)

$(CUDA_OBJECTS):
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FLAGS) -o $@ -c $(patsubst %.o,%.cu,$@) $(shell cat $(dir $@)/sm_version)

$(EXECUTABLE): $(USER_OBJECTS) $(HOST_OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(LINK_FLAGS) -o $@ $+

.PHONY: release
release: CCFLAGS += -O3 -ffast-math
release: NVCCFLAGS += --compiler-options "-O3 -ffast-math"
release: $(EXECUTABLE)
	mkdir -p "$(ROOTDIR)/bin/$(OSLOWER)/release"
	mv $(EXECUTABLE) $(ROOTDIR)/bin/$(OSLOWER)/release

.PHONY: debug
debug: CCFLAGS += -g
debug: NVCCFLAGS += -g -G
debug: $(EXECUTABLE)
	mkdir -p "$(ROOTDIR)/bin/$(OSLOWER)/debug"
	mv $(EXECUTABLE) $(ROOTDIR)/bin/$(OSLOWER)/debug

.PHONY: clean
clean:
	rm -rf $(ROOTDIR)/bin $(ROOTDIR)/*.o $(ROOTDIR)/*_CODE_*/*.o

.PHONY: purge
purge: clean
	rm -rf $(ROOTDIR)/*_CODE_*
