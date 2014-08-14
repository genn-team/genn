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

# OS name (Linux or Darwin) and architecture (32 bit or 64 bit).
OS_SIZE		:= $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OSUPPER 	:= $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER 	:= $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
DARWIN  	:= $(strip $(findstring DARWIN, $(OSUPPER)))

# Global C++ / CUDA compiler settings and CUDA SDK directory.
CUDA_PATH	?=/usr/local/cuda
NVCC		?=$(CUDA_PATH)/bin/nvcc
NVCCFLAGS	+=
CXX		?=g++
CXXFLAGS	+=

# Global include flags and link flags.
INCLUDE_FLAGS	+=-I$(CUDA_PATH)/include -I$(GeNNPATH)/lib/include -I$(GeNNPATH)/userproject/include
ifeq ($(DARWIN),DARWIN)
  LINK_FLAGS	+=-Xlinker -L$(CUDA_PATH)/lib -lcudart -lstdc++ -lc++
else
  ifeq ($(OS_SIZE),32)
    LINK_FLAGS	+=-L$(CUDA_PATH)/lib -lcudart 
  else
    LINK_FLAGS	+=-L$(CUDA_PATH)/lib64 -lcudart 
  endif
endif

# An auto-generated file containing your cuda device's compute version.
-include sm_version.mk

# Locations of src, obj and bin directories.
SRCDIR		:=.
OBJDIR		:=obj
BINDIR		:=bin

# Enumerate all source and object files (if they have not already been listed).
SOURCES		?=$(wildcard $(SRCDIR)/*.cc $(SRCDIR)/*.cpp $(SRCDIR)/*.cu)
OBJECTS		?=$(foreach obj, $(SOURCES), $(OBJDIR)/$(obj).o)

# Target rules.
.PHONY: all
all: release

$(OBJDIR)/%.cc.o: $(SRCDIR)/%.cc $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -o $@ -c $<

$(OBJDIR)/%.cpp.o: $(SRCDIR)/%.cpp $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -o $@ -c $<

$(OBJDIR)/%.cu.o: $(SRCDIR)/%.cu $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDE_FLAGS) -o $@ -c $<

$(BINDIR)/$(EXECUTABLE): $(OBJECTS) $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJECTS) $(LINK_FLAGS) 

.PHONY: release
release: CXXFLAGS +=-O3 -ffast-math
release: NVCCFLAGS +=--compiler-options "-O3 -ffast-math"
release: $(BINDIR)/$(EXECUTABLE)

.PHONY: debug
debug: CXXFLAGS +=-g
debug: NVCCFLAGS +=-g -G
debug: $(BINDIR)/$(EXECUTABLE)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

.PHONY: clean
clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: purge
purge: clean
	rm -rf ./*_CODE
