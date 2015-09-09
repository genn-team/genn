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

# OS name (Linux or Darwin) and architecture (32 bit or 64 bit).
OS_SIZE 	:=$(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_LOWER 	:=$(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
OS_UPPER 	:=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
DARWIN  	:=$(strip $(findstring DARWIN, $(OS_UPPER)))

# Global C++ and CUDA compiler settings.
NVCC		:=$(CUDA_PATH)/bin/nvcc
CXXFLAGS	+=-DNVCC=\"$(NVCC)\" -std=c++0x

# Global include flags and link flags.
INCLUDE_FLAGS	:=-include $(MODEL) -I$(CUDA_PATH)/include -I$(GENN_PATH)/lib/include -I$(GENN_PATH)/userproject/include
ifeq ($(DARWIN),DARWIN)
  LINK_FLAGS	:=-Xlinker -L$(CUDA_PATH)/lib -lcuda -lcudart -lstdc++ -lc++
else
  ifeq ($(OS_SIZE),32)
    LINK_FLAGS	:=-L$(CUDA_PATH)/lib -lcuda -lcudart -lstdc++ 
  else
    LINK_FLAGS	:=-L$(CUDA_PATH)/lib64 -lcuda -lcudart -lstdc++
  endif
endif

# Locations of src, obj and bin directories.
SRCDIR		:=src
OBJDIR		:=obj
BINDIR		:=bin

# Target rules.
.PHONY: all
all: $(BINDIR)/generateALL

$(OBJDIR)/generateALL.o: $(SRCDIR)/generateALL.cc $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $(EXTRA_DEF) $(SRCDIR)/generateALL.cc -o $(OBJDIR)/generateALL.o -c

$(BINDIR)/generateALL: $(OBJDIR)/generateALL.o
	$(CXX) $(CXXFLAGS) $(OBJDIR)/generateALL.o -o $(BINDIR)/generateALL $(LINK_FLAGS) 

.PHONY: debug
debug: CXXFLAGS +=-g
debug: all

$(OBJDIR):
	mkdir -p $(OBJDIR)

.PHONY: clean
clean:
	rm -rf $(OBJDIR) $(BINDIR)/generateALL
