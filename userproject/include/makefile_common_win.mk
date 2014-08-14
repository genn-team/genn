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
# This is a Windows Makefile, to be used by the MS nmake build system
#--------------------------------------------------------------------





# REPLACE ALL BELOW VVVVVVVVVVVVVVVVVVVVVVV


# OS name (Windows) and architecture (32 bit or 64 bit).
OSWIN		:= Win32
OSVAR		:= $(shell uname | tr A-Z a-z)
OS_SIZE 	:= $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH 	:= $(shell uname -m | sed -e "s/i386/i686/")

# C / C++ compiler, cuda compiler, include flags and link flags. Specify
# additional lib and include paths by defining LINK_FLAGS and INCLUDE_FLAGS in
# a project's main Makefile.  Declare cuda's install directory with CUDA_PATH.
CUDA_PATH	?= "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\ "
COMPILER	?= cl
NVCC		?= $(CUDA_PATH)\bin\nvcc
INCLUDE_FLAGS 	+= -I"$(shell cygpath -m '$(CUDA_PATH)')include" -I"$(GeNNPATH)/lib/include" -I"$(GeNNPATH)/lib/include/numlib" -I"$(GeNNPATH)/userproject/include" -I.

LLIB 		:= "$(shell cygpath -m '$(CUDA_PATH)')lib/$(OSWIN)/cudart.lib"
LINK_FLAGS   	+= -L"$(shell cygpath -m '$(CUDA_PATH)')lib" -lcudart 

# An auto-generated file containing your cuda device's compute capability.
# The appropriate -gencode flag is added to NVCCFLAGS (if it exists yet).
-include $(GeNNPATH)/lib/src/sm_version.mk

# Global compiler flags to be used by all projects. Declate CCFLAGS and NVCCFLAGS
# in a project's main Makefile to specify compiler flags on a per-project basis.
CCFLAGS          += # put your global compiler flags here
NVCCFLAGS        += -idp /cygwin/ --machine 32 # put your global nvcc flags here

# Get the OBJECTS rule targets from the files listed by SOURCES (use all source
# files in a project's root directory by default). Define your own SOURCES
# variable in the project's Makefile to specify main source files explicitly.
SOURCES          ?= $(wildcard *.cc *.cpp *.cu)
OBJECTS          ?= $(foreach obj, $(SOURCES), obj/$(obj).o)


#################################################################################
# Target rules
#################################################################################

.PHONY: all
all: release

obj/%.cc.o: %.cc
	mkdir -p "$(ROOTDIR)/obj"
	$(COMPILER) $(CCFLAGS) $(INCLUDE_FLAGS)  /Fe$@ -c $<

obj/%.cpp.o: %.cpp
	mkdir -p "$(ROOTDIR)/obj"
	$(COMPILER) $(CCFLAGS) $(INCLUDE_FLAGS) $< /c /Fo$@

obj/%.cu.o: %.cu
	mkdir -p "$(ROOTDIR)/obj"
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDE_FLAGS) -o $@ -c $< #/Fe$@ -c $<

$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $+ $(LINK_FLAGS) $(LLIB)

.PHONY: release
release: CCFLAGS += -O3 -ffast-math
release: NVCCFLAGS += --compiler-options "-O3 -ffast-math"
release: $(EXECUTABLE)
	echo "WARNING CYGWIN SUPPORT IS AN EXPERIMENTAL FEATURE AND MAY NOT WORK AS INTENDED" 
	mkdir -p "$(ROOTDIR)/bin/$(OSVAR)/release"
	mv $(EXECUTABLE) $(ROOTDIR)/bin/$(OSVAR)/release

.PHONY: debug
debug: CCFLAGS += -g
debug: NVCCFLAGS += -g -G
debug: $(EXECUTABLE)
	mkdir -p "$(ROOTDIR)/bin/$(OSVAR)/debug"
	mv $(EXECUTABLE) $(ROOTDIR)/bin/$(OSVAR)/debug

.PHONY: clean
clean:
	rm -rf $(ROOTDIR)/bin $(ROOTDIR)/obj

.PHONY: purge
purge: clean
	rm -rf $(ROOTDIR)/*_CODE sm_Version.mk currentModel.cc
