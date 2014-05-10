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

# OS name (Windows) and architecture (32 bit or 64 bit).
OSWIN		:= Win32
OSVAR		:= $(shell uname | tr A-Z a-z)
OS_SIZE 	:= $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH 	:= $(shell uname -m | sed -e "s/i386/i686/")

# C / C++ compiler, cuda compiler, include flags and link flags. Specify
# additional lib and include paths by defining LINK_FLAGS and INCLUDE_FLAGS in
# a project's main Makefile.  Declare cuda's install directory with CUDA_PATH.
ROOTDIR		?= $(CURDIR)
COMPILER	?= cl
CUDA_PATH	?= "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\ "
NVCC		?= $(CUDA_PATH)\bin\nvcc
INCLUDE_FLAGS 	+= -I"$(shell cygpath -m '$(CUDA_PATH)')include" -I"$(GeNNPATH)/lib/include" -I"$(GeNNPATH)/lib/include/numlib" -I.
LLIB 		:= "$(shell cygpath -m '$(CUDA_PATH)')lib/$(OSWIN)/cudart.lib"
LINK_FLAGS   	+= -L"$(shell cygpath -m '$(CUDA_PATH)')lib" -lcudart 

# Global compiler flags to be used by all projects. Declate CCFLAGS and NVCCFLAGS
# in a project's main Makefile to specify compiler flags on a per-project basis.
CCFLAGS          += # put your global C++ compiler flags here
NVCCFLAGS        += -idp /cygwin/ --machine 32 # put your global NVCC flags here

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
	$(COMPILER) $(CCFLAGS) $(INCLUDE_FLAGS) /Fe$@ -c $<

%.o: %.cpp
	$(COMPILER) $(CCFLAGS) $(INCLUDE_FLAGS) $< /c /Fo$@

$(HOST_OBJECTS):
	$(COMPILER) $(CCFLAGS) $(INCLUDE_FLAGS) -o $@ -c $(patsubst %.o,%.cc,$@)

$(CUDA_OBJECTS):
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FLAGS) -o $@ -c $(patsubst %.o,%.cu,$@) $(shell cat $(dir $@)/sm_version)

$(EXECUTABLE): $(USER_OBJECTS) $(HOST_OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(LINK_FLAGS) -o $@ $+

.PHONY: release
release: $(EXECUTABLE)
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
	rm -rf $(ROOTDIR)/bin $(ROOTDIR)/*.o $(ROOTDIR)/*_CODE_*/*.o

.PHONY: purge
purge: clean
	rm -rf $(ROOTDIR)/*_CODE_*
