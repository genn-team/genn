##--------------------------------------------------------------------------
##   Author: Thomas Nowotny
##  
##   Institute: Center for Computational Neuroscience and Robotics
##              University of Sussex
##	      Falmer, Brighton BN1 9QJ, UK 
##  
##   email to:  T.Nowotny@sussex.ac.uk
##  
##   initial version: 2010-02-07
##  
##--------------------------------------------------------------------------
##
## parts of the source code obtained and modified from NVIDIA developer 
## kit under the condtions below:

################################################################################
#
# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Build script for project
#
################################################################################

verbose=1
ifdef ON_WINDOWS
export compiler-bindir := ~/vs_bin/bin
endif

.SUFFIXES : .cu .cu_dbg.o .c_dbg.o .cpp_dbg.o .cu_rel.o .c_rel.o .cpp_rel.o .cubin .ptx
# Add new SM Versions here as devices with new Compute Capability are released
#SM_VERSIONS := sm_10 sm_11 sm_12 sm_13 sm_20 sm_21

CUDA_INSTALL_PATH ?= $(CUDA_PATH)

#ROOTDIR		:= $(NVIDIASDKPATH)/C/src
ROOTDIR		:= $(NVIDIASDKPATH)/common/inc
ROOTBINDIR      := .

# TODO: detect OS : in cygwin you may change this to Win32 by hand because cygwin messes it up
ifdef ON_WINDOWS
	OSUPPER = WIN32
	OSLOWER = win32	
	OSREAL = Win32
else
	OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
	OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
	OSREAL = $(OSLOWER)
        # 'linux' is output for Linux system, 'darwin' for OS X
	DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
	ifneq "$(DARWIN)" ""
		ON_MAC = 1
	endif
endif

# Basic directory setup for SDK
CUDADIR    := $(CUDA_INSTALL_PATH)
#SDKDIR     := $(NVIDIASAMPPATH)
NVIDIASAMPPATH := $(NVIDIASDKPATH)/common/inc
SDKDIR     := $(NVIDIASDKPATH)
BINDIR     := bin/$(OSREAL)
ROOTOBJDIR := obj
CUDAINC := "$(CUDADIR)/include"
CUDASDKINC := "$(SDKDIR)/common/inc"
GeNNPATH_WIN := C:\cygwin\home\esin\genn-0.3\genn
ifdef ON_MAC
	NVIDIASAMPLIB := $(NVIDIASDKPATH)/common/lib/$(OSLOWER)
else
	NVIDIASAMPLIB := $(NVIDIASDKPATH)/common/lib/$(OSLOWER)/x86_64
endif

# -L"$(cygpath -u SDKDIR)/lib" -L"$(cygpath -u SDKDIR)/common/lib/$(OSLOWER)"
# Includes
INCLUDES  += -I. -I $(CUDAINC) -I $(CUDASDKINC) -I$(GeNNPATH)/lib/include/numlib -I"$(GeNNPATH)/lib/include" 
#-I"$(GeNNPATH_WIN)\lib\include" -I"$(GeNNPATH_WIN)\lib\include\numlib"  -I../include
ifdef ON_WINDOWS
	INCLUDES_UNIX  += -I. -I" $(CUDADIR)/include" -I "$(NVIDIASAMPPATH)/common/inc" $(INCLUDES) #-I$(GeNNPATH)/lib/include/numlib
else
	INCLUDES_UNIX  += $(INCLUDES)
endif
# Compilers
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc -lcuda -lcudart
ifdef ON_WINDOWS
	CXX        := cl #-fPIC -m32 #g++
	CC         := cl #-fPIC -m32 #gcc
	LINK       := cl #-fPIC -m32 #g++
else ifdef ON_MAC
	CXX        := g++ -fPIC -arch i386 #-m32 #g++
	CC         := gcc -fPIC -arch i386  #-m32 #gcc
	LINK       := g++ -fPIC -arch i386  #-m32 #g++
else
	CXX        := g++ -fPIC #-m32 #g++
	CC         := gcc -fPIC #-m32 #gcc
	LINK       := g++ -fPIC #-m32 #g++
endif
# architecture flag for cubin build
CUBIN_ARCH_FLAG := #-m32 
#SMVERSIONFLAGS := -arch sm_20
include $(GeNNPATH)/lib/src/sm_Version.mk
#SMVERSIONFLAGS := -arch sm_13

# Compiler-specific flags
ifdef ON_MAC
	NVCCFLAGS := #-idp /cygwin/ --use-local-env --cl-version 2008 #-ccbin "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin" 
else
	NVCCFLAGS := #-idp /cygwin/ --use-local-env --cl-version 2008 #-ccbin "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin" 
endif
CXXFLAGS  :=  
CFLAGS    := 

# Common flags
COMMONFLAGS += $(INCLUDES_UNIX) -lcudart #-DUNIX -O2 -lcudart
COMMONFLAGSG++ += $(INCLUDES_UNIX) -lcudart #-DUNIX -O2 -lcudart

# Debug/release configuration
ifeq ($(dbg),1)
	COMMONFLAGS += -g
	COMMONFLAGSG++ += -g
	NVCCFLAGS   += --compiler-options -g -G
	BINSUBDIR   := debug
	LIBSUFFIX   := D
else 
	COMMONFLAGS += #-O3 
	BINSUBDIR   := release
	LIBSUFFIX   :=
	NVCCFLAGS   += #--compiler-options -fno-strict-aliasing
	CXXFLAGS    += #-fno-strict-aliasing
	CFLAGS      += #-fno-strict-aliasing
endif

# Libs
#LIB       := -L $(NVIDIASDKPATH)/lib -L$(NVIDIASAMPPATH)/lib" -L"$(NVIDIASAMPPATH)/common/lib/$(OSLOWER)
ifdef ON_WINDOWS
	export LIB       := $(LIB);"$(CUDADIR)lib\$(OSREAL)\ ";"$(VCINSTALLDIR)\lib\ " #"$(CUDADIR)lib\$(OSREAL)\cudart.lib" "$(VCINSTALLDIR)\lib\libcpmt.lib" 
	LLIB := "$(CUDADIR)lib/$(OSREAL)/cudart.lib"
else
ifdef ON_MAC
	LIB       := -L$(CUDADIR)/lib -L$(NVIDIASDKPATH)/lib -L$(NVIDIASAMPLIB) 
else
	LIB       := -L$(CUDADIR)/lib64 -L$(NVIDIASDKPATH)/lib -L$(NVIDIASAMPLIB) 
endif
endif
# Add source files here
EXECUTABLE	:= classol_sim
# Cuda source files (compiled with cudacc)
CUFILES		:= classol_sim.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 

NVCCFLAGS += --ptxas-options=-v
# -keep to keep all compile stages

TARGETDIR := $(BINDIR)/$(BINSUBDIR)
TARGET    := $(TARGETDIR)/$(EXECUTABLE)

# Add common flags
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)
################################################################################
# Set up object files
################################################################################
OBJDIR := $(ROOTOBJDIR)/$(BINSUBDIR)

OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.obj,$(notdir $(CCFILES)))
OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c_o,$(notdir $(CFILES)))
OBJS +=  $(patsubst %.cu,$(OBJDIR)/%.cu_o,$(notdir $(CUFILES)))

#OBJS_W 	  = $(shell cygpath -m $(OBJS))
#TARGET_W 	  = $(shell cygpath -m $(TARGET))

ifdef ON_WINDOWS
	LINKLINE  = $(LINK) /Fe$(TARGET) $(OBJS) $(LLIB) #? -o FOR LINUX
else	
	LINKLINE  = $(LINK) -o $(TARGET) $(OBJS) $(LIB) $(COMMONFLAGSG++)
endif
################################################################################
# Set up cubin files
################################################################################
CUBINDIR := $(SRCDIR)data
CUBINS +=  $(patsubst %.cu,$(CUBINDIR)/%.cubin,$(notdir $(CUBINFILES)))

################################################################################
# Rules and targets
#SHOULD CHANGE ALL FOR WINDOWS 


$(OBJDIR)/%.c_o : $(SRCDIR)%.c $(C_DEPS)
	$(VERBOSE)$(CC) $(CFLAGS) /Fe$@ -c $<
#	$(NVCC) -m32 -c $^ $(CFLAGS) -o $@
#	$(NVCC) -m32 -M $^ $(CFLAGS) > $@.dep

ifdef ON_WINDOWS
$(OBJDIR)/%.obj : $(SRCDIR)%.cpp $(C_DEPS) 	
	$(VERBOSE)$(CXX) $(CXXFLAGS) $< /c /Fo$(OBJDIR)/classol_sim	#was $(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $< before
else
$(OBJDIR)/%.obj : $(SRCDIR)%.cpp $(C_DEPS) 	
	$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $< 
endif


$(OBJDIR)/%.cu_o : $(SRCDIR)%.cu $(CU_DEPS)
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -c $<
#	$(NVCC) -c %^ $(CFLAGS) -o $@
#	$(NVCC) -M %^ $(CFLAGS) > $@.dep


$(OBJDIR)/%.obj : $(SRCDIR)%.cpp $(C_DEPS) 	
	$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $< 
#	$(NVCC) -m32 -c $^ $(CFLAGS) -o $@
#	$(NVCC) -m32 -M $^ $(CFLAGS) > $@.dep

$(OBJDIR)/%.cu_o : $(SRCDIR)%.cu $(CU_DEPS)
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -c $<
#	$(NVCC) -c %^ $(CFLAGS) -o $@
#	$(NVCC) -M %^ $(CFLAGS) > $@.dep


$(CUBINDIR)/%.cubin : $(SRCDIR)%.cu cubindirectory
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -cubin $<

$(TARGET): makedirectories $(OBJS) $(CUBINS) Makefile
	$(VERBOSE)$(LINKLINE)

cubindirectory:
	$(VERBOSE)mkdir -p $(CUBINDIR)

makedirectories:
	$(VERBOSE)mkdir -p $(OBJDIR)
	$(VERBOSE)mkdir -p $(TARGETDIR)

tidy :
	$(VERBOSE)find . | egrep "#" | xargs rm -f
	$(VERBOSE)find . | egrep "\~" | xargs rm -f

clean : tidy
	$(VERBOSE)rm -f $(OBJS)
	$(VERBOSE)rm -f $(CUBINS)
	$(VERBOSE)rm -f $(TARGET)
	$(VERBOSE)rm -f $(NVCC_KEEP_CLEAN)
#
# cuda and C/C++ compilation rules, with
# dependency generation:
#
