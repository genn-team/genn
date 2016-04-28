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

EXECUTABLE		:=classol_sim
ifndef CPU_ONLY
    SOURCES		:=$(EXECUTABLE).cu
else
    SOURCES		:=$(EXECUTABLE)_cpu_only.cc
endif
OPTIMIZATIONFLAGS	:=-O3 -ffast-math
NVCC_OPTIMIZATIONFLAGS	:=-use_fast_math -O3

include	$(GENN_PATH)/userproject/include/makefile_common_gnu.mk

$(EXECUTABLE)_cpu_only.cc: $(EXECUTABLE).cu
	cp $(EXECUTABLE).cu $(EXECUTABLE)_cpu_only.cc
