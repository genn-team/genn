##--------------------------------------------------------------------------
##   Author: Thomas Nowotny
##  
##   Institute: Center for Computational Neuroscience and Robotics
##              University of Sussex
##            Falmer, Brighton BN1 9QJ, UK 
##  
##   email to:  T.Nowotny@sussex.ac.uk
##  
##   initial version: 2010-02-07
##  
##--------------------------------------------------------------------------

EXECUTABLE	:=classol_sim
SOURCES		:=classol_sim.cu
OPTIMIZATIONFLAGS:= -O3 -ffast-math

include	$(GENN_PATH)/userproject/include/makefile_common_gnu.mk

.PHONY: cpu_only
cpu_only: INCLUDE_FLAGS := -DCPU_ONLY
cpu_only: release
