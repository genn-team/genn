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

EXECUTABLE	:=PoissonIzh_sim
ifeq ($(CPU_ONLY),1)
  SOURCES :=PoissonIzh_sim_cpu_only.cc
else
  SOURCES :=PoissonIzh_sim.cu
endif

OPTIMIZATIONFLAGS:= -O3

include	$(GENN_PATH)/userproject/include/makefile_common_gnu.mk
