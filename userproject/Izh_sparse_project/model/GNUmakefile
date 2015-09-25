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

EXECUTABLE	:=Izh_sim_sparse
ifeq ($(CPU_ONLY),1)
  SOURCES 	:=Izh_sim_sparse_cpu_only.cc
else
  SOURCES 	:=Izh_sim_sparse.cu
endif
OPTIMIZATIONFLAGS:= -O3

include	$(GENN_PATH)/userproject/include/makefile_common_gnu.mk
