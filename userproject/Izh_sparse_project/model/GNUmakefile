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

EXECUTABLE         :=Izh_sim_sparse
SOURCES            :=$(EXECUTABLE).cu
ifndef CPU_ONLY
    SOURCES        += $(GENN_PATH)/userproject/include/GeNNHelperKrnls.cu
endif
OPTIMIZATIONFLAGS  :=-O3

include	$(GENN_PATH)/userproject/include/makefile_common_gnu.mk

$(GENN_PATH)/userproject/include/GeNNHelperKrnls.o: \
 $(GENN_PATH)/userproject/include/GeNNHelperKrnls.cu \
 $(GENN_PATH)/userproject/include/GeNNHelperKrnls.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FLAGS) -c $< -o $@
