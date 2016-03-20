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

EXECUTABLE	:=VClampGA
ifeq ($(CPU_ONLY),1)
  SOURCES :=$(EXECUTABLE)_cpu_only.cc
else
  SOURCES :=$(EXECUTABLE).cu
endif

OPTIMIZATIONFLAGS:= -O3

include	$(GENN_PATH)/userproject/include/makefile_common_gnu.mk

$(EXECUTABLE)_cpu_only.o: $(EXECUTABLE)_cpu_only.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $< -o $@ -c
	rm $(EXECUTABLE)_cpu_only.cc

$(EXECUTABLE)_cpu_only.cc: $(EXECUTABLE).cu
	cp $(EXECUTABLE).cu $(EXECUTABLE)_cpu_only.cc
