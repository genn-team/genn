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

NVCCFLAGS	:=--compiler-options "-ffast-math"
CXXFLAGS	:=-ffast-math

#INCLUDE_FLAGS+= -g -G 
NVCCFLAGS	:=--compiler-options "-ffast-math"
CXXFLAGS	:=-ffast-math
include	$(GENN_PATH)/userproject/include/makefile_common_gnu.mk

.PHONY: cpu_only
#cpu_only: $(SOURCES)
#	$(CXX) $(CXXFLAGS) -I$(GENN_PATH)/lib/include -I$(GENN_PATH)/userproject/include -o classol_sim_cpu_only classol_sim.cc -lc++ -stdlib=libstdc++ -lc++
