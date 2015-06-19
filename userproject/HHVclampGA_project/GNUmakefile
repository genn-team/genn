#--------------------------------------------------------------------------
#   Author: Thomas Nowotny
#  
#   Institute: Informatics
#              University of Sussex 
#              Brighton BN1 9QJ, UK
#  
#   email to:  t.nowotny@sussex.ac.uk
#  
#   initial version: 2014-06-26
#  
#--------------------------------------------------------------------------

INCLUDE_FLAGS	= -I$(GENN_PATH)/lib/include 
CXXFLAGS	= $(INCLUDE_FLAGS) -Wall -Winline -O3

all: generate_run

generate_run: generate_run.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o generate_run generate_run.cc

clean:
	$(RM) generate_run
