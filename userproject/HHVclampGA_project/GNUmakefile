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

CXXFLAGS	= -Wall -Winline -O3
INCLUDE_FLAGS 	:=-I$(GENN_PATH)/lib/include -I$(GENN_PATH)/userproject/include

all: generate_run

generate_run: generate_run.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -o generate_run generate_run.cc

clean:
	$(RM) generate_run
