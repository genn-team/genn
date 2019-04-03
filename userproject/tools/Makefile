#--------------------------------------------------------------------------
#  Author: Thomas Nowotny
#  
#  Institute: Center for Computational Neuroscience and Robotics
#             University of Sussex
#             Falmer, Brighton BN1 9QJ, UK
#  
#  email to:  T.Nowotny@sussex.ac.uk
#  
#  initial version: 2010-02-07
#  
#--------------------------------------------------------------------------

CXXFLAGS        :=-Wall -Winline -O3 -std=c++11
INCLUDE_FLAGS   :=-I"$(GENN_PATH)/userproject/include"

all: gen_input_structured

%: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(INCLUDE_FLAGS)

clean:
	rm -rf *.o *.dSYM gen_input_structured
