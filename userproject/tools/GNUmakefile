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

CXXFLAGS        :=-Wall -Winline -O3 -std=c++0x
INCLUDE_FLAGS   :=-I"$(GENN_PATH)/userproject/include"

all: gen_input_structured gen_pnlhi_syns gen_kcdn_syns gen_kcdn_syns_fixto10K gen_pnkc_syns gen_pnkc_syns_indivID gen_syns_sparse gen_syns_sparse_izhModel 

%: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(INCLUDE_FLAGS)

clean:
	rm -rf *.o *.dSYM gen_input_structured gen_pnlhi_syns gen_kcdn_syns gen_kcdn_syns_fixto10K gen_pnkc_syns gen_pnkc_syns_indivID gen_syns_sparse gen_syns_sparse_izhModel 
