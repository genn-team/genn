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

CXXFLAGS	=-Wall -Winline -O3 -g -I$(GENN_PATH)/lib/include/numlib

all: gen_input_structured gen_pnlhi_syns gen_kcdn_syns gen_pnkc_syns gen_pnkc_syns_indivID gen_syns_sparse gen_syns_sparse_izhModel 

%: %.cc
	g++ $(CXXFLAGS) -o $@ $<

clean:
	$(RM) -rf *.o *.dSYM gen_input_structured gen_pnlhi_syns gen_kcdn_syns gen_pnkc_syns gen_pnkc_syns_indivID gen_syns_sparse gen_syns_sparse_izhModel 
