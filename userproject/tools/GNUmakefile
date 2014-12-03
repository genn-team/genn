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

all: gen_input_structured_FLOAT gen_pnlhi_syns_FLOAT gen_kcdn_syns_FLOAT gen_pnkc_syns_FLOAT gen_pnkc_syns_indivID_FLOAT gen_syns_sparse_FLOAT gen_syns_sparse_izhModel_FLOAT gen_input_structured_DOUBLE gen_pnlhi_syns_DOUBLE gen_kcdn_syns_DOUBLE gen_pnkc_syns_DOUBLE gen_pnkc_syns_indivID_DOUBLE gen_syns_sparse_DOUBLE gen_syns_sparse_izhModel_DOUBLE

%_FLOAT: %.cc
	g++ $(CXXFLAGS) -Dscalar=float -o $@ $<

%_DOUBLE: %.cc
	g++ $(CXXFLAGS) -Dscalar=double -o $@ $<
clean:
	$(RM) gen_input_structured_FLOAT gen_pnlhi_syns_FLOAT gen_kcdn_syns_FLOAT gen_pnkc_syns_FLOAT gen_pnkc_syns_indivID_FLOAT gen_syns_sparse_FLOAT gen_syns_sparse_izhModel_FLOAT gen_input_structured_DOUBLE gen_pnlhi_syns_DOUBLE gen_kcdn_syns_DOUBLE gen_pnkc_syns_DOUBLE gen_pnkc_syns_indivID_DOUBLE gen_syns_sparse_DOUBLE gen_syns_sparse_izhModel_DOUBLE
