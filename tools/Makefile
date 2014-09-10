#--------------------------------------------------------------------------
# Author: Thomas Nowotny
#
# Institute: Institute for Nonlinear Dynamics
#            University of California San Diego
#            La Jolla, CA 92093-0402
#
# email to:  tnowotny@ucsd.edu
#
# initial version: 2002-01-24
#
#--------------------------------------------------------------------------

C++ = g++
C = gcc
RM= rm -f

INCLUDE= -I$(GeNNPATH)/lib/include/numlib

FLAGS= -Wall -Winline -O3
#-O2 -march=athlon

all: generate_run generate_run_BM generate_run_1comp generate_run_PoissonIzh gen_input_fixfixfixno_struct gen_pnlhi_syns gen_kcdn_syns gen_pnkc_syns gen_pnkc_syns_indivID gen_syns_sparse generate_izhikevich_network_run gen_syns_sparse_izhModel

generate_run: generate_run.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o generate_run generate_run.cc

generate_run_BM: generate_run_BM.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o generate_run_BM generate_run_BM.cc

generate_run_1comp: generate_run_1comp.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o generate_run_1comp generate_run_1comp.cc

generate_run_PoissonIzh: generate_run_PoissonIzh.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o generate_run_PoissonIzh generate_run_PoissonIzh.cc
	
gen_pnkc_syns: gen_pnkc_syns.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o gen_pnkc_syns gen_pnkc_syns.cc

gen_pnkc_syns_sparse: gen_pnkc_syns_sparse.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o gen_pnkc_syns_sparse gen_pnkc_syns_sparse.cc

gen_pnkc_syns_indivID: gen_pnkc_syns_indivID.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o gen_pnkc_syns_indivID gen_pnkc_syns_indivID.cc

gen_pnlhi_syns: gen_pnlhi_syns.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o gen_pnlhi_syns gen_pnlhi_syns.cc

gen_kcdn_syns: gen_kcdn_syns.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o gen_kcdn_syns gen_kcdn_syns.cc

gen_input_fixfixfixno_struct: gen_input_fixfixfixno_struct.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o gen_input_fixfixfixno_struct gen_input_fixfixfixno_struct.cc

gen_syns_sparse: gen_syns_sparse.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o gen_syns_sparse gen_syns_sparse.cc

generate_izhikevich_network_run: generate_izhikevich_network_run.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o generate_izhikevich_network_run generate_izhikevich_network_run.cc

gen_syns_sparse_izhModel: gen_syns_sparse_izhModel.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o gen_syns_sparse_izhModel gen_syns_sparse_izhModel.cc


clean:
	$(RM) *.o generate_run generate_run_BM generate_run_1comp gen_pnkc_syns gen_pnkc_syns_indivID gen_pnlhi_syns gen_kcdn_syns gen_input_fixfixfixno_struct generate_izhikevich_network_run gen_syns_sparse_izhModel
