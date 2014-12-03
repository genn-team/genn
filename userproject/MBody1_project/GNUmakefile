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

CXXFLAGS	=-Wall -Winline -O3 -g -I$(GENN_PATH)/lib/include 

all: generate_run tools

generate_run: generate_run.cc 
	g++  $(CXXFLAGS) -L$(CUDA_PATH)/lib -o generate_run generate_run.cc 

clean:
	$(RM) generate_run
	cd ../tools && make clean

tools:
	cd ../tools && make
