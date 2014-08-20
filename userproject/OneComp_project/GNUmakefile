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

CXX	=g++
RM	=rm -f
INCLUDE	=-I$(GENNPATH)/lib/include/numlib
FLAGS	=-Wall -Winline -O3 #-O2 -march=athlon

all: generate_run

generate_run: generate_run.cc
	$(CXX) $(FLAGS) -g $(INCLUDE) -o generate_run generate_run.cc

clean:
	$(RM) generate_run
