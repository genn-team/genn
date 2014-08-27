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

C++ = g++
C = gcc
RM= rm -f

INCLUDE= -I$(GeNNPATH)/lib/include/numlib

FLAGS= -Wall -Winline -O3
#-O2 -march=athlon

all: generate_run 

generate_run: generate_run.cc
	$(C++) $(FLAGS) -g $(INCLUDE) -o generate_run generate_run.cc

clean:
	$(RM) *.o generate_run
