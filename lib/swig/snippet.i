/*--------------------------------------------------------------------------
   Author: Anton Komissarov
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2018-05-18
  
--------------------------------------------------------------------------*/

/* %module(directors="1") Snippet // for inheritance in python */
%module Snippet
%{
#include "snippet.h"
%}
%include <std_string.i>
/* %include "std_function.i" */
%include <std_pair.i>
%include <std_vector.i>
/* %feature("director") Snippet::Base; // for inheritance in python */
%include "include/snippet.h"

