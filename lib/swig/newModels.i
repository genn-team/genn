/*--------------------------------------------------------------------------
   Author: Anton Komissarov
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2018-05-18
  
--------------------------------------------------------------------------*/

/* %module(directors="1") NewModels // for inheritance in python */
%module NewModels
%{
#include "newModels.h"
%}
%ignore LegacyWrapper;
%include <std_string.i>
%include <std_pair.i>
%include <std_vector.i>
%import "snippet.i"
%import "initVarSnippet.i"
/* %feature("director") NewModels::Base; // for inheritance in python */
%include "include/newModels.h"
%template() std::pair<std::string, double>;
%template() std::pair<std::string, std::pair<std::string, double>>;
%template() std::vector<std::pair<std::string, std::pair<std::string, double>>>;
