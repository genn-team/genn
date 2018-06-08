/*--------------------------------------------------------------------------
   Author: Anton Komissarov
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2018-05-18
  
--------------------------------------------------------------------------*/

%module(directors="1") NewModels // for inheritance in python
%{
#include "newModels.h"
#include "../swig/customValues.h"
%}
%ignore LegacyWrapper;

%import "snippet.i"
%import "initVarSnippet.i"

%feature("director") NewModels::Base; // for inheritance in python
%include "include/newModels.h"

%nodefaultctor CustomValues::VarValues;
%nodefaultctor CustomValues::ParamValues;
%include "customValues.h"

%template(CustomParamValues) CustomValues::ParamValues::ParamValues<double>; 
%template(CustomVarValues) CustomValues::VarValues::VarValues<double>;
