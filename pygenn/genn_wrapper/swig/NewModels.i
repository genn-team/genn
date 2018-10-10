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

// GeNN includes
#include "newModels.h"

// PyGenn includes
#include "customParamValues.h"
#include "customVarValues.h"

// Generated includes
#include "initVarSnippetCustom.h"
%}
%ignore LegacyWrapper;

%include <std_vector.i>

%import "Snippet.i"
%import "InitVarSnippet.i"

%feature("director") NewModels::Base; // for inheritance in python
%nodefaultctor NewModels::VarInit;
%include "newModels.h"

%nodefaultctor CustomValues::VarValues;
%include "customVarValues.h"
%nodefaultctor CustomValues::ParamValues;
%include "customParamValues.h"

%template(CustomVarValues) CustomValues::VarValues::VarValues<double>;
%template(CustomVarValues) CustomValues::VarValues::VarValues<NewModels::VarInit>;

// ignore vector(size) contructor & resize, otherwise compiler will complain about
// missing default ctor in VarInit
%ignore std::vector<NewModels::VarInit>::vector(size_type);
%ignore std::vector<NewModels::VarInit>::resize;
%template(VarInitVector) std::vector<NewModels::VarInit>;
