/*--------------------------------------------------------------------------
   Author: Anton Komissarov
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2018-05-18
  
--------------------------------------------------------------------------*/

%module(package="genn_wrapper",directors="1") Models // for inheritance in python
%{

// GeNN includes
#include "models.h"

// PyGenn includes
#include "customParamValues.h"
#include "customVarValues.h"

// Generated includes
#include "initVarSnippetCustom.h"
%}

%feature("flatnested", "1");
%rename("%(undercase)s", %$isfunction, notregexmatch$name="add[a-zA-Z]*Population", notregexmatch$name="addCurrentSource", notregexmatch$name="assignExternalPointer[a-zA-Z]*") "";

%ignore LegacyWrapper;

%include <std_vector.i>

%import "Snippet.i"
%import "InitVarSnippet.i"

%include "gennExport.h"
%feature("director") Models::Base; // for inheritance in python
%nodefaultctor Models::VarInit;

// flatten nested classes
%rename (Var) Models::Base::Var;

// add vector overrides for them
%template(VarVector) std::vector<Models::Base::Var>;

%include "models.h"


%nodefaultctor CustomValues::VarValues;
%include "customVarValues.h"
%nodefaultctor CustomValues::ParamValues;
%include "customParamValues.h"

%template(CustomVarValues) CustomValues::VarValues::VarValues<double>;
%template(CustomVarValues) CustomValues::VarValues::VarValues<Models::VarInit>;

// ignore vector(size) contructor & resize, otherwise compiler will complain about
// missing default ctor in VarInit
%ignore std::vector<Models::VarInit>::vector(size_type);
%ignore std::vector<Models::VarInit>::resize;
%template(VarInitVector) std::vector<Models::VarInit>;
