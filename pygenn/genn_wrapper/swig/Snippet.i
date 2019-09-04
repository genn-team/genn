/*--------------------------------------------------------------------------
   Author: Anton Komissarov
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2018-05-18
  
--------------------------------------------------------------------------*/

%module(package="genn_wrapper",directors="1") Snippet // for inheritance in python
%{
#include "snippet.h"
%}
%include <std_string.i>
%include <std_vector.i>

%feature("flatnested", "1");
%rename("%(undercase)s", %$isfunction, notregexmatch$name="add[a-zA-Z]*Population", notregexmatch$name="addCurrentSource", notregexmatch$name="assignExternalPointer[a-zA-Z]*") "";

%import "StlContainers.i"


%feature("director") Snippet::Base; // for inheritance in python


// flatten nested classes
%rename (EGP) Snippet::Base::EGP;
%rename (ParamVal) Snippet::Base::ParamVal;
%rename (DerivedParam) Snippet::Base::DerivedParam;

// add vector overrides for them
%template(EGPVector) std::vector<Snippet::Base::EGP>;
%template(ParamValVector) std::vector<Snippet::Base::ParamVal>;
%template(DerivedParamVector) std::vector<Snippet::Base::DerivedParam>;

%include "gennExport.h"
%include "snippet.h"

// Extend each of the underlying structs with constructors
%extend Snippet::Base::EGP {
    EGP(const std::string &name, const std::string &type) 
    {
        Snippet::Base::EGP* v = new Snippet::Base::EGP();
        v->name = name;
        v->type = type;
        return v;
    }
};

%extend Snippet::Base::ParamVal {
    ParamVal(const std::string &name, const std::string &type, double value) 
    {
        Snippet::Base::ParamVal* v = new Snippet::Base::ParamVal();
        v->name = name;
        v->type = type;
        v->value = value;
        return v;
    }
};

%extend Snippet::Base::DerivedParam {
    DerivedParam(const std::string &name, std::function<double(const std::vector<double> &, double)> func) 
    {
        Snippet::Base::DerivedParam* v = new Snippet::Base::DerivedParam();
        v->name = name;
        v->func = func;
        return v;
    }
};
// helper class for callbacks
%feature("director") DerivedParamFunc;
%rename(__call__) DerivedParamFunc::operator();
%inline %{
struct DerivedParamFunc {
  virtual double operator()( const std::vector<double> & pars, double dt ) const = 0;
  virtual ~DerivedParamFunc() {}
};
%}



// helper function to convert DerivedParamFunc to std::function
%inline %{
std::function<double( const std::vector<double> &, double )> makeDPF( DerivedParamFunc* dpf )
{
  return std::bind( &DerivedParamFunc::operator(), dpf, std::placeholders::_1, std::placeholders::_2 );
}
%}
