/*--------------------------------------------------------------------------
   Author: Anton Komissarov
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2018-05-18
  
--------------------------------------------------------------------------*/

%module(directors="1") Snippet // for inheritance in python
%{
#include "snippet.h"
%}

%include <std_string.i>
%include <std_pair.i>
%include <std_vector.i>

%feature("director") Snippet::Base; // for inheritance in python
%include "include/snippet.h"

// helper class for callbacks
%feature("director") DerivedParamFunc;
%rename(__call__) DerivedParamFunc::operator();
%inline %{
struct DerivedParamFunc {
  virtual double operator()( const std::vector<double> & pars, double dt ) const = 0;
  virtual ~DerivedParamFunc() {}
};
%}

%{
#include <functional>
%}
%rename(STD_DPFunc) std::function<double( const std::vector<double> &, double )>;
%rename(__call__) std::function<double( const std::vector<double> &, double )>::operator();
%feature("director") std::function<double( const std::vector<double> &, double )>;
namespace std {
  struct function<double( const std::vector<double> &, double )> {
    
    function<double( const std::vector<double> &, double )>(const std::function<double( const std::vector<double> &, double )>&);

    double operator()( const std::vector<double> &, double) const;
  };
}

// helper function to convert DerivedParamFunc to std::function
%inline %{
std::function<double( const std::vector<double> &, double )> makeDPF( DerivedParamFunc* dpf )
{
  return std::bind( &DerivedParamFunc::operator(), dpf, std::placeholders::_1, std::placeholders::_2 );
}
%}


%template(StringDPFPair) std::pair<std::string, std::function<double( const std::vector<double> &, double )>>;
%template(StringDPFPairVector) std::vector<std::pair<std::string, std::function<double( const std::vector<double> &, double )>>>;


