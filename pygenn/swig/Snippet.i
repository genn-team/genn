/*--------------------------------------------------------------------------
   Author: Anton Komissarov
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2018-05-18
  
--------------------------------------------------------------------------*/

%module(directors="1",package="pygenn") Snippet // for inheritance in python
%{
#include "snippet.h"
%}

%import "StlContainers.i"

%feature("director") Snippet::Base; // for inheritance in python
%include "snippet.h"

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
