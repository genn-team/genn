/*--------------------------------------------------------------------------
   Author: Anton Komissarov
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2018-05-18
  
--------------------------------------------------------------------------*/

%module WeightUpdateModels
%{
#include "newWeightUpdateModels.h"
%}
%include <std_string.i>
%import "newModels.i"
%include "include/newWeightUpdateModels.h"
