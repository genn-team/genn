/*--------------------------------------------------------------------------
   Author: Anton Komissarov
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2018-05-18
  
--------------------------------------------------------------------------*/

%module(package="pygenn") GeNNPreferences
%{
#include "global.h"
%}

// ignore other global variables so that they are not wrapped as GeNNPreferences
%ignore GENN_FLAGS;
%ignore neuronBlkSz;
%ignore synapseBlkSz;
%ignore learnBlkSz;
%ignore synDynBlkSz;
%ignore initBlkSz;
%ignore initSparseBlkSz;
%ignore deviceProp;
%ignore theDevice;
%ignore deviceCount;
%ignore hostCount;
%include "global.h"

%define %unignore %rename("%s") %enddef
// unignore variables so that they still can be wrapped
%unignore GENN_FLAGS;
%unignore neuronBlkSz;
%unignore synapseBlkSz;
%unignore learnBlkSz;
%unignore synDynBlkSz;
%unignore initBlkSz;
%unignore initSparseBlkSz;
%unignore deviceProp;
%unignore theDevice;
%unignore deviceCount;
%unignore hostCount;
