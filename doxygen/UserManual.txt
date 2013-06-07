/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/


//----------------------------------------------------------------------------
/*!  \page Manual User Manual

\tableofcontents

\section sIntro Introduction

\section sect1 Defining a model
A model is defined by the user by providing the function 
\code{.cc}
void modelDefinition(NNmodel &model) 
\endcode
in this function the following tasks must be completed:
- The name of the model must be defined:
\code{.cc}
model.setName("MyModel");
\endcode

- neuron populations (at least one) must be added:
\code{.cc}
model.addNeuronPopulation("PN", _NAL, POISSONNEURON, myPOI_p, myPOI_ini);
\endcode
where the arguments are:
\arg \c const \c char* name: Name of the neuron population
\arg \c int n: number of neurons in the population
\arg \c int TYPE: Type of the neurons, refers to either a standard type (see below) or user-defined type
\arg \c float *para: Parameters of this neuron type
\arg \c float *ini: Initial values of this neuron type 


The user may add as many neuron populations as the model necessitates. They should all have unique names.

\section sect2 Predefined neuron types


\section sect3 Predefined synapose models

\section sect4 Defining your own neuron model
*/
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
/*! \page Credits Credits
 

*/
//----------------------------------------------------------------------------
