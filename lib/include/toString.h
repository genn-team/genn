/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

#ifndef _TOSTRING_H_
#define _TOSTRING_H_

using namespace std;
#include <string>
#include <sstream>

////////////////////////////////////////////////////////////////////////////////
// Template function for string conversion 
////////////////////////////////////////////////////////////////////////////////

template<typename T>
std::string toString(T t)
{
  std::stringstream s;
  s << t;
  return s.str();
} 

#define tS(X) toString(X)

#endif
