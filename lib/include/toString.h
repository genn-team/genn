/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

#ifndef _TOSTRING_H_
#define _TOSTRING_H_ //!< macro for avoiding multiple inclusion during compilation


//--------------------------------------------------------------------------
/*! \file toString.h

\brief Contains a template function for string conversion from const char* to C++ string
*/
//--------------------------------------------------------------------------

using namespace std;

#include <string>
#include <sstream>

//--------------------------------------------------------------------------
/*! \brief template function for string conversion from const char* to C++ string
 */
//--------------------------------------------------------------------------

template<class T>
std::string toString(T t)
{
  std::stringstream s;
  //  s << std::showpoint;
  s << t;
  return s.str();
}

template<>
std::string toString(float t)
{
  std::stringstream s;
  //  s << std::showpoint;
  s << t;
  if (floor(t) == t) { // it's a whole number
      s << ".0";
  }
  return s.str();
}

template<>
std::string toString(double t)
{
  std::stringstream s;
  //  s << std::showpoint;
  s << t;
  if (floor(t) == t) { // it's a whole number
      s << ".0";
  }
  return s.str();
}

#define tS(X) toString(X) //!< Macro providing the abbreviated syntax tS() instead of toString().

#endif
