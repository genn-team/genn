/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2015-09-12
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file command_line_processing.h

\brief This file contains some tools for parsing the argv array which contains the command line options.
*/


#ifndef COMMAND_LINE_PROCESSING
#define COMMAND_LINE_PROCESSING

//--------------------------------------------------------------------------
/*! \brief template function for string conversion from const char* to C++ string
 */
//--------------------------------------------------------------------------

#include "toString.h"

string toUpper(string s)
{
    for (unsigned int i= 0; i < s.length(); i++) {
	s[i]= toupper(s[i]);
    }
    return s;
}

string toLower(string s)
{
    for (unsigned int i= 0; i < s.length(); i++) {
	s[i]= tolower(s[i]);
    }
    return s;
}

int extract_option(char *op, string &option) 
{
    string sop= tS(op);
    size_t pos= sop.find(tS("="));
    if (pos == string::npos) {
	return -1;
    }
    option= sop.substr(0,pos);

    return 0;
}

int extract_bool_value(char *op, unsigned int &val) 
{
    string sop= tS(op);
    size_t pos= sop.find(tS("="));
    if (pos == string::npos) {
	return -1;
    }
    string sval= sop.substr(pos+1);
    int tmpval= atoi(sval.c_str());
    if ((tmpval != 0) && (tmpval != 1)) {
	return -1;
    }
    val= tmpval;
    
    return 0;
}

int extract_string_value(char *op, string &val) 
{
    string sop= tS(op);
    size_t pos= sop.find(tS("="));
    if (pos == string::npos) {
	return -1;
    }
    val= sop.substr(pos+1);

    return 0;
}

#endif
