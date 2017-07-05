
#ifndef STRINGUTILS_H
#define STRINGUTILS_H

#include <string>
#include <sstream>


// Forward declarations
class NNmodel;

using namespace std;


//--------------------------------------------------------------------------
/*! \brief template functions for conversion of various types to C++ strings
 */
//--------------------------------------------------------------------------

template<class T> std::string toString(T t)
{
    std::stringstream s;
    s << std::showpoint << t;
    return s.str();
}

#define tS(X) toString(X) //!< Macro providing the abbreviated syntax tS() instead of toString().

#endif // STRINGUTILS_H
