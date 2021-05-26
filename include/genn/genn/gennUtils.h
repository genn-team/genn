#pragma once

// Standard C++ includes
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// Boost includes
#include <boost/uuid/detail/sha1.hpp>

// GeNN includes
#include "gennExport.h"

// Forward declarations
namespace Models
{
class VarInit;
}

//--------------------------------------------------------------------------
// Utils
//--------------------------------------------------------------------------
namespace Utils
{
//--------------------------------------------------------------------------
//! \brief Does the code string contain any functions requiring random number generator
//--------------------------------------------------------------------------
GENN_EXPORT bool isRNGRequired(const std::string &code);

//--------------------------------------------------------------------------
//! \brief Does the model with the vectors of variable initialisers and modes require an RNG for the specified init location i.e. host or device
//--------------------------------------------------------------------------
GENN_EXPORT bool isRNGRequired(const std::vector<Models::VarInit> &varInitialisers);

//--------------------------------------------------------------------------
//! \brief Function to determine whether a string containing a type is a pointer
//--------------------------------------------------------------------------
GENN_EXPORT bool isTypePointer(const std::string &type);

//--------------------------------------------------------------------------
//! \brief Function to determine whether a string containing a type is a pointer to a pointer
//--------------------------------------------------------------------------
GENN_EXPORT bool isTypePointerToPointer(const std::string &type);

//--------------------------------------------------------------------------
//! \brief Assuming type is a string containing a pointer type, function to return the underlying type
//--------------------------------------------------------------------------
GENN_EXPORT std::string getUnderlyingType(const std::string &type);

//--------------------------------------------------------------------------
//! \brief This function writes a floating point value to a stream -setting the precision so no digits are lost
//--------------------------------------------------------------------------
template<class T, typename std::enable_if<std::is_floating_point<T>::value>::type * = nullptr>
void writePreciseString(std::ostream &os, T value)
{
    // Cache previous precision
    const std::streamsize previousPrecision = os.precision();

    // Set scientific formatting
    os << std::scientific;

    // Set precision to what is required to fully represent T
    os << std::setprecision(std::numeric_limits<T>::max_digits10);

    // Write value to stream
    os << value;

    // Reset to default formatting
    // **YUCK** GCC 4.8.X doesn't seem to include std::defaultfloat
    os.unsetf(std::ios_base::floatfield);
    //os << std::defaultfloat;

    // Restore previous precision
    os << std::setprecision(previousPrecision);
}

//--------------------------------------------------------------------------
//! \brief This function writes a floating point value to a string - setting the precision so no digits are lost
//--------------------------------------------------------------------------
template<class T, typename std::enable_if<std::is_floating_point<T>::value>::type * = nullptr>
std::string writePreciseString(T value)
{
    std::stringstream s;
    writePreciseString(s, value);
    return s.str();
}

//! Hash strings
void updateHash(const std::string &string, boost::uuids::detail::sha1 &hash)
{
    hash.process_bytes(string.data(), string.size());
}

//! Hash vectors of types which can, themselves, be hashed
// **THINK** could add override for vectors of arithmetic types where data() is passed in
template<typename T>
void updateHash(const std::vector<T> &vector, boost::uuids::detail::sha1 &hash)
{
    for(const auto &v : vector) {
        update(v, hash);
    }
}

//! Hash arithmetic types
template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
void updateHash(T value, boost::uuids::detail::sha1 &hash)
{
    hash.process_bytes(&value, sizeof(T));
}
}   // namespace Utils
