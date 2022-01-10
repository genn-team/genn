#pragma once

// Standard C++ includes
#include <array>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

// Standard C includes
#include <cstring>

// Boost includes
#include <sha1.hpp>

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
GENN_EXPORT bool isRNGRequired(const std::unordered_map<std::string, Models::VarInit> &varInitialisers);

//--------------------------------------------------------------------------
//! \brief Function to determine whether a string containing a type is a pointer
//--------------------------------------------------------------------------
GENN_EXPORT bool isTypePointer(const std::string &type);

//--------------------------------------------------------------------------
//! \brief Function to determine whether a string containing a type is a pointer to a pointer
//--------------------------------------------------------------------------
GENN_EXPORT bool isTypePointerToPointer(const std::string &type);

//--------------------------------------------------------------------------
//! \brief Function to determine whether a string containing a type is floating point
//--------------------------------------------------------------------------
GENN_EXPORT bool isTypeFloatingPoint(const std::string &type);

//--------------------------------------------------------------------------
//! \brief Assuming type is a string containing a pointer type, function to return the underlying type
//--------------------------------------------------------------------------
GENN_EXPORT std::string getUnderlyingType(const std::string &type);

//--------------------------------------------------------------------------
//! \brief Is the variable name valid? GeNN variable names must obey C variable naming rules
//--------------------------------------------------------------------------
GENN_EXPORT void validateVarName(const std::string &name, const std::string &description);

//--------------------------------------------------------------------------
//! \brief Is the population name valid? GeNN population names obey C variable naming rules but can start with a number
//--------------------------------------------------------------------------
GENN_EXPORT void validatePopName(const std::string &name, const std::string &description);

//--------------------------------------------------------------------------
//! \brief Are values provided for all of the the parameter names in the vector?
//--------------------------------------------------------------------------
GENN_EXPORT void validateParamValues(const std::vector<std::string> &paramNames, const std::unordered_map<std::string, double> &paramValues, 
                                     const std::string &description);

//--------------------------------------------------------------------------
//! \brief Are all the parameter names in vector valid? GeNN variables and population names must obey C variable naming rules
//--------------------------------------------------------------------------
GENN_EXPORT void validateParamNames(const std::vector<std::string> &paramNames);

//--------------------------------------------------------------------------
//! \brief Are the 'name' fields of all structs in vector valid? GeNN variables and population names must obey C variable naming rules
//--------------------------------------------------------------------------
template<typename T>
void validateVecNames(const std::vector<T> &vec, const std::string &description)
{
    for(const auto &v : vec) {
        validateVarName(v.name, description);
    }
}

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
inline std::string writePreciseString(T value)
{
    std::stringstream s;
    writePreciseString(s, value);
    return s.str();
}

//! Hash strings
inline void updateHash(const std::string &string, boost::uuids::detail::sha1 &hash)
{
    hash.process_bytes(string.data(), string.size());
}

//! Hash arithmetic types and enums
template<typename T, typename std::enable_if<std::is_arithmetic<T>::value || std::is_enum<T>::value>::type * = nullptr>
inline void updateHash(const T &value, boost::uuids::detail::sha1 &hash)
{
    hash.process_bytes(&value, sizeof(T));
}

//! Hash arrays of types which can, themselves, be hashed
template<typename T, size_t N>
inline void updateHash(const std::array<T, N> &array, boost::uuids::detail::sha1 &hash)
{
    for(const auto &v : array) {
        updateHash(v, hash);
    }
}

//! Hash vectors of types which can, themselves, be hashed
template<typename T>
inline void updateHash(const std::vector<T> &vector, boost::uuids::detail::sha1 &hash)
{
    for(const auto &v : vector) {
        updateHash(v, hash);
    }
}

//! Hash vectors of bools
inline void updateHash(const std::vector<bool> &vector, boost::uuids::detail::sha1 &hash)
{
    for(bool v : vector) {
        updateHash(v, hash);
    }
}


//! Hash unordered maps of types which can, themselves, be hashed
template<typename K, typename V>
inline void updateHash(const std::unordered_map<K, V> &map, boost::uuids::detail::sha1 &hash)
{
    for(const auto &v : map) {
        updateHash(v.first, hash);
        updateHash(v.second, hash);
    }
}

//! Functor for generating a hash suitable for use in std::unordered_map etc (i.e. size_t size) from a SHA1 digests
struct SHA1Hash
{
    size_t operator()(const boost::uuids::detail::sha1::digest_type &digest) const
    {
        size_t hash;
        memcpy(&hash, &digest[0], sizeof(size_t));
        return hash;
    };
};
}   // namespace Utils
