#pragma once

// Standard C++ includes
#include <array>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

// Standard C includes
#include <cstring>

// Boost includes
#include <sha1.hpp>

// GeNN includes
#include "gennExport.h"

// GeNN code generator includes
#include "transpiler/token.h"

// Forward declarations
namespace GeNN::Models
{
class VarInit;
}

//--------------------------------------------------------------------------
// GeNN::Utils
//--------------------------------------------------------------------------
namespace GeNN::Utils
{
//--------------------------------------------------------------------------
//! \brief Does the code string contain any functions requiring random number generator
//--------------------------------------------------------------------------
GENN_EXPORT bool isIdentifierReferenced(const std::string &identifierName, const std::vector<Transpiler::Token> &tokens);

//--------------------------------------------------------------------------
//! \brief Does the code string contain any functions requiring random number generator
//--------------------------------------------------------------------------
GENN_EXPORT bool isRNGRequired(const std::vector<Transpiler::Token> &tokens);

//--------------------------------------------------------------------------
//! \brief Does the model with the vectors of variable initialisers and modes require an RNG for the specified init location i.e. host or device
//--------------------------------------------------------------------------
GENN_EXPORT bool isRNGRequired(const std::unordered_map<std::string, std::vector<Transpiler::Token>> &varInitialisers);

//--------------------------------------------------------------------------
//! \brief Is the variable name valid? GeNN variable names must obey C variable naming rules
//--------------------------------------------------------------------------
GENN_EXPORT void validateVarName(const std::string &name, const std::string &description);

//--------------------------------------------------------------------------
//! \brief Is the population name valid? GeNN population names obey C variable naming rules but can start with a number
//--------------------------------------------------------------------------
GENN_EXPORT void validatePopName(const std::string &name, const std::string &description);

//--------------------------------------------------------------------------
//! \brief Are all the parameter names in vector valid? GeNN variables and population names must obey C variable naming rules
//--------------------------------------------------------------------------
GENN_EXPORT void validateParamNames(const std::vector<std::string> &paramNames);

//--------------------------------------------------------------------------
//! \brief Are initialisers provided for all of the the item names in the vector?
//--------------------------------------------------------------------------
template<typename T, typename V>
void validateInitialisers(const std::vector<T> &vec, const std::unordered_map<std::string, V> &values, 
                          const std::string &type, const std::string description)
{
    // If there are a different number of sizes than values, give error
    if(vec.size() != values.size()) {
        throw std::runtime_error(description + " expected " + std::to_string(vec.size()) + " " + type + " but got " + std::to_string(values.size()));
    }

    // Loop through variables
    for(const auto &v : vec) {
        // If there is no values, give error
        if(values.find(v.name) == values.cend()) {
            throw std::runtime_error(description + " missing initialiser for " + type + ": '" + v.name + "'");
        }
    }
}

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
void writePreciseString(std::ostream &os, T value, int maxDigits10 = std::numeric_limits<T>::max_digits10)
{
    // Cache previous precision
    const std::streamsize previousPrecision = os.precision();

    // Set scientific formatting
    os << std::scientific;

    // Set precision
    os << std::setprecision(maxDigits10);

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
inline std::string writePreciseString(T value, int maxDigits10 = std::numeric_limits<T>::max_digits10)
{
    std::stringstream s;
    writePreciseString(s, value, maxDigits10);
    return s.str();
}

//! Boilerplate for overloading base std::visit
template<class... Ts> struct Overload : Ts... { using Ts::operator()...; };
template<class... Ts> Overload(Ts...) -> Overload<Ts...>; // line not needed in

//! Hash arithmetic types and enums
template<typename T, typename std::enable_if<std::is_arithmetic<T>::value || std::is_enum<T>::value>::type* = nullptr>
inline void updateHash(const T& value, boost::uuids::detail::sha1& hash)
{
    hash.process_bytes(&value, sizeof(T));
}

//! Hash monostate
inline void updateHash(std::monostate, boost::uuids::detail::sha1&)
{
}

//! Hash strings
inline void updateHash(const std::string &string, boost::uuids::detail::sha1 &hash)
{
    updateHash(string.size(), hash);
    hash.process_bytes(string.data(), string.size());
}

//! Hash arrays of types which can, themselves, be hashed
template<typename T, size_t N>
inline void updateHash(const std::array<T, N> &array, boost::uuids::detail::sha1 &hash)
{
    updateHash(array.size(), hash);
    for(const auto &v : array) {
        updateHash(v, hash);
    }
}

//! Hash vectors of types which can, themselves, be hashed
template<typename T>
inline void updateHash(const std::vector<T> &vector, boost::uuids::detail::sha1 &hash)
{
    updateHash(vector.size(), hash);
    for(const auto &v : vector) {
        updateHash(v, hash);
    }
}

//! Hash vectors of bools
inline void updateHash(const std::vector<bool> &vector, boost::uuids::detail::sha1 &hash)
{
    updateHash(vector.size(), hash);
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

//! Hash optional types which can, themeselves, be hashed
template<typename T>
inline void updateHash(const std::optional<T> &optional, boost::uuids::detail::sha1 &hash)
{
    updateHash(optional.has_value(), hash);
    if (optional) {
        updateHash(optional.value(), hash);
    }
}

//! Hash variants of types which can, themeselves, be hashed
template<typename... T>
inline void updateHash(const std::variant<T...> &variant, boost::uuids::detail::sha1 &hash)
{
    updateHash(variant.index(), hash);
    std::visit(
         Utils::Overload{
             [&hash](const auto &v)
             {
                 updateHash(v, hash);
             }},
        variant);
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
}   // namespace GeNN::Utils
