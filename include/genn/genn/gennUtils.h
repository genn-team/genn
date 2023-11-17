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
#include "initVarSnippet.h"
#include "type.h"

// GeNN code generator includes
#include "transpiler/token.h"

//--------------------------------------------------------------------------
// GeNN::Utils
//--------------------------------------------------------------------------
namespace GeNN::Utils
{
//! Helper to scan a multi-line code string, giving meaningful errors with the specified context string
GENN_EXPORT std::vector<Transpiler::Token> scanCode(const std::string &code, const std::string &errorContext);

//! Helper to scan a type specifier string e.g "unsigned int" and parse it into a resolved type 
GENN_EXPORT Type::ResolvedType parseNumericType(const std::string &type, const Type::TypeContext &typeContext);

//! Is this sequence of tokens empty?
/*! For ease of parsing and as an extra check that we have scanned SOMETHING, 
    empty token sequences should have a single EOF token */
GENN_EXPORT bool areTokensEmpty(const std::vector<Transpiler::Token> &tokens);

//! Checks whether the sequence of token references a given identifier
GENN_EXPORT bool isIdentifierReferenced(const std::string &identifierName, const std::vector<Transpiler::Token> &tokens);

//! Checks whether the sequence of token includes an RNG function identifier
GENN_EXPORT bool isRNGRequired(const std::vector<Transpiler::Token> &tokens);

//! Checks whether any of the variable initialisers in the vector require an RNG for initialisation
GENN_EXPORT bool isRNGRequired(const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers);

//! Checks variable name is valid? GeNN variable names must obey C variable naming rules
GENN_EXPORT void validateVarName(const std::string &name, const std::string &description);

//! Checks whether population name is valid? GeNN population names obey C variable naming rules but can start with a number
GENN_EXPORT void validatePopName(const std::string &name, const std::string &description);

//! Extra global parameters used to support both pointer and non-pointer types. Now only the behaviour that used to
//! be provided by pointer types is provided but, internally, non-pointer types are used. This handles pointer types specified by string.
GENN_EXPORT std::string handleLegacyEGPType(const std::string &type);

//! Count leading zeros
GENN_EXPORT int clz(unsigned int value);

//! Checks that initialisers provided for all of the the item names in the vector?
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

//! Checks whether the 'name' fields of all structs in vector valid? GeNN variables and population names must obey C variable naming rules
template<typename T>
void validateVecNames(const std::vector<T> &vec, const std::string &description)
{
    for(const auto &v : vec) {
        validateVarName(v.name, description);
    }
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
        [&hash](const auto &v)
        {
            updateHash(v, hash);
        },
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
