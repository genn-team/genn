#pragma once

// Standard C includes
#include <cassert>
#include <cstdint>

// Standard C++ includes
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

// GeNN includes
#include "gennExport.h"

//----------------------------------------------------------------------------
// GeNN::Type::Qualifier
//----------------------------------------------------------------------------
namespace GeNN::Type
{
enum class Qualifier : unsigned int
{
    CONSTANT   = (1 << 0)
};

inline bool operator & (Qualifier a, Qualifier b)
{
    return (static_cast<unsigned int>(a) & static_cast<unsigned int>(b)) != 0;
}

inline Qualifier operator | (Qualifier a, Qualifier b)
{
    return static_cast<Qualifier>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

//----------------------------------------------------------------------------
// GeNN::Type::Type
//----------------------------------------------------------------------------
struct Type
{
    //------------------------------------------------------------------------
    // Numeric
    //------------------------------------------------------------------------
    struct Numeric
    {
        const int rank;
        const double min;
        const double max;
        const double lowest;
        const int maxDigits10;

        const bool isSigned;
        const bool isIntegral;

        const std::string literalSuffix;
    };

    //------------------------------------------------------------------------
    // Pointer
    //------------------------------------------------------------------------
    struct Pointer
    {
        Pointer(const Type &valueType) : valueType(std::make_unique<Type const>(valueType))
        {}
        Pointer(const Pointer &other) : valueType(std::make_unique<Type const>(*other.valueType))
        {}
        
        const std::unique_ptr<Type const> valueType;
    };

    //------------------------------------------------------------------------
    // Function
    //------------------------------------------------------------------------
    /*struct Function
    {
        const std::unique_ptr<Type const> returnType;
        const std::vector<Type> argTypes;
    };*/
    
    Type(size_t size, Qualifier qualifiers, const Numeric &numeric)
    :   size(size), qualifiers(qualifiers), detail(numeric)
    {}
    Type(size_t size, Qualifier qualifiers, const Pointer &pointer)
    :   size(size), qualifiers(qualifiers), detail(pointer)
    {}
    Type(const Type &other) : size(other.size), qualifiers(qualifiers), detail(other.detail)
    {}
    Type(const Type other, Qualifier qualifiers) : size(other.size), qualifiers(qualifiers), detail(other.detail)
    {}

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const size_t size;
    
    const Qualifier qualifiers;

    const std::variant<Numeric, Pointer/*, Function*/> detail;
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const Numeric &getNumeric() const{ return std::get<Numeric>(detail); }
    const Pointer &getPointer() const{ return std::get<Pointer>(detail); }
    const Type addQualifier(Qualifier qualifier) const{ return Type(*this, qualifier); }

    bool operator == (const Type &other) const
    {
        return (size == other.size && qualifiers == other.qualifiers && detail == other.detail);
    }

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    template<typename T>
    static Type createNumeric(int rank, const std::string &literalSuffix = "", Qualifier qualifiers = Qualifier{0})
    {
        return Type(sizeof(T), qualifiers, Numeric{rank, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(),
                                                   std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max_digits10, 
                                                   std::is_signed<T>::value, std::is_integral<T>::value, literalSuffix});
    }
};

typedef std::unordered_map<std::string, const class Base*> TypeContext;

//----------------------------------------------------------------------------
// Declare numeric types
//----------------------------------------------------------------------------
inline static const Type Bool = Type::createNumeric<bool>(0);
inline static const Type Int8 = Type::createNumeric<int8_t>(10);
inline static const Type Int16 = Type::createNumeric<int16_t>(20);
inline static const Type Int32 = Type::createNumeric<int32_t>(30);
//DECLARE_NUMERIC_TYPE(Int64, int64_t, 40);
inline static const Type Uint8 = Type::createNumeric<uint8_t>(10, "u");
inline static const Type Uint16 = Type::createNumeric<uint16_t>(20, "u");
inline static const Type Uint32 = Type::createNumeric<uint32_t>(30, "u");
//DECLARE_NUMERIC_TYPE(Uint64, uint64_t, 40);
inline static const Type Float = Type::createNumeric<float>(50, "f");
inline static const Type Double = Type::createNumeric<double>(60);

//! Parse a numeric type
Type parseNumeric(const std::string &typeString);

//! Look up numeric type based on set of type specifiers
Type getNumericType(const std::set<std::string> &typeSpecifiers);

//! Apply C type promotion rules to numeric type
Type getPromotedType(const Type &type);

//! Apply C rules to get common type between numeric types a and b
Type getCommonType(const Type &a, const Type &b);


}   // namespace GeNN::Type
