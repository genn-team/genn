#pragma once

// Standard C includes
#include <cassert>
#include <cstdint>

// Standard C++ includes
#include <limits>
#include <memory>
#include <optional>
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
// Macros
//----------------------------------------------------------------------------
#define CREATE_NUMERIC(TYPE, RANK, L_SUFFIX) Type::createNumeric<TYPE>(#TYPE, RANK, L_SUFFIX)

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
        int rank;
        double min;
        double max;
        double lowest;
        int maxDigits10;

        bool isSigned;
        bool isIntegral;

        std::string literalSuffix;

        //------------------------------------------------------------------------
        // Operators
        //------------------------------------------------------------------------
        bool operator == (const Numeric &other) const
        {
            return (std::tie(rank, min, max, lowest, maxDigits10, isSigned, isIntegral) 
                    == std::tie(other.rank, other.min, other.max, other.lowest, other.maxDigits10, other.isSigned, other.isIntegral));
        }

        bool operator < (const Numeric &other) const
        {
            return (std::tie(rank, min, max, lowest, maxDigits10, isSigned, isIntegral) 
                    < std::tie(other.rank, other.min, other.max, other.lowest, other.maxDigits10, other.isSigned, other.isIntegral));
        }
    };

    //------------------------------------------------------------------------
    // Value
    //------------------------------------------------------------------------
    struct Value
    {
        std::string name;
        std::optional<Numeric> numeric;
        
        //------------------------------------------------------------------------
        // Operators
        //------------------------------------------------------------------------
        bool operator == (const Value &other) const
        {
            return (numeric == other.numeric);
        }

        bool operator < (const Value &other) const
        {
            return (numeric < other.numeric);
        }
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
        
        std::unique_ptr<Type const> valueType;

        bool operator == (const Pointer &other) const
        {
            return (*valueType == *other.valueType);
        }

        bool operator < (const Pointer &other) const
        {
            return (*valueType < *other.valueType);
        }

        Pointer &operator = (const Pointer &other)
        {
           valueType.reset(new Type(*other.valueType));
           return *this;
        }
    };

    //------------------------------------------------------------------------
    // Function
    //------------------------------------------------------------------------
    struct Function
    {
        Function(const Type &returnType, const std::vector<Type> &argTypes) 
        :   returnType(std::make_unique<Type const>(returnType)), argTypes(argTypes)
        {}
        Function(const Function &other)
        :   returnType(std::make_unique<Type const>(*other.returnType)), argTypes(other.argTypes)
        {}

        std::unique_ptr<Type const> returnType;
        std::vector<Type> argTypes;

        bool operator == (const Function &other) const
        {
            return std::tie(*returnType, argTypes) == std::tie(*other.returnType, other.argTypes);
        }

        bool operator < (const Function &other) const
        {
            return std::tie(*returnType, argTypes) < std::tie(*other.returnType, other.argTypes);
        }

        Function &operator = (const Function &other)
        {
           returnType.reset(new Type(*other.returnType));
           argTypes = other.argTypes;
           return *this;
        }
    };
    
    Type(size_t size, Qualifier qualifiers, const Value &numeric)
    :   size(size), qualifiers(qualifiers), detail(numeric)
    {}
    Type(Qualifier qualifiers, const Pointer &pointer)
    :   size(sizeof(char*)), qualifiers(qualifiers), detail(pointer)
    {}
    Type(const Function &function)
        : size(0), qualifiers(Qualifier{0}), detail(function)
    {}
    Type(const Type &other, Qualifier qualifiers) : size(other.size), qualifiers(qualifiers), detail(other.detail)
    {}

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    size_t size;
    Qualifier qualifiers;

    std::variant<Value, Pointer, Function> detail;
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool isValue() const{ return std::holds_alternative<Value>(detail); }
    bool isPointer() const{ return std::holds_alternative<Pointer>(detail); }
    bool isFunction() const{ return std::holds_alternative<Function>(detail); }
    bool isNumeric() const{ return isValue() && getValue().numeric; }
    const Value &getValue() const{ return std::get<Value>(detail); }
    const Pointer &getPointer() const{ return std::get<Pointer>(detail); }
    const Function &getFunction() const{ return std::get<Function>(detail); }
    const Numeric &getNumeric() const{ return *getValue().numeric; }

    const Type addQualifier(Qualifier qualifier) const{ return Type(*this, qualifier); }
    bool hasQualifier(Qualifier qualifier) const{ return (qualifiers & qualifier); }

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    bool operator == (const Type &other) const
    {
        return (std::tie(size, qualifiers, detail) 
                == std::tie(other.size, other.qualifiers, other.detail));
    }

    bool operator < (const Type &other) const
    {
        return (std::tie(size, qualifiers, detail) 
                < std::tie(other.size, other.qualifiers, other.detail));
    }

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    template<typename T>
    static Type createNumeric(const std::string &name, int rank, const std::string &literalSuffix = "", Qualifier qualifiers = Qualifier{0})
    {
        return Type{sizeof(T), qualifiers, Value{name, Numeric{rank, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(),
                                                               std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max_digits10,
                                                               std::is_signed<T>::value, std::is_integral<T>::value, literalSuffix}}};
    }

    static Type createPointer(const Type &valueType, Qualifier qualifiers = Qualifier{0})
    {
         return Type(qualifiers, Pointer{valueType});
    }
};

typedef std::unordered_map<std::string, const class Base*> TypeContext;

//----------------------------------------------------------------------------
// Declare numeric types
//----------------------------------------------------------------------------
inline static const Type Bool = CREATE_NUMERIC(bool, 0, "");
inline static const Type Int8 = CREATE_NUMERIC(int8_t, 10, "");
inline static const Type Int16 = CREATE_NUMERIC(int16_t, 20, "");
inline static const Type Int32 = CREATE_NUMERIC(int32_t, 30, "");
//DECLARE_NUMERIC_TYPE(Int64, int64_t, 40);
inline static const Type Uint8 = CREATE_NUMERIC(uint8_t, 10, "u");
inline static const Type Uint16 = CREATE_NUMERIC(uint16_t, 20, "u");
inline static const Type Uint32 = CREATE_NUMERIC(uint32_t, 30, "u");
//DECLARE_NUMERIC_TYPE(Uint64, uint64_t, 40);
inline static const Type Float = CREATE_NUMERIC(float, 50, "f");
inline static const Type Double = CREATE_NUMERIC(double, 60, "");

//! Parse a numeric type
Type parseNumeric(const std::string &typeString);

//! Look up numeric type based on set of type specifiers
Type getNumericType(const std::set<std::string> &typeSpecifiers);

//! Apply C type promotion rules to numeric type
Type getPromotedType(const Type &type);

//! Apply C rules to get common type between numeric types a and b
Type getCommonType(const Type &a, const Type &b);


}   // namespace GeNN::Type
