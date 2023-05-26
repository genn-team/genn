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
#include "gennUtils.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define CREATE_NUMERIC(TYPE, RANK, L_SUFFIX) ResolvedType::createNumeric<TYPE>(#TYPE, RANK, L_SUFFIX)

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
// GeNN::Type::ResolvedType
//----------------------------------------------------------------------------
struct ResolvedType
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

        bool operator != (const Numeric &other) const
        {
            return (std::tie(rank, min, max, lowest, maxDigits10, isSigned, isIntegral) 
                    != std::tie(other.rank, other.min, other.max, other.lowest, other.maxDigits10, other.isSigned, other.isIntegral));
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
        size_t size;
        std::optional<Numeric> numeric;
        
        //------------------------------------------------------------------------
        // Operators
        //------------------------------------------------------------------------
        bool operator == (const Value &other) const
        {
            return (std::tie(size, numeric) == std::tie(other.size, other.numeric));
        }

        bool operator != (const Value &other) const
        {
            return (std::tie(size, numeric) != std::tie(other.size, other.numeric));
        }

        bool operator < (const Value &other) const
        {
            return (std::tie(size, numeric) < std::tie(other.size, other.numeric));
        }
    };

    //------------------------------------------------------------------------
    // Pointer
    //------------------------------------------------------------------------
    struct Pointer
    {
        Pointer(const ResolvedType &valueType) : valueType(std::make_unique<ResolvedType const>(valueType))
        {}
        Pointer(const Pointer &other) : valueType(std::make_unique<ResolvedType const>(*other.valueType))
        {}
        
        std::unique_ptr<ResolvedType const> valueType;

        bool operator == (const Pointer &other) const
        {
            return (*valueType == *other.valueType);
        }

        bool operator != (const Pointer &other) const
        {
            return (*valueType != *other.valueType);
        }

        bool operator < (const Pointer &other) const
        {
            return (*valueType < *other.valueType);
        }

        Pointer &operator = (const Pointer &other)
        {
           valueType.reset(new ResolvedType(*other.valueType));
           return *this;
        }
    };

    //------------------------------------------------------------------------
    // Function
    //------------------------------------------------------------------------
    struct Function
    {
        Function(const ResolvedType &returnType, const std::vector<ResolvedType> &argTypes) 
        :   returnType(std::make_unique<ResolvedType const>(returnType)), argTypes(argTypes)
        {}
        Function(const Function &other)
        :   returnType(std::make_unique<ResolvedType const>(*other.returnType)), argTypes(other.argTypes)
        {}

        std::unique_ptr<ResolvedType const> returnType;
        std::vector<ResolvedType> argTypes;

        bool operator == (const Function &other) const
        {
            return (std::tie(*returnType, argTypes) == std::tie(*other.returnType, other.argTypes));
        }

        bool operator != (const Function &other) const
        {
            return (std::tie(*returnType, argTypes) != std::tie(*other.returnType, other.argTypes));
        }

        bool operator < (const Function &other) const
        {
            return (std::tie(*returnType, argTypes) < std::tie(*other.returnType, other.argTypes));
        }

        Function &operator = (const Function &other)
        {
           returnType.reset(new ResolvedType(*other.returnType));
           argTypes = other.argTypes;
           return *this;
        }
    };
    
    ResolvedType(const Value &value, Qualifier qualifiers = Qualifier{0})
    :   qualifiers(qualifiers), detail(value)
    {}
    ResolvedType(const Pointer &pointer, Qualifier qualifiers = Qualifier{0})
    :   qualifiers(qualifiers), detail(pointer)
    {}
    ResolvedType(const Function &function)
    :   qualifiers(Qualifier{0}), detail(function)
    {}
    ResolvedType(const ResolvedType &other, Qualifier qualifiers) : qualifiers(qualifiers), detail(other.detail)
    {}

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
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

    const ResolvedType addQualifier(Qualifier qualifier) const{ return ResolvedType(*this, qualifiers | qualifier); }
    bool hasQualifier(Qualifier qualifier) const{ return (qualifiers & qualifier); }

    std::string getName() const;
    size_t getSize(size_t pointerBytes) const;

    ResolvedType createPointer(Qualifier qualifiers = Qualifier{0}) const
    {
        return ResolvedType(Pointer{*this}, qualifiers);
    }

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    bool operator == (const ResolvedType &other) const
    {
        return (std::tie(qualifiers, detail) == std::tie(other.qualifiers, other.detail));
    }

    bool operator != (const ResolvedType &other) const
    {
        return (std::tie(qualifiers, detail) != std::tie(other.qualifiers, other.detail));
    }

    bool operator < (const ResolvedType &other) const
    {
        return (std::tie(qualifiers, detail) < std::tie(other.qualifiers, other.detail));
    }

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    template<typename T>
    static ResolvedType createNumeric(const std::string &name, int rank, const std::string &literalSuffix = "", Qualifier qualifiers = Qualifier{0})
    {
        return ResolvedType{Value{name, sizeof(T), Numeric{rank, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(),
                                                           std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max_digits10,
                                                           std::is_signed<T>::value, std::is_integral<T>::value, literalSuffix}},
                            qualifiers};
    }

    template<typename T>
    static ResolvedType createValue(const std::string &name, Qualifier qualifiers = Qualifier{0})
    {
        return ResolvedType{Value{name, sizeof(T), std::nullopt}, qualifiers};
    }

    static ResolvedType createFunction(const ResolvedType &returnType, const std::vector<ResolvedType> &argTypes)
    {
        return ResolvedType{Function{returnType, argTypes}, Qualifier{0}};
    }
};

typedef std::unordered_map<std::string, ResolvedType> TypeContext;

//----------------------------------------------------------------------------
// UnresolvedType
//----------------------------------------------------------------------------
struct UnresolvedType
{
    UnresolvedType(const ResolvedType &type)
    :   detail(type)
    {}
    UnresolvedType(const std::string &name)
        : detail(name)
    {}

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::variant<ResolvedType, std::string> detail;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    ResolvedType resolve(const TypeContext &typeContext) const;

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    bool operator == (const UnresolvedType &other) const
    {
        return (detail == other.detail);
    }

    bool operator != (const UnresolvedType &other) const
    {
        return (detail != other.detail);
    }

    bool operator < (const UnresolvedType &other) const
    {
        return (detail < other.detail);
    }
};


//----------------------------------------------------------------------------
// Declare numeric types
//----------------------------------------------------------------------------
inline static const ResolvedType Bool = CREATE_NUMERIC(bool, 0, "");
inline static const ResolvedType Int8 = CREATE_NUMERIC(int8_t, 10, "");
inline static const ResolvedType Int16 = CREATE_NUMERIC(int16_t, 20, "");
inline static const ResolvedType Int32 = CREATE_NUMERIC(int32_t, 30, "");
//DECLARE_NUMERIC_TYPE(Int64, int64_t, 40);
inline static const ResolvedType Uint8 = CREATE_NUMERIC(uint8_t, 10, "u");
inline static const ResolvedType Uint16 = CREATE_NUMERIC(uint16_t, 20, "u");
inline static const ResolvedType Uint32 = CREATE_NUMERIC(uint32_t, 30, "u");
//DECLARE_NUMERIC_TYPE(Uint64, uint64_t, 40);
inline static const ResolvedType Float = CREATE_NUMERIC(float, 50, "f");
inline static const ResolvedType Double = CREATE_NUMERIC(double, 60, "");

//! Parse a numeric type
GENN_EXPORT ResolvedType parseNumeric(const std::string &typeString, const TypeContext &context);

//! Look up numeric type based on set of type specifiers
GENN_EXPORT ResolvedType getNumericType(const std::set<std::string> &typeSpecifiers, const TypeContext &context);

//! Apply C type promotion rules to numeric type
GENN_EXPORT ResolvedType getPromotedType(const ResolvedType &type);

//! Apply C rules to get common type between numeric types a and b
GENN_EXPORT ResolvedType getCommonType(const ResolvedType &a, const ResolvedType &b);

//----------------------------------------------------------------------------
// updateHash overrides
//----------------------------------------------------------------------------
GENN_EXPORT void updateHash(const ResolvedType::Numeric &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const ResolvedType::Value &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const ResolvedType::Pointer &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const ResolvedType::Function &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const ResolvedType &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const UnresolvedType &v, boost::uuids::detail::sha1 &hash);
}   // namespace GeNN::Type
