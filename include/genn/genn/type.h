#pragma once

// Standard C includes
#include <cassert>
#include <cstddef>
#include <cstdint>

// Standard C++ includes
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

// FFI includes
#include <ffi.h>

// Boost includes
#include <sha1.hpp>

// GeNN includes
#include "gennExport.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define CREATE_NUMERIC(TYPE, RANK, FFI_TYPE, L_SUFFIX) ResolvedType::createNumeric<TYPE>(#TYPE, RANK, FFI_TYPE, L_SUFFIX)

//----------------------------------------------------------------------------
// GeNN::Type::FunctionFlags
//----------------------------------------------------------------------------
namespace GeNN::Type
{
//! Flags that can be applied to function types
enum class FunctionFlags : unsigned int
{
    VARIADIC                    = (1 << 0),  //! Function is variadic
    ARRAY_SUBSCRIPT_OVERRIDE    = (1 << 1)   //! Function is used to override [] for something
};

inline bool operator & (FunctionFlags a, FunctionFlags b)
{
    return (static_cast<unsigned int>(a) & static_cast<unsigned int>(b)) != 0;
}

inline FunctionFlags operator | (FunctionFlags a, FunctionFlags b)
{
    return static_cast<FunctionFlags>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

//----------------------------------------------------------------------------
// GeNN::Type::NumericValue
//----------------------------------------------------------------------------
//! ResolvedType::Numeric has various values attached e.g. min and max. These
//! Cannot be represented using any single type (double can't represent all uint64_t for example)
//! Therefore, this type is used as a wrapper.
class GENN_EXPORT NumericValue
{
public:
    NumericValue(double value) : m_Value(value){}
    NumericValue(uint64_t value) : m_Value(value){}
    NumericValue(int64_t value) : m_Value(value){}
    NumericValue(int value) : m_Value(int64_t{value}){}
    NumericValue(unsigned int value) : m_Value(uint64_t{value}){}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    template<typename T>
    T cast() const
    { 
        return std::visit(
            [](auto x)->T
            {
                return static_cast<T>(x);
            },
            m_Value);
    }
    const std::variant<double, uint64_t, int64_t> &get() const{ return m_Value; }

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    bool operator == (const NumericValue &other) const;
    bool operator != (const NumericValue &other) const;
    bool operator < (const NumericValue &other) const;
    bool operator > (const NumericValue &other) const;
    bool operator <= (const NumericValue &other) const;
    bool operator >= (const NumericValue &other) const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::variant<double, uint64_t, int64_t> m_Value;
};


//----------------------------------------------------------------------------
// GeNN::Type::ResolvedType
//----------------------------------------------------------------------------
struct GENN_EXPORT ResolvedType
{
    //------------------------------------------------------------------------
    // Numeric
    //------------------------------------------------------------------------
    struct Numeric
    {
        int rank;
        NumericValue min;
        NumericValue max;
        NumericValue lowest;
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
        ffi_type *ffiType;
        bool device;
        std::optional<Numeric> numeric;
        
        //------------------------------------------------------------------------
        // Operators
        //------------------------------------------------------------------------
        bool operator == (const Value &other) const
        {
            return (std::tie(size, numeric, device) == std::tie(other.size, other.numeric, other.device));
        }

        bool operator != (const Value &other) const
        {
            return (std::tie(size, numeric, device) != std::tie(other.size, other.numeric, other.device));
        }

        bool operator < (const Value &other) const
        {
            return (std::tie(size, numeric, device) < std::tie(other.size, other.numeric, other.device));
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
        Function(const ResolvedType &returnType, const std::vector<ResolvedType> &argTypes, 
                 FunctionFlags flags = FunctionFlags{0}) 
        :   returnType(std::make_unique<ResolvedType const>(returnType)), argTypes(argTypes), flags(flags)
        {}
        Function(const Function &other)
        :   returnType(std::make_unique<ResolvedType const>(*other.returnType)), 
            argTypes(other.argTypes), flags(other.flags)
        {}

        std::unique_ptr<ResolvedType const> returnType;
        std::vector<ResolvedType> argTypes;
        FunctionFlags flags;

        bool operator == (const Function &other) const
        {
            return (std::tie(*returnType, argTypes, flags) 
                    == std::tie(*other.returnType, other.argTypes, other.flags));
        }

        bool operator != (const Function &other) const
        {
            return (std::tie(*returnType, argTypes, flags) 
                    != std::tie(*other.returnType, other.argTypes, other.flags));
        }

        bool operator < (const Function &other) const
        {
            return (std::tie(*returnType, argTypes, flags) 
                    < std::tie(*other.returnType, other.argTypes, other.flags));
        }

        Function &operator = (const Function &other)
        {
           returnType.reset(new ResolvedType(*other.returnType));
           argTypes = other.argTypes;
           flags = other.flags;
           return *this;
        }

        bool hasFlag(FunctionFlags flag) const
        {
            return (flags & flag);
        }
    };
    
    ResolvedType(const Value &value, bool isConst = false)
    :   isConst(isConst), detail(value)
    {}
    ResolvedType(const Pointer &pointer, bool isConst = false)
    :   isConst(isConst), detail(pointer)
    {}
    ResolvedType(const Function &function)
    :   isConst(false), detail(function)
    {}
    ResolvedType(const ResolvedType &other, bool isConst) : isConst(isConst), detail(other.detail)
    {}
    explicit ResolvedType(bool isConst = false) : isConst(isConst), detail(std::monostate{})
    {}

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    bool isConst;

    std::variant<Value, Pointer, Function, std::monostate> detail;
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool isValue() const{ return std::holds_alternative<Value>(detail); }
    bool isPointer() const{ return std::holds_alternative<Pointer>(detail); }
    bool isPointerToPointer() const{ return isPointer() && getPointer().valueType->isPointer(); }
    bool isFunction() const{ return std::holds_alternative<Function>(detail); }
    bool isNumeric() const{ return isValue() && getValue().numeric; }
    bool isVoid() const{ return std::holds_alternative<std::monostate>(detail); }

    const Value &getValue() const{ return std::get<Value>(detail); }
    const Pointer &getPointer() const{ return std::get<Pointer>(detail); }
    const Function &getFunction() const{ return std::get<Function>(detail); }
    const Numeric &getNumeric() const{ return *getValue().numeric; }

    ResolvedType addConst() const{ return ResolvedType(*this, true); }
    ResolvedType removeConst() const{ return ResolvedType(*this, false); }

    std::string getName() const;
    size_t getSize(size_t pointerBytes) const;

    ffi_type *getFFIType() const;

    ResolvedType createPointer(bool isConst = false) const
    {
        return ResolvedType(Pointer{*this}, isConst);
    }

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    bool operator == (const ResolvedType &other) const
    {
        return (std::tie(isConst, detail) == std::tie(other.isConst, other.detail));
    }

    bool operator != (const ResolvedType &other) const
    {
        return (std::tie(isConst, detail) != std::tie(other.isConst, other.detail));
    }

    bool operator < (const ResolvedType &other) const
    {
        return (std::tie(isConst, detail) < std::tie(other.isConst, other.detail));
    }

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    template<typename T>
    static ResolvedType createNumeric(const std::string &name, int rank, ffi_type *ffiType, 
                                      const std::string &literalSuffix = "", bool isConst = false, bool device = false)
    {
        return ResolvedType{Value{name, sizeof(T), ffiType, device, Numeric{rank, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(),
                                                                            std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max_digits10,
                                                                            std::is_signed<T>::value, std::is_integral<T>::value, literalSuffix}},
                            isConst};
    }

    template<typename T>
    static ResolvedType createValue(const std::string &name, bool isConst = false, 
                                    ffi_type *ffiType = nullptr, bool device = false)
    {
        return ResolvedType{Value{name, sizeof(T), ffiType, device, std::nullopt}, isConst};
    }

    static ResolvedType createFunction(const ResolvedType &returnType, const std::vector<ResolvedType> &argTypes,
                                       FunctionFlags flags=FunctionFlags{0})
    {
        return ResolvedType{Function{returnType, argTypes, flags}, false};
    }
};

typedef std::unordered_map<std::string, ResolvedType> TypeContext;

//----------------------------------------------------------------------------
// UnresolvedType
//----------------------------------------------------------------------------
struct GENN_EXPORT UnresolvedType
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
inline static const ResolvedType Bool = CREATE_NUMERIC(bool, 0, nullptr, "");
inline static const ResolvedType Int8 = CREATE_NUMERIC(int8_t, 10, &ffi_type_sint8, "");
inline static const ResolvedType Int16 = CREATE_NUMERIC(int16_t, 20, &ffi_type_sint16, "");
inline static const ResolvedType Int32 = CREATE_NUMERIC(int32_t, 30, &ffi_type_sint32, "");
inline static const ResolvedType Int64 = CREATE_NUMERIC(int64_t, 40, &ffi_type_sint32, "");

inline static const ResolvedType Uint8 = CREATE_NUMERIC(uint8_t, 10, &ffi_type_uint8, "u");
inline static const ResolvedType Uint16 = CREATE_NUMERIC(uint16_t, 20, &ffi_type_uint16, "u");
inline static const ResolvedType Uint32 = CREATE_NUMERIC(uint32_t, 30, &ffi_type_uint32, "u");
inline static const ResolvedType Uint64 = CREATE_NUMERIC(uint64_t, 40, &ffi_type_uint64, "u");

inline static const ResolvedType Float = CREATE_NUMERIC(float, 50, &ffi_type_float, "f");
inline static const ResolvedType Double = CREATE_NUMERIC(double, 60, &ffi_type_double, "");

// Void
inline static const ResolvedType Void = ResolvedType();

//----------------------------------------------------------------------------
// Standard function types
//----------------------------------------------------------------------------
inline static const ResolvedType AllocatePushPullEGP = ResolvedType::createFunction(Void, {Uint32});
inline static const ResolvedType PushPull = ResolvedType::createFunction(Void, {});
inline static const ResolvedType Assert = ResolvedType::createFunction(Void, {Bool});

//! Get type to add a weight type
inline ResolvedType getAddToPrePost(ResolvedType weightType) { return ResolvedType::createFunction(Void, {weightType}); }

//! Get type to add a weight type with delay
inline ResolvedType getAddToPrePostDelay(ResolvedType weightType) { return ResolvedType::createFunction(Void, {weightType, Uint32}); }

//! Apply C type promotion rules to numeric type
GENN_EXPORT ResolvedType getPromotedType(const ResolvedType &type);

//! Apply C rules to get common type between numeric types a and b
GENN_EXPORT ResolvedType getCommonType(const ResolvedType &a, const ResolvedType &b);

//! Write numeric value to string, formatting correctly for type
GENN_EXPORT std::string writeNumeric(const NumericValue &value, const ResolvedType &type);

//! Serialise numeric value to bytes
GENN_EXPORT void serialiseNumeric(const NumericValue &value, const ResolvedType &type, std::vector<std::byte> &bytes);

//----------------------------------------------------------------------------
// updateHash overrides
//----------------------------------------------------------------------------
GENN_EXPORT void updateHash(const NumericValue &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const ResolvedType::Numeric &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const ResolvedType::Value &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const ResolvedType::Pointer &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const ResolvedType::Function &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const ResolvedType &v, boost::uuids::detail::sha1 &hash);
GENN_EXPORT void updateHash(const UnresolvedType &v, boost::uuids::detail::sha1 &hash);
}   // namespace GeNN::Type
