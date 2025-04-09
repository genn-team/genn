#pragma once

// Standard C includes
#include <cassert>
#include <cmath>
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
        bool saturating;
        std::optional<int> fixedPoint;

        std::string literalSuffix;

        //------------------------------------------------------------------------
        // Operators
        //------------------------------------------------------------------------
        bool operator == (const Numeric &other) const
        {
            return (std::tie(rank, min, max, lowest, maxDigits10, isSigned, isIntegral, saturating, fixedPoint) 
                    == std::tie(other.rank, other.min, other.max, other.lowest, other.maxDigits10, other.isSigned, 
                                other.isIntegral, other.saturating, other.fixedPoint));
        }

        bool operator != (const Numeric &other) const
        {
            return (std::tie(rank, min, max, lowest, maxDigits10, isSigned, isIntegral, saturating, fixedPoint) 
                    != std::tie(other.rank, other.min, other.max, other.lowest, other.maxDigits10, other.isSigned,
                                other.isIntegral, other.saturating, other.fixedPoint));
        }

        bool operator < (const Numeric &other) const
        {
            return (std::tie(rank, min, max, lowest, maxDigits10, isSigned, isIntegral, saturating, fixedPoint)
                    < std::tie(other.rank, other.min, other.max, other.lowest, other.maxDigits10,
                               other.isSigned, other.isIntegral, other.saturating, other.fixedPoint));
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
        bool isWriteOnly;
        std::optional<Numeric> numeric;
        
        //------------------------------------------------------------------------
        // Operators
        //------------------------------------------------------------------------
        bool operator == (const Value &other) const
        {
            return (std::tie(size, numeric, device, isWriteOnly) == std::tie(other.size, other.numeric, other.device, other.isWriteOnly));
        }

        bool operator != (const Value &other) const
        {
            return (std::tie(size, numeric, device, isWriteOnly) != std::tie(other.size, other.numeric, other.device, other.isWriteOnly));
        }

        bool operator < (const Value &other) const
        {
            return (std::tie(size, numeric, device, isWriteOnly) < std::tie(other.size, other.numeric, other.device, other.isWriteOnly));
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
    bool isScalar() const{ return isPointer() || (isNumeric() && !getValue().isWriteOnly); }

    const Value &getValue() const{ return std::get<Value>(detail); }
    const Pointer &getPointer() const{ return std::get<Pointer>(detail); }
    const Function &getFunction() const{ return std::get<Function>(detail); }
    const Numeric &getNumeric() const{ return *getValue().numeric; }

    ResolvedType addConst() const{ return ResolvedType(*this, true); }
    ResolvedType removeConst() const{ return ResolvedType(*this, false); }
    ResolvedType addWriteOnly() const;

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
        return ResolvedType{Value{name, sizeof(T), ffiType, device, false, 
                                  Numeric{rank, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(),
                                          std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max_digits10,
                                          std::is_signed<T>::value, std::is_integral<T>::value, false, std::nullopt,
                                          literalSuffix}},
                            isConst};
    }

    template<typename T>
    static ResolvedType createFixedPointNumeric(const std::string &name, int rank, bool saturating, int fixedPoint, ffi_type *ffiType,
                                                const std::string &literalSuffix = "", bool isConst = false, bool device = false)
    {
        const double scale = 1.0 / std::pow(2.0, fixedPoint);
        return ResolvedType{Value{name, sizeof(T), ffiType, device, false, 
                                  Numeric{rank, std::numeric_limits<T>::min() * scale, std::numeric_limits<T>::max() * scale,
                                          std::numeric_limits<T>::lowest() * scale, 
                                          (int)std::ceil(std::numeric_limits<T>::digits * std::log10(2) + 1),
                                          std::is_signed<T>::value, false, saturating, fixedPoint, literalSuffix}},
                            isConst};
    }

    template<typename T>
    static ResolvedType createValue(const std::string &name, bool isConst = false, 
                                    ffi_type *ffiType = nullptr, bool device = false)
    {
        return ResolvedType{Value{name, sizeof(T), ffiType, device, false, std::nullopt}, isConst};
    }

    static ResolvedType createValue(const std::string &name, size_t size, bool isConst = false, 
                                    ffi_type *ffiType = nullptr, bool device = false)
    {
        return ResolvedType{Value{name, size, ffiType, device, false, std::nullopt}, isConst};
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

inline static const ResolvedType S0_15 = ResolvedType::createFixedPointNumeric<int16_t>("s0_15_t", 50, false, 15, &ffi_type_sint16, "");
inline static const ResolvedType S0_15Sat = ResolvedType::createFixedPointNumeric<int16_t>("s0_15_sat_t", 50, true, 15, &ffi_type_sint16, "");
inline static const ResolvedType S1_14 = ResolvedType::createFixedPointNumeric<int16_t>("s1_14_t", 51, false, 14, &ffi_type_sint16, "");
inline static const ResolvedType S1_14Sat = ResolvedType::createFixedPointNumeric<int16_t>("s1_14_sat_t", 51, true, 14, &ffi_type_sint16, "");
inline static const ResolvedType S2_13 = ResolvedType::createFixedPointNumeric<int16_t>("s2_13_t", 52, false, 13, &ffi_type_sint16, "");
inline static const ResolvedType S2_13Sat = ResolvedType::createFixedPointNumeric<int16_t>("s2_13_sat_t", 52, true, 13, &ffi_type_sint16, "");
inline static const ResolvedType S3_12 = ResolvedType::createFixedPointNumeric<int16_t>("s3_12_t", 53, false, 12, &ffi_type_sint16, "");
inline static const ResolvedType S3_12Sat = ResolvedType::createFixedPointNumeric<int16_t>("s3_12_sat_t", 53, true, 12, &ffi_type_sint16, "");
inline static const ResolvedType S4_11 = ResolvedType::createFixedPointNumeric<int16_t>("s4_11_t", 54, false, 11, &ffi_type_sint16, "");
inline static const ResolvedType S4_11Sat = ResolvedType::createFixedPointNumeric<int16_t>("s4_11_sat_t", 54, true, 11, &ffi_type_sint16, "");
inline static const ResolvedType S5_10 = ResolvedType::createFixedPointNumeric<int16_t>("s5_10_t", 55, false, 10, &ffi_type_sint16, "");
inline static const ResolvedType S5_10Sat = ResolvedType::createFixedPointNumeric<int16_t>("s5_10_sat_t", 55, true, 10, &ffi_type_sint16, "");
inline static const ResolvedType S6_9 = ResolvedType::createFixedPointNumeric<int16_t>("s6_9_t", 56, false, 9, &ffi_type_sint16, "");
inline static const ResolvedType S6_9Sat = ResolvedType::createFixedPointNumeric<int16_t>("s6_9_sat_t", 56, true, 9, &ffi_type_sint16, "");
inline static const ResolvedType S7_8 = ResolvedType::createFixedPointNumeric<int16_t>("s7_8_t", 57, false, 8, &ffi_type_sint16, "");
inline static const ResolvedType S7_8Sat = ResolvedType::createFixedPointNumeric<int16_t>("s7_8_sat_t", 57, true, 8, &ffi_type_sint16, "");
inline static const ResolvedType S8_7 = ResolvedType::createFixedPointNumeric<int16_t>("s8_7_t", 58, false, 7, &ffi_type_sint16, "");
inline static const ResolvedType S8_7Sat = ResolvedType::createFixedPointNumeric<int16_t>("s8_7_sat_t", 58, true, 7, &ffi_type_sint16, "");
inline static const ResolvedType S9_6 = ResolvedType::createFixedPointNumeric<int16_t>("s9_6_t", 59, false, 6, &ffi_type_sint16, "");
inline static const ResolvedType S9_6Sat = ResolvedType::createFixedPointNumeric<int16_t>("s9_6_sat_t", 59, true, 6, &ffi_type_sint16, "");
inline static const ResolvedType S10_5 = ResolvedType::createFixedPointNumeric<int16_t>("s10_5_t", 60, false, 5, &ffi_type_sint16, "");
inline static const ResolvedType S10_5Sat = ResolvedType::createFixedPointNumeric<int16_t>("s10_5_sat_t", 60, true, 5, &ffi_type_sint16, "");
inline static const ResolvedType S11_4 = ResolvedType::createFixedPointNumeric<int16_t>("s11_4_t", 61, false, 4, &ffi_type_sint16, "");
inline static const ResolvedType S11_4Sat = ResolvedType::createFixedPointNumeric<int16_t>("s11_4_sat_t", 61, false, 4, &ffi_type_sint16, "");
inline static const ResolvedType S12_3 = ResolvedType::createFixedPointNumeric<int16_t>("s12_3_t", 62, false, 3, &ffi_type_sint16, "");
inline static const ResolvedType S12_3Sat = ResolvedType::createFixedPointNumeric<int16_t>("s12_3_sat_t", 62, false, 3, &ffi_type_sint16, "");
inline static const ResolvedType S13_2 = ResolvedType::createFixedPointNumeric<int16_t>("s13_2_t", 63, false, 2, &ffi_type_sint16, "");
inline static const ResolvedType S13_2Sat = ResolvedType::createFixedPointNumeric<int16_t>("s13_2_sat_t", 63, false, 2, &ffi_type_sint16, "");
inline static const ResolvedType S14_1 = ResolvedType::createFixedPointNumeric<int16_t>("s14_1_t", 64, false, 1, &ffi_type_sint16, "");
inline static const ResolvedType S14_1Sat = ResolvedType::createFixedPointNumeric<int16_t>("s14_1_sat_t", 64, false, 1, &ffi_type_sint16, "");

inline static const ResolvedType Float = CREATE_NUMERIC(float, 80, &ffi_type_float, "f");
inline static const ResolvedType Double = CREATE_NUMERIC(double, 90, &ffi_type_double, "");

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

//! Get type for array subscript overload functions
inline ResolvedType getArraySubscript(ResolvedType valueType) { return ResolvedType::createFunction(valueType, {Int32}, FunctionFlags::ARRAY_SUBSCRIPT_OVERRIDE); }

//----------------------------------------------------------------------------
// Type helper functions
//----------------------------------------------------------------------------
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
