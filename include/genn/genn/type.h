#pragma once

// Standard C includes
#include <cassert>
#include <cstdint>

// Standard C++ includes
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <typeinfo>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// GeNN includes
#include "gennExport.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DECLARE_TYPE(TYPE)                          \
    private:                                        \
        GENN_EXPORT static TYPE *s_Instance;        \
    public:                                         \
        static const TYPE *getInstance()            \
        {                                           \
            if(s_Instance == NULL)                  \
            {                                       \
                s_Instance = new TYPE;              \
            }                                       \
            return s_Instance;                      \
        }

#define DECLARE_NUMERIC_TYPE(TYPE, UNDERLYING_TYPE, RANK, LITERAL_SUFFIX)                                       \
    class TYPE : public Numeric<UNDERLYING_TYPE, RANK>                                                          \
    {                                                                                                           \
        DECLARE_TYPE(TYPE)                                                                                      \
        TYPE(Qualifier qualifiers = Qualifier{0}) : Numeric<UNDERLYING_TYPE, RANK>(qualifiers){}                \
        virtual std::string getName() const final{ return #UNDERLYING_TYPE; }                                   \
        virtual std::string getResolvedName(const TypeContext&) const final{ return #UNDERLYING_TYPE; }         \
        virtual Base *getQualifiedType(Qualifier qualifiers) const final{ return new TYPE(qualifiers); }        \
        virtual std::string getLiteralSuffix(const TypeContext&) const final{ return LITERAL_SUFFIX; } \
    };                                                                                                          \
    template<>                                                                                                  \
    struct TypeTraits<UNDERLYING_TYPE>                                                                          \
    {                                                                                                           \
        using NumericType = TYPE;                                                                               \
    }                                                                      

#define DECLARE_FUNCTION_TYPE(TYPE, RETURN_TYPE, ...)                                                       \
    class TYPE : public Function<RETURN_TYPE, __VA_ARGS__>                                                  \
    {                                                                                                       \
        DECLARE_TYPE(TYPE)                                                                                  \
        TYPE(Qualifier qualifiers = Qualifier{0}) : Function<RETURN_TYPE, __VA_ARGS__>(qualifiers){}        \
        virtual Base *getQualifiedType(Qualifier qualifiers) const final{ return new TYPE(qualifiers); }    \
    }

#define IMPLEMENT_TYPE(TYPE) TYPE *TYPE::s_Instance = NULL

//! Helper macro to declare single and double precision one argument function types
#define DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(TYPE)    \
    DECLARE_FUNCTION_TYPE(TYPE##F, Float, Float);           \
    DECLARE_FUNCTION_TYPE(TYPE##D, Double, Double)

//! Helper macro to declare single and double precision two argument function types
#define DECLARE_TWO_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(TYPE)    \
    DECLARE_FUNCTION_TYPE(TYPE##F, Float, Float, Float);    \
    DECLARE_FUNCTION_TYPE(TYPE##D, Double, Double, Double)

//! Helper macro to declare single and double precision three argument function types
#define DECLARE_THREE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(TYPE)      \
    DECLARE_FUNCTION_TYPE(TYPE##F, Float, Float, Float, Float); \
    DECLARE_FUNCTION_TYPE(TYPE##D, Double, Double, Double, Double)

//! Helper macro to implement single and double precision function types
#define IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(TYPE)  \
    IMPLEMENT_TYPE(TYPE##F);                        \
    IMPLEMENT_TYPE(TYPE##D)

//----------------------------------------------------------------------------
// GeNN::Type::TypeTraits
//----------------------------------------------------------------------------
namespace GeNN::Type
{
//! Empty type trait structure
template<typename T>
struct TypeTraits
{
};

typedef std::unordered_map<std::string, const class Base*> TypeContext;

//----------------------------------------------------------------------------
// GeNN::Type::Qualifier
//----------------------------------------------------------------------------
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
// GeNN::Type::Base
//----------------------------------------------------------------------------
//! Base class for all types
class Base
{
public:
    Base(Qualifier qualifiers = Qualifier{0}) : m_Qualifiers(qualifiers){}

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //! Get the (unqualified) name of this type
    virtual std::string getName() const = 0;

    //! Get fully-resolved (unqualified) name of this type
    virtual std::string getResolvedName(const TypeContext &context) const = 0;
    
    //! Get size of this type in bytes
    virtual size_t getSizeBytes(const TypeContext &context) const = 0;

    //! Return new version of this type with specified qualifiers
    virtual Base *getQualifiedType(Qualifier qualifiers) const = 0;
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Return a pointer to this type, optionally, with specified qualifiers
    const class Pointer *getPointerType(Qualifier qualifiers = Qualifier{0}) const;
    
    //! Does this type have qualifier?
    bool hasQualifier(Qualifier qualifier) const{ return (m_Qualifiers & qualifier); };
    
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //! Bitfield of qualifiers
    const Qualifier m_Qualifiers;
};

//----------------------------------------------------------------------------
// GeNN::Type::Pointer
//----------------------------------------------------------------------------
//! Type representing a pointer
class Pointer : public Base
{
public:
    Pointer(const Base *valueType, Qualifier qualifiers = Qualifier{0})
    :   Base(qualifiers), m_ValueType(valueType)
    {
    }

    //------------------------------------------------------------------------
    // Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getName() const{ return getValueType()->getName() + "*";}
    virtual std::string getResolvedName(const TypeContext &context) const{ return getValueType()->getResolvedName(context) + "*"; }
    virtual size_t getSizeBytes(const TypeContext&) const final{ return sizeof(char*); }
    virtual Base *getQualifiedType(Qualifier qualifiers) const final{ return new Pointer(m_ValueType, qualifiers); }
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const Base *getValueType() const{ return m_ValueType; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const Base *m_ValueType;
};

//----------------------------------------------------------------------------
// GeNN::Type::ValueBase
//----------------------------------------------------------------------------
class ValueBase : public Base
{
public:
    ValueBase(Qualifier qualifiers = Qualifier{0}) : Base(qualifiers){}
};

//----------------------------------------------------------------------------
// GeNN::Type::NumericBase
//----------------------------------------------------------------------------
class NumericBase : public ValueBase
{
public:
    NumericBase(Qualifier qualifiers = Qualifier{0}) : ValueBase(qualifiers){}
    
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual int getRank(const TypeContext&) const = 0;
    virtual double getMin(const TypeContext&) const = 0;
    virtual double getMax(const TypeContext&) const = 0;
    virtual double getLowest(const TypeContext&) const = 0;
    virtual int getMaxDigits10(const TypeContext&) const = 0;

    virtual bool isSigned(const TypeContext&) const = 0;
    virtual bool isIntegral(const TypeContext&) const = 0;

    virtual std::string getLiteralSuffix(const TypeContext&) const = 0;
};

//----------------------------------------------------------------------------
// GeNN::Type::Numeric
//----------------------------------------------------------------------------
template<typename T, int Rank>
class Numeric : public NumericBase
{
public:
    Numeric(Qualifier qualifiers = Qualifier{0}) : NumericBase(qualifiers){}

    //------------------------------------------------------------------------
    // Base virtuals
    //------------------------------------------------------------------------
    virtual size_t getSizeBytes(const TypeContext&) const final{ return sizeof(T); }

    //------------------------------------------------------------------------
    // NumericBase virtuals
    //------------------------------------------------------------------------
    virtual int getRank(const TypeContext&) const final { return Rank; }
    virtual double getMin(const TypeContext&) const final { return std::numeric_limits<T>::min(); }
    virtual double getMax(const TypeContext&) const final { return std::numeric_limits<T>::max(); }
    virtual double getLowest(const TypeContext&) const final { return std::numeric_limits<T>::lowest(); }
    virtual int getMaxDigits10(const TypeContext&) const final{ return std::numeric_limits<T>::max_digits10; }

    virtual bool isSigned(const TypeContext&) const final { return std::is_signed<T>::value; }
    virtual bool isIntegral(const TypeContext&) const final { return std::is_integral<T>::value; }
};

//----------------------------------------------------------------------------
// GeNN::Type::NumericTypedef
//----------------------------------------------------------------------------
class NumericTypedef : public NumericBase
{
public:
    NumericTypedef(const std::string &name, Qualifier qualifiers = Qualifier{0}) 
    :   NumericBase(qualifiers), m_Name(name){}

    //------------------------------------------------------------------------
    // Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getName() const final{ return m_Name; }
    virtual std::string getResolvedName(const TypeContext &context) const;
    virtual size_t getSizeBytes(const TypeContext &context) const final;
    virtual Base *getQualifiedType(Qualifier qualifiers) const final;

    //------------------------------------------------------------------------
    // NumericBase virtuals
    //------------------------------------------------------------------------
    virtual int getRank(const TypeContext &context) const final;
    virtual double getMin(const TypeContext &context) const final;
    virtual double getMax(const TypeContext &context) const final;
    virtual double getLowest(const TypeContext &context) const final;
    virtual int getMaxDigits10(const TypeContext &context) const final;
    
    virtual bool isSigned(const TypeContext &context) const final;
    virtual bool isIntegral(const TypeContext &context) const final;

    virtual std::string getLiteralSuffix(const TypeContext &context) const final;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const Type::NumericBase *getResolvedType(const TypeContext &context) const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::string m_Name;
};

//----------------------------------------------------------------------------
// GeNN::Type::FunctionBase
//----------------------------------------------------------------------------
class FunctionBase : public Base
{
public:
    FunctionBase(Qualifier qualifiers = Qualifier{0}) : Base(qualifiers){}

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual const Base *getReturnType() const = 0;
    virtual std::vector<const Base*> getArgumentTypes() const = 0;
};

//----------------------------------------------------------------------------
// GeNN::Type::Function
//----------------------------------------------------------------------------
template<typename ReturnType, typename ...ArgTypes>
class Function : public FunctionBase
{
public:
    Function(Qualifier qualifiers = Qualifier{0}) : FunctionBase(qualifiers){}
    
    //------------------------------------------------------------------------
    // Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getName() const final
    {
        std::string typeName = getReturnType()->getName() + "(";
        updateTypeName<ArgTypes...>(typeName);
        typeName += ")";
        return typeName;
    }

    virtual std::string getResolvedName(const TypeContext &context) const final
    {
        std::string typeName = getReturnType()->getResolvedName(context) + "(";
        updateResolvedTypeName<ArgTypes...>(context, typeName);
        typeName += ")";
        return typeName;
    }
    
    virtual size_t getSizeBytes(const TypeContext&) const final
    {
        assert(false);
        return 0;
    }

    //------------------------------------------------------------------------
    // FunctionBase virtuals
    //------------------------------------------------------------------------
    virtual const Base *getReturnType() const final
    {
        return ReturnType::getInstance();
    }

    virtual std::vector<const Base*> getArgumentTypes() const final
    {
        std::vector<const Base*> args;
        args.reserve(sizeof...(ArgTypes));
        updateArgumentTypes<ArgTypes...>(args);
        return args;
    }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    template <typename T, typename... Args>
    static void updateTypeName(std::string &typeName)
    {
        // Add argument typename to string
        typeName += T::getInstance()->getName();

        // If there are more arguments left in pack, add comma and recurse
        if constexpr (sizeof...(Args) > 0) {
            typeName += ", ";
            updateTypeName<Args...>(typeName);
        }
    }

    template <typename T, typename... Args>
    static void updateResolvedTypeName(const TypeContext &context, std::string &typeName)
    {
        // Add argument typename to string
        typeName += T::getInstance()->getResolvedName(context);

        // If there are more arguments left in pack, add comma and recurse
        if constexpr (sizeof...(Args) > 0) {
            typeName += ", ";
            updateResolvedTypeName<Args...>(context, typeName);
        }
    }

    template <typename T, typename... Args>
    static void updateArgumentTypes(std::vector<const Base*> &args)
    {
        // Add argument typename to string
        args.push_back(T::getInstance());

        // If there are more arguments left in pack, recurse
        if constexpr (sizeof...(Args) > 0) {
            updateArgumentTypes<Args...>(args);
        }
    }
};

//----------------------------------------------------------------------------
// Declare numeric types
//----------------------------------------------------------------------------
DECLARE_NUMERIC_TYPE(Bool, bool, 0, "");
DECLARE_NUMERIC_TYPE(Int8, int8_t, 10, "");
DECLARE_NUMERIC_TYPE(Int16, int16_t, 20, "");
DECLARE_NUMERIC_TYPE(Int32, int32_t, 30, "");
//DECLARE_NUMERIC_TYPE(Int64, int64_t, 40);
DECLARE_NUMERIC_TYPE(Uint8, uint8_t, 10, "u");
DECLARE_NUMERIC_TYPE(Uint16, uint16_t, 20, "u");
DECLARE_NUMERIC_TYPE(Uint32, uint32_t, 30, "u");
//DECLARE_NUMERIC_TYPE(Uint64, uint64_t, 40);
DECLARE_NUMERIC_TYPE(Float, float, 50, "f");
DECLARE_NUMERIC_TYPE(Double, double, 60, "");

//----------------------------------------------------------------------------
// Declare standard library function types
//----------------------------------------------------------------------------
// Trigonometric functions
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Cos);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Sin);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Tan);

// Inverse trigonometric functions
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Acos);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Asin);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Atan);
DECLARE_TWO_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Atan2);

// Hyperbolic functions
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Cosh);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Sinh);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Tanh);

// Inverse Hyperbolic functions
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Acosh);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Asinh);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Atanh);

// Exponential functions
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Exp);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(ExpM1);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Exp2);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Pow);
DECLARE_FUNCTION_TYPE(ScalBNF, Float, Float, Int32);
DECLARE_FUNCTION_TYPE(ScalBND, Double, Double, Int32);

// Logarithm functions
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Log);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Log1P);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Log2);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Log10);
DECLARE_FUNCTION_TYPE(LdExpF, Float, Float, Int32);
DECLARE_FUNCTION_TYPE(LdExpD, Double, Double, Int32);
DECLARE_FUNCTION_TYPE(ILogBF, Int32, Float);
DECLARE_FUNCTION_TYPE(ILogBD, Int32, Double);

// Root functions
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Sqrt);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Cbrt);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Hypot);

// Rounding functions
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Ceil);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Floor);
DECLARE_TWO_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Fmod);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Round);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Rint);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Trunc);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(NearbyInt);
DECLARE_TWO_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(NextAfter);
DECLARE_TWO_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Remainder);

// Range functions
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(FAbs);
DECLARE_TWO_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(FDim);
DECLARE_TWO_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(FMax);
DECLARE_TWO_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(FMin);

// Other functions
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(Erf);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(ErfC);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(TGamma);
DECLARE_ONE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(LGamma);
DECLARE_TWO_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(CopySign);
DECLARE_THREE_ARG_FLOAT_DOUBLE_FUNCTION_TYPE(FMA);
/*{,
{"frexp", "frexpf"},    // pointer arguments
{"modf", "modff"},      // pointer arguments
{"scalbln", "scalblnf"},    // long type
{"lround", "lroundf"},  // long return type
{"lrint", "lrintf"},    // long return type
{"remquo", "remquof"},  // pointer arguments
*/
//! Parse a numeric type
const NumericBase *parseNumeric(const std::string &typeString);

//! Look up numeric type based on set of type specifiers
const NumericBase *getNumericType(const std::set<std::string> &typeSpecifiers);

//! Apply C type promotion rules to numeric type
const NumericBase *getPromotedType(const NumericBase *type, const TypeContext &context);

//! Apply C rules to get common type between numeric types a and b
const NumericBase *getCommonType(const NumericBase *a, const NumericBase *b, const TypeContext &context);

// **YUCK** unimplemented stream operator so we get linker errors if you try and write types directly to an IO stream
std::ostream& operator<<(std::ostream &stream, const Base* value);
}   // namespace GeNN::Type
