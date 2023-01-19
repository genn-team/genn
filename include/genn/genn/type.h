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

// Forward declarations
namespace GeNN::Type
{
struct UnaryVisitor;
struct BinaryVisitor;
}

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
    }

#define DECLARE_FOREIGN_FUNCTION_TYPE(TYPE, RETURN_TYPE, ...)                                               \
    class TYPE : public ForeignFunction<RETURN_TYPE, __VA_ARGS__>                                           \
    {                                                                                                       \
        DECLARE_TYPE(TYPE)                                                                                  \
        TYPE(Qualifier qualifiers = Qualifier{0}) : ForeignFunction<RETURN_TYPE, __VA_ARGS__>(qualifiers){} \
        virtual Base *getQualifiedType(Qualifier qualifiers) const final{ return new TYPE(qualifiers); }    \
    }

#define IMPLEMENT_TYPE(TYPE) TYPE *TYPE::s_Instance = NULL
#define IMPLEMENT_NUMERIC_TYPE(TYPE) IMPLEMENT_TYPE(TYPE)

//----------------------------------------------------------------------------
// GeNN::Type::TypeContext
//----------------------------------------------------------------------------
//! Map of 'typedef' names to concrete classes
namespace GeNN::Type
{
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
    virtual const Type::Base *accept(UnaryVisitor &visitor) const = 0;

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

//---------------------------------------------------------------------------
// GeNN::Type::UnaryAcceptable
//---------------------------------------------------------------------------
template<typename T, typename B>
class UnaryAcceptable : public B
{
public:
    UnaryAcceptable(Qualifier qualifiers = Qualifier{0}) : B(qualifiers)
    {
    }

    virtual const Type::Base *accept(UnaryVisitor &visitor) const final
    {
        return visitor.visit(static_cast<const T*>(this));
    }
};

//----------------------------------------------------------------------------
// GeNN::Type::Pointer
//----------------------------------------------------------------------------
//! Type representing a pointer
class Pointer : public UnaryAcceptable<Pointer, Base>
{
public:
    Pointer(const Base *valueType, Qualifier qualifiers = Qualifier{0})
    :   UnaryAcceptable<Pointer, Base>(qualifiers), m_ValueType(valueType)
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
class NumericBase : public UnaryAcceptable<NumericBase, ValueBase>
{
public:
    NumericBase(Qualifier qualifiers = Qualifier{0}) : UnaryAcceptable<NumericBase, ValueBase>(qualifiers){}
    
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

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    const Type::NumericBase *getNumeric(const TypeContext &context) const;

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::string m_Name;
};

//----------------------------------------------------------------------------
// GeNN::Type::ForeignFunctionBase
//----------------------------------------------------------------------------
class ForeignFunctionBase : public UnaryAcceptable<ForeignFunctionBase, Base>
{
public:
    ForeignFunctionBase(Qualifier qualifiers = Qualifier{0}) : UnaryAcceptable<ForeignFunctionBase, Base>(qualifiers){}

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual const NumericBase *getReturnType() const = 0;
    virtual std::vector<const NumericBase*> getArgumentTypes() const = 0;
};

//----------------------------------------------------------------------------
// GeNN::Type::ForeignFunction
//----------------------------------------------------------------------------
template<typename ReturnType, typename ...ArgTypes>
class ForeignFunction : public ForeignFunctionBase
{
public:
    ForeignFunction(Qualifier qualifiers = Qualifier{0}) : ForeignFunctionBase(qualifiers){}
    
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
    // ForeignFunctionBase virtuals
    //------------------------------------------------------------------------
    virtual const NumericBase *getReturnType() const final
    {
        return ReturnType::getInstance();
    }

    virtual std::vector<const NumericBase*> getArgumentTypes() const final
    {
        std::vector<const NumericBase*> args;
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
        if constexpr (sizeof...(Args)) {
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
        if constexpr (sizeof...(Args)) {
            typeName += ", ";
            updateResolvedTypeName<Args...>(context, typeName);
        }
    }

    template <typename T, typename... Args>
    static void updateArgumentTypes(std::vector<const NumericBase*> &args)
    {
        // Add argument typename to string
        args.push_back(T::getInstance());

        // If there are more arguments left in pack, recurse
        if constexpr (sizeof...(Args)) {
            updateArgumentTypes<Args...>(args);
        }
    }
};


//----------------------------------------------------------------------------
// GeNN::Type::UnaryVisitor
//----------------------------------------------------------------------------
//! Visitor class used for implementing logic on a single type
struct UnaryVisitor
{
    virtual const Type::Base *visit(const ForeignFunctionBase *function) = 0;
    virtual const Type::Base *visit(const Pointer *pointer) = 0;
    virtual const Type::Base *visit(const NumericBase *numeric) = 0;
};


//----------------------------------------------------------------------------
// GeNN::Type::BinaryTypeVisitor
//----------------------------------------------------------------------------
//! Visitor class used for implementing logic on pairs of types
struct BinaryTypeVisitor
{
    virtual void visit(const Pointer *pointerLeft, const Pointer &pointerRight) = 0;
    virtual void visit(const Pointer *pointerLeft, const NumericBase &numericRight) = 0;
    virtual void visit(const NumericBase *numericLeft, const NumericBase &numericRight) = 0;
    virtual void visit(const NumericBase *numericLeft, const Pointer &pointerRight) = 0;
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
// Declare standard library foreign function types
//----------------------------------------------------------------------------
DECLARE_FOREIGN_FUNCTION_TYPE(Exp, Double, Double);
DECLARE_FOREIGN_FUNCTION_TYPE(Sqrt, Double, Double);

//! Parse a numeric type
const NumericBase *parseNumeric(std::string_view typeString);

//! Look up numeric type based on set of type specifiers
const NumericBase *getNumericType(const std::set<std::string_view> &typeSpecifiers);

//! Apply C type promotion rules to numeric type
const NumericBase *getPromotedType(const NumericBase *type, const TypeContext &context);

//! Apply C rules to get common type between numeric types a and b
const NumericBase *getCommonType(const NumericBase *a, const NumericBase *b, const TypeContext &context);
}   // namespace GeNN::Type
