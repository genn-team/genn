#include "type.h"

// Standard C++ includes
#include <map>
#include <unordered_map>

// GeNN includes
#include "logging.h"

// Transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/parser.h"
#include "transpiler/scanner.h"

using namespace GeNN;

// Anonymous namespace
namespace
{
const std::map<std::set<std::string>, const Type::NumericBase*> numericTypeSpecifiers{
    {{"char"}, Type::Int8::getInstance()},
    {{"int8_t"}, Type::Int8::getInstance()},
    
    {{"unsigned", "char"}, Type::Uint8::getInstance()},
    {{"uint8_t"}, Type::Uint8::getInstance()},

    {{"short"}, Type::Int16::getInstance()},
    {{"short", "int"}, Type::Int16::getInstance()},
    {{"signed", "short"}, Type::Int16::getInstance()},
    {{"signed", "short", "int"}, Type::Int16::getInstance()},
    {{"int16_t"}, Type::Int16::getInstance()},
    
    {{"unsigned", "short"}, Type::Uint16::getInstance()},
    {{"unsigned", "short", "int"}, Type::Uint16::getInstance()},
    {{"uint16_t"}, Type::Uint8::getInstance()},

    {{"int"}, Type::Int32::getInstance()},
    {{"signed"}, Type::Int32::getInstance()},
    {{"signed", "int"}, Type::Int32::getInstance()},
    {{"int32_t"}, Type::Int32::getInstance()},

    {{"unsigned"}, Type::Uint32::getInstance()},
    {{"unsigned", "int"}, Type::Uint32::getInstance()},
    {{"uint32_t"}, Type::Uint32::getInstance()},

    {{"float"}, Type::Float::getInstance()},
    {{"double"}, Type::Double::getInstance()},
};
//----------------------------------------------------------------------------
const std::set<std::string> scalarTypeSpecifier{{"scalar"}};
//----------------------------------------------------------------------------
// Mapping of signed integer numericTypeSpecifiers to their unsigned equivalents
const std::unordered_map<const Type::NumericBase*, const Type::NumericBase*> unsignedType{
    {Type::Int8::getInstance(), Type::Uint8::getInstance()},
    {Type::Int16::getInstance(), Type::Uint16::getInstance()},
    {Type::Int32::getInstance(), Type::Uint32::getInstance()}
};
}   // Anonymous namespace

//----------------------------------------------------------------------------
// GeNN::Type
//----------------------------------------------------------------------------
namespace GeNN::Type
{
// Implement numeric types
IMPLEMENT_TYPE(Bool);
IMPLEMENT_TYPE(Int8);
IMPLEMENT_TYPE(Int16);
IMPLEMENT_TYPE(Int32);
IMPLEMENT_TYPE(Uint8);
IMPLEMENT_TYPE(Uint16);
IMPLEMENT_TYPE(Uint32);
IMPLEMENT_TYPE(Float);
IMPLEMENT_TYPE(Double);

IMPLEMENT_TYPE(PrintF);

// Implement trigonometric functions
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Cos);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Sin);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Tan);

// Implement inverse trigonometric functions
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Acos);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Asin);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Atan);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Atan2);

// Implement hyperbolic  functions
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Cosh);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Sinh);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Tanh);

// Implement inverse hyperbolic functions
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Acosh);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Asinh);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Atanh);

// Implement exponential functions
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Exp);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(ExpM1);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Exp2);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Pow);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(ScalBN);

// Implement logarithm functions
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Log);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Log1P);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Log2);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Log10);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(LdExp);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(ILogB);

// Implement root functions
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Sqrt);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Cbrt);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Hypot);

// Implement rounding functions
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Ceil);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Floor);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Fmod);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Round);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Rint);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Trunc);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(NearbyInt);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(NextAfter);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Remainder);

// Implement range functions
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(FAbs);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(FDim);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(FMax);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(FMin);

// Implement other functions
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(Erf);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(ErfC);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(TGamma);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(LGamma);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(CopySign);
IMPLEMENT_FLOAT_DOUBLE_FUNCTION_TYPE(FMA);


//----------------------------------------------------------------------------
// GeNN::Type::Base
//----------------------------------------------------------------------------
const Pointer *Base::getPointerType(Qualifier qualifiers) const 
{ 
    // **TODO** befriend constructor
    // **TODO** don't just leak these!
    return new Pointer(this, qualifiers); 
}

//----------------------------------------------------------------------------
// GeNN::Type::NumericTypedef
//----------------------------------------------------------------------------
std::string NumericTypedef::getResolvedName(const TypeContext &context) const
{
    return getResolvedType(context)->getResolvedName(context);
}
//----------------------------------------------------------------------------
size_t NumericTypedef::getSizeBytes(const TypeContext &context) const
{
    return getResolvedType(context)->getSizeBytes(context);
}
//----------------------------------------------------------------------------
Base *NumericTypedef::getQualifiedType(Qualifier qualifiers) const
{ 
    return new NumericTypedef(m_Name, qualifiers); 
} 
//----------------------------------------------------------------------------
int NumericTypedef::getRank(const TypeContext &context) const
{
    return getResolvedType(context)->getRank(context);
}
//----------------------------------------------------------------------------
double NumericTypedef::getMin(const TypeContext &context) const
{
    return getResolvedType(context)->getMin(context);
}
//----------------------------------------------------------------------------
double NumericTypedef::getMax(const TypeContext &context) const
{
    return getResolvedType(context)->getMax(context);
}
//----------------------------------------------------------------------------
double NumericTypedef::getLowest(const TypeContext &context) const
{
    return getResolvedType(context)->getLowest(context);
}
//----------------------------------------------------------------------------
int NumericTypedef::getMaxDigits10(const TypeContext &context) const
{
    return getResolvedType(context)->getMaxDigits10(context);
}
//----------------------------------------------------------------------------
bool NumericTypedef::isSigned(const TypeContext &context) const
{
    return getResolvedType(context)->getSizeBytes(context);
}
//----------------------------------------------------------------------------
bool NumericTypedef::isIntegral(const TypeContext &context) const
{
    return getResolvedType(context)->isIntegral(context);
}
//----------------------------------------------------------------------------
std::string NumericTypedef::getLiteralSuffix(const TypeContext &context) const
{
    return getResolvedType(context)->getLiteralSuffix(context);
}
//----------------------------------------------------------------------------
const Type::NumericBase *NumericTypedef::getResolvedType(const TypeContext &context) const
{
    const auto t = context.find(m_Name);
    if (t == context.cend()) {
        throw std::runtime_error("No context for typedef '" + m_Name + "'");
    }
    else {
        const NumericBase *numericType = dynamic_cast<const NumericBase*>(t->second);
        if (numericType) {
            return numericType;
        }
        else {
            throw std::runtime_error("Numeric typedef '" + m_Name + "' resolved to non-numeric type '" + t->second->getName() + "'");
        }
    }
}

//----------------------------------------------------------------------------
// GeNN::Type::FunctionBase
//----------------------------------------------------------------------------
bool FunctionBase::isVariadic() const
{
    // If variadic marker (nullptr) isn't found, function isn't variadic
    const auto argTypes = getArgumentTypes();
    const auto variadicMarker = std::find(argTypes.cbegin(), argTypes.cend(), nullptr);
    if(variadicMarker == argTypes.cend()) {
        return false;
    }
    // Otherwise, after checking variadic marker is last argument, return true
    else {
        assert(argTypes.back() == nullptr);
        return true;
    }

}

//----------------------------------------------------------------------------
// Free functions
//----------------------------------------------------------------------------
const NumericBase *parseNumeric(const std::string &typeString)
{
    using namespace Transpiler;

    // Scan type
    SingleLineErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource(typeString, errorHandler);

    // Parse type numeric type
    const auto *type = Parser::parseNumericType(tokens, errorHandler);

    // If an error was encountered while scanning or parsing, throw exception
    if (errorHandler.hasError()) {
        throw std::runtime_error("Error parsing type '" + std::string{typeString} + "'");
    }

    // If tokens did not contain a valid numeric type, throw exception
    if (!type) {
        throw std::runtime_error("Unable to parse type '" + std::string{typeString} + "'");
    }
    return type;
}
//----------------------------------------------------------------------------
const NumericBase *getNumericType(const std::set<std::string> &typeSpecifiers)
{
    // If type matches scalar type specifiers
    if(typeSpecifiers == scalarTypeSpecifier) {
        return new NumericTypedef("scalar");
    }
    // Otherwise
    else {
        const auto type = numericTypeSpecifiers.find(typeSpecifiers);
        return (type == numericTypeSpecifiers.cend()) ? nullptr : type->second;
    }
}
//----------------------------------------------------------------------------
const NumericBase *getPromotedType(const NumericBase *type, const TypeContext &context)
{
    // If a small integer type is used in an expression, it is implicitly converted to int which is always signed. 
    // This is known as the integer promotions or the integer promotion rule 
    // **NOTE** this is true because in our type system unsigned short is uint16 which can be represented in int32
    if(type->getRank(context) < Int32::getInstance()->getRank(context)) {
        return Int32::getInstance();
    }
    else {
        return type;
    }
}
//----------------------------------------------------------------------------
const NumericBase *getCommonType(const NumericBase *a, const NumericBase *b, const TypeContext &context)
{
    // If either type is double, common type is double
    const auto &aTypeName = a->getResolvedName(context);
    const auto &bTypeName = b->getResolvedName(context);
    if(aTypeName == Double::getInstance()->getName() || bTypeName == Double::getInstance()->getName()) {
        return Double::getInstance();
    }
    // Otherwise, if either type is float, common type is float
    if(aTypeName == Float::getInstance()->getName() || bTypeName == Float::getInstance()->getName()) {
        return Float::getInstance();
    }
    // Otherwise, must be an integer type
    else {
        // Promote both numeric types
        const auto *aPromoted = getPromotedType(a, context);
        const auto *bPromoted = getPromotedType(b, context);

        // If both promoted operands have the same type, then no further conversion is needed.
        if(aPromoted->getResolvedName(context) == bPromoted->getResolvedName(context)) {
            return aPromoted;
        }
        // Otherwise, if both promoted operands have signed integer numeric types or both have unsigned integer numeric types, 
        // the operand with the type of lesser integer conversion rank is converted to the type of the operand with greater rank.
        else if(aPromoted->isSigned(context) == bPromoted->isSigned(context)) {
            return (aPromoted->getRank(context) > bPromoted->getRank(context)) ? aPromoted : bPromoted;
        }
        // Otherwise, if signedness of promoted operands differ
        else {
            const auto *signedOp = aPromoted->isSigned(context) ? aPromoted : bPromoted;
            const auto *unsignedOp = aPromoted->isSigned(context) ? bPromoted : aPromoted;

            // Otherwise, if the operand that has unsigned integer type has rank greater or equal to the rank of the type of the other operand, 
            // then the operand with signed integer type is converted to the type of the operand with unsigned integer type.
            if(unsignedOp->getRank(context) >= signedOp->getRank(context)) {
                return unsignedOp;
            }
            // Otherwise, if the type of the operand with signed integer type can represent all of the values of the type of the operand with unsigned integer type, 
            // then the operand with unsigned integer type is converted to the type of the operand with signed integer type.
            else if((signedOp->getMin(context) <= unsignedOp->getMin(context))
                    && (signedOp->getMax(context) >= unsignedOp->getMax(context))) 
            {
                return signedOp;
            }
            // Otherwise, both operands are converted to the unsigned integer type corresponding to the type of the operand with signed integer type.
            else {
                return unsignedType.at(signedOp);
            }
        }
    }
}
}   // namespace GeNN::Type
