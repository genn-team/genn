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
const std::map<std::set<std::string_view>, const Type::NumericBase*> numericTypeSpecifiers{
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
const std::set<std::string_view> scalarTypeSpecifier{{"scalar"}};
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
IMPLEMENT_NUMERIC_TYPE(Bool);
IMPLEMENT_NUMERIC_TYPE(Int8);
IMPLEMENT_NUMERIC_TYPE(Int16);
IMPLEMENT_NUMERIC_TYPE(Int32);
IMPLEMENT_NUMERIC_TYPE(Uint8);
IMPLEMENT_NUMERIC_TYPE(Uint16);
IMPLEMENT_NUMERIC_TYPE(Uint32);
IMPLEMENT_NUMERIC_TYPE(Float);
IMPLEMENT_NUMERIC_TYPE(Double);

// Implement foreign function types
IMPLEMENT_TYPE(Exp);
IMPLEMENT_TYPE(Sqrt);

//----------------------------------------------------------------------------
// GeNN::Type::Base
//----------------------------------------------------------------------------
const Base *Base::getPointerType(Qualifier qualifiers) const 
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
    return getNumeric(context)->getResolvedName(context);
}
//----------------------------------------------------------------------------
size_t NumericTypedef::getSizeBytes(const TypeContext &context) const
{
    return getNumeric(context)->getSizeBytes(context);
}
//----------------------------------------------------------------------------
Base *NumericTypedef::getQualifiedType(Qualifier qualifiers) const
{ 
    return new NumericTypedef(m_Name, qualifiers); 
} 
//----------------------------------------------------------------------------
int NumericTypedef::getRank(const TypeContext &context) const
{
    return getNumeric(context)->getRank(context);
}
//----------------------------------------------------------------------------
double NumericTypedef::getMin(const TypeContext &context) const
{
    return getNumeric(context)->getMin(context);
}
//----------------------------------------------------------------------------
double NumericTypedef::getMax(const TypeContext &context) const
{
    return getNumeric(context)->getMax(context);
}
//----------------------------------------------------------------------------
double NumericTypedef::getLowest(const TypeContext &context) const
{
    return getNumeric(context)->getLowest(context);
}
//----------------------------------------------------------------------------
int NumericTypedef::getMaxDigits10(const TypeContext &context) const
{
    return getNumeric(context)->getMaxDigits10(context);
}
//----------------------------------------------------------------------------
bool NumericTypedef::isSigned(const TypeContext &context) const
{
    return getNumeric(context)->getSizeBytes(context);
}
//----------------------------------------------------------------------------
bool NumericTypedef::isIntegral(const TypeContext &context) const
{
    return getNumeric(context)->isIntegral(context);
}
//----------------------------------------------------------------------------
std::string NumericTypedef::getLiteralSuffix(const TypeContext &context) const
{
    return getNumeric(context)->getLiteralSuffix(context);
}
//----------------------------------------------------------------------------
const Type::NumericBase *NumericTypedef::getNumeric(const TypeContext &context) const
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
// Free functions
//----------------------------------------------------------------------------
const NumericBase *parseNumeric(std::string_view typeString)
{
    using namespace Transpiler;

    // Scan type
    SingleLineErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource(typeString, errorHandler);

    // Parse type and cast to numeric
    const auto *type = dynamic_cast<const NumericBase*>(Parser::parseType(tokens, false, errorHandler));

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
const Pointer *parseNumericPtr(std::string_view typeString)
{
    using namespace Transpiler;

    // Scan type
    SingleLineErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource(typeString, errorHandler);

    // Parse type and cast to numeric pointer
    const auto *type = dynamic_cast<const Pointer*>(Parser::parseType(tokens, true, errorHandler));

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
const NumericBase *getNumericType(const std::set<std::string_view> &typeSpecifiers)
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
