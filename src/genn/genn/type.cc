#include "type.h"

// Standard C++ includes
#include <map>
#include <unordered_map>

// Anonymous namespace
namespace
{
const std::map<std::set<std::string_view>, const Type::NumericBase*> numericTypes{
    {{"char"}, Type::Int8::getInstance()},
    
    {{"unsigned", "char"}, Type::Uint8::getInstance()},

    {{"short"}, Type::Int16::getInstance()},
    {{"short", "int"}, Type::Int16::getInstance()},
    {{"signed", "short"}, Type::Int16::getInstance()},
    {{"signed", "short", "int"}, Type::Int16::getInstance()},
    
    {{"unsigned", "short"}, Type::Uint16::getInstance()},
    {{"unsigned", "short", "int"}, Type::Uint16::getInstance()},

    {{"int"}, Type::Int32::getInstance()},
    {{"signed"}, Type::Int32::getInstance()},
    {{"signed", "int"}, Type::Int32::getInstance()},

    {{"unsigned"}, Type::Uint32::getInstance()},
    {{"unsigned", "int"}, Type::Uint32::getInstance()},

    {{"float"}, Type::Float::getInstance()},
    {{"double"}, Type::Double::getInstance()},
};
//----------------------------------------------------------------------------
// Mapping of signed integer numericTypes to their unsigned equivalents
const std::unordered_map<const Type::NumericBase*, const Type::NumericBase*> unsignedType{
    {Type::Int8::getInstance(), Type::Uint8::getInstance()},
    {Type::Int16::getInstance(), Type::Uint16::getInstance()},
    {Type::Int32::getInstance(), Type::Uint32::getInstance()}
};
}   // Anonymous namespace

//----------------------------------------------------------------------------
// Type
//----------------------------------------------------------------------------
namespace Type
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
// Free functions
//----------------------------------------------------------------------------
const NumericBase *getNumericType(const std::set<std::string_view> &typeSpecifiers)
{
    const auto type = numericTypes.find(typeSpecifiers);
    return (type == numericTypes.cend()) ? nullptr : type->second;
}
//----------------------------------------------------------------------------
const NumericPtrBase *getNumericPtrType(const std::set<std::string_view> &typeSpecifiers)
{
    const auto type = numericTypes.find(typeSpecifiers);
    return (type == numericTypes.cend()) ? nullptr : type->second->getPointerType();
}
//----------------------------------------------------------------------------
const NumericBase *getPromotedType(const NumericBase *type)
{
    // If a small integer type is used in an expression, it is implicitly converted to int which is always signed. 
    // This is known as the integer promotions or the integer promotion rule 
    // **NOTE** this is true because in our type system unsigned short is uint16 which can be represented in int32
    if(type->getRank() < Int32::getInstance()->getRank()) {
        return Int32::getInstance();
    }
    else {
        return type;
    }
}
//----------------------------------------------------------------------------
const NumericBase *getCommonType(const NumericBase *a, const NumericBase *b)
{
    // If either type is double, common type is double
    const auto aTypeHash = a->getTypeHash();
    const auto bTypeHash = b->getTypeHash();
    if(aTypeHash == Double::getInstance()->getTypeHash() || bTypeHash == Double::getInstance()->getTypeHash()) {
        return Double::getInstance();
    }
    // Otherwise, if either type is float, common type is float
    if(aTypeHash == Float::getInstance()->getTypeHash() || bTypeHash == Float::getInstance()->getTypeHash()) {
        return Float::getInstance();
    }
    // Otherwise, must be an integer type
    else {
        // Promote both numericTypes
        const auto *aPromoted = getPromotedType(a);
        const auto *bPromoted = getPromotedType(b);

        // If both promoted operands have the same type, then no further conversion is needed.
        if(aPromoted->getTypeHash() == bPromoted->getTypeHash()) {
            return aPromoted;
        }
        // Otherwise, if both promoted operands have signed integer numericTypes or both have unsigned integer numericTypes, 
        // the operand with the type of lesser integer conversion rank is converted to the type of the operand with greater rank.
        else if(aPromoted->isSigned() == bPromoted->isSigned()) {
            return (aPromoted->getRank() > bPromoted->getRank()) ? aPromoted : bPromoted;
        }
        // Otherwise, if signedness of promoted operands differ
        else {
            const auto *signedOp = aPromoted->isSigned() ? aPromoted : bPromoted;
            const auto *unsignedOp = aPromoted->isSigned() ? bPromoted : aPromoted;

            // Otherwise, if the operand that has unsigned integer type has rank greater or equal to the rank of the type of the other operand, 
            // then the operand with signed integer type is converted to the type of the operand with unsigned integer type.
            if(unsignedOp->getRank() >= signedOp->getRank()) {
                return unsignedOp;
            }
            // Otherwise, if the type of the operand with signed integer type can represent all of the values of the type of the operand with unsigned integer type, 
            // then the operand with unsigned integer type is converted to the type of the operand with signed integer type.
            else if(signedOp->getMin() <= unsignedOp->getMin() && signedOp->getMax() >= unsignedOp->getMax()) {
                return signedOp;
            }
            // Otherwise, both operands are converted to the unsigned integer type corresponding to the type of the operand with signed integer type.
            else {
                return unsignedType.at(signedOp);
            }
        }
    }
}
}