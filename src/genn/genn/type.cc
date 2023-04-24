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
const std::map<std::set<std::string>, Type::Type> numericTypeSpecifiers{
    {{"char"}, Type::Int8},
    {{"int8_t"}, Type::Int8},
    
    {{"unsigned", "char"}, Type::Uint8},
    {{"uint8_t"}, Type::Uint8},

    {{"short"}, Type::Int16},
    {{"short", "int"}, Type::Int16},
    {{"signed", "short"}, Type::Int16},
    {{"signed", "short", "int"}, Type::Int16},
    {{"int16_t"}, Type::Int16},
    
    {{"unsigned", "short"}, Type::Uint16},
    {{"unsigned", "short", "int"}, Type::Uint16},
    {{"uint16_t"}, Type::Uint8},

    {{"int"}, Type::Int32},
    {{"signed"}, Type::Int32},
    {{"signed", "int"}, Type::Int32},
    {{"int32_t"}, Type::Int32},

    {{"unsigned"}, Type::Uint32},
    {{"unsigned", "int"}, Type::Uint32},
    {{"uint32_t"}, Type::Uint32},

    {{"float"}, Type::Float},
    {{"double"}, Type::Double}};
//----------------------------------------------------------------------------
const std::set<std::string> scalarTypeSpecifier{{"scalar"}};
//----------------------------------------------------------------------------
// Mapping of signed integer numericTypeSpecifiers to their unsigned equivalents
const std::unordered_map<Type::Type, Type::Type> unsignedType{
    {Type::Int8, Type::Uint8},
    {Type::Int16, Type::Uint16},
    {Type::Int32, Type::Uint32}};
}   // Anonymous namespace

//----------------------------------------------------------------------------
// GeNN::Type
//----------------------------------------------------------------------------
namespace GeNN::Type
{
//----------------------------------------------------------------------------
// Free functions
//----------------------------------------------------------------------------
Type parseNumeric(const std::string &typeString)
{
    using namespace Transpiler;

    // Scan type
    SingleLineErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource(typeString, errorHandler);

    // Parse type numeric type
    const auto type = Parser::parseNumericType(tokens, errorHandler);

    // If an error was encountered while scanning or parsing, throw exception
    if (errorHandler.hasError()) {
        throw std::runtime_error("Error parsing type '" + std::string{typeString} + "'");
    }

    return type;
}
//----------------------------------------------------------------------------
Type getNumericType(const std::set<std::string> &typeSpecifiers)
{
    // If type matches scalar type specifiers
    if(typeSpecifiers == scalarTypeSpecifier) {
        //return new NumericTypedef("scalar");
    }
    // Otherwise
    else {
        const auto type = numericTypeSpecifiers.find(typeSpecifiers);
        //return (type == numericTypeSpecifiers.cend()) ? nullptr : type->second;
        return type->second;
    }
}
//----------------------------------------------------------------------------
Type getPromotedType(const Type &type)
{
    // If a small integer type is used in an expression, it is implicitly converted to int which is always signed. 
    // This is known as the integer promotions or the integer promotion rule 
    // **NOTE** this is true because in our type system unsigned short is uint16 which can be represented in int32
    if(type.getNumeric().rank < Int32.getNumeric().rank) {
        return Int32;
    }
    else {
        return type;
    }
}
//----------------------------------------------------------------------------
Type getCommonType(const Type &a, const Type &b)
{
    // If either type is double, common type is double
    if(a == Double || b == Double) {
        return Double;
    }
    // Otherwise, if either type is float, common type is float
    if(a == Float || b == Float) {
        return Float;
    }
    // Otherwise, must be an integer type
    else {
        // Promote both numeric types
        const Type aPromoted = getPromotedType(a);
        const Type bPromoted = getPromotedType(b);

        // If both promoted operands have the same type, then no further conversion is needed.
        if(aPromoted == bPromoted) {
            return aPromoted;
        }
        // Otherwise, if both promoted operands have signed integer numeric types or both have unsigned integer numeric types, 
        // the operand with the type of lesser integer conversion rank is converted to the type of the operand with greater rank.
        else if(aPromoted.getNumeric().isSigned == bPromoted.getNumeric().isSigned) {
            return (aPromoted.getNumeric().rank > bPromoted.getNumeric().rank) ? aPromoted : bPromoted;
        }
        // Otherwise, if signedness of promoted operands differ
        else {
            const Type signedOp = aPromoted.getNumeric().isSigned ? aPromoted : bPromoted;
            const Type unsignedOp = aPromoted.getNumeric().isSigned ? bPromoted : aPromoted;

            // Otherwise, if the operand that has unsigned integer type has rank greater or equal to the rank of the type of the other operand, 
            // then the operand with signed integer type is converted to the type of the operand with unsigned integer type.
            if(unsignedOp.getNumeric().rank >= signedOp.getNumeric().rank) {
                return unsignedOp;
            }
            // Otherwise, if the type of the operand with signed integer type can represent all of the values of the type of the operand with unsigned integer type, 
            // then the operand with unsigned integer type is converted to the type of the operand with signed integer type.
            else if((signedOp.getNumeric().min <= unsignedOp.getNumeric().min)
                    && (signedOp.getNumeric().max >= unsignedOp.getNumeric().max)) 
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
