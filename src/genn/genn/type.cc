#include "type.h"

// Standard C++ includes
#include <map>
#include <unordered_map>
#include <variant>

// GeNN includes
#include "gennUtils.h"
#include "logging.h"

// Transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/parser.h"
#include "transpiler/scanner.h"

using namespace GeNN;

// Anonymous namespace
namespace
{
// Mapping of signed integer numericTypeSpecifiers to their unsigned equivalents
const std::map<Type::ResolvedType, Type::ResolvedType> unsignedType{
    {Type::Int8, Type::Uint8},
    {Type::Int16, Type::Uint16},
    {Type::Int32, Type::Uint32}};
}   // Anonymous namespace

//----------------------------------------------------------------------------
// GeNN::Type::NumericValue
//----------------------------------------------------------------------------
namespace GeNN::Type
{
bool NumericValue::operator == (const NumericValue &other) const
{
    return std::visit(
        Utils::Overload{
            [](auto a, auto b)
            {
                return a == b;
            }},
        m_Value, other.m_Value);
}
//----------------------------------------------------------------------------
bool NumericValue::operator != (const NumericValue &other) const
{
    return std::visit(
        Utils::Overload{
            [](auto a, auto b)
            {
                return a != b;
            }},
        m_Value, other.m_Value);
}
//----------------------------------------------------------------------------
bool NumericValue::operator < (const NumericValue &other) const
{
    // **YUCK** C++20 std::cmp_greater would do this in one line
    return std::visit(
        Utils::Overload{
            [](int64_t a, uint64_t b)
            {
                if(a < 0) {
                    return true;
                }
                else {
                    return (static_cast<uint64_t>(a) < b);
                }
            },
            [](uint64_t a, int64_t b)
            {
                if(b < 0) {
                    return false;
                }
                else {
                    return (a < static_cast<uint64_t>(b));
                }
            },
            [](auto a, auto b)
            {
                return a < b;
            }},
        m_Value, other.m_Value);
}
//----------------------------------------------------------------------------
bool NumericValue::operator > (const NumericValue &other) const
{
    // **YUCK** C++20 std::cmp_less would do this in one line
    return std::visit(
        Utils::Overload{
            [](int64_t a, uint64_t b)
            {
                if(a < 0) {
                    return false;
                }
                else {
                    return (static_cast<uint64_t>(a) > b);
                }
            },
            [](uint64_t a, int64_t b)
            {
                if(b < 0) {
                    return true;
                }
                else {
                    return (a > static_cast<uint64_t>(b));
                }
            },
            [](auto a, auto b)
            {
                return a > b;
            }},
        m_Value, other.m_Value);
}
//----------------------------------------------------------------------------
bool NumericValue::operator <= (const NumericValue &other) const
{
    return !this->operator > (other);
}
//----------------------------------------------------------------------------
bool NumericValue::operator >= (const NumericValue &other) const
{
    return !this->operator < (other);
}

//----------------------------------------------------------------------------
// GeNN::Type::ResolvedType
//----------------------------------------------------------------------------
std::string ResolvedType::getName() const
{
    const std::string qualifier = hasQualifier(Type::Qualifier::CONSTANT) ? "const " : "";
    return std::visit(
        Utils::Overload{
            [&qualifier](const Type::ResolvedType::Value &value)
            {
                return qualifier + value.name;
            },
            [&qualifier](const Type::ResolvedType::Pointer &pointer)
            {
                return qualifier + pointer.valueType->getName() + "*";
            },
            [&qualifier](const Type::ResolvedType::Function &function)
            {
                std::string description = qualifier + function.returnType->getName() + "(";
                for (const auto &a : function.argTypes) {
                    description += (a.getName() + ",");
                }
                return description + ")";
            },
            [&qualifier](std::monostate)
            {
                return qualifier + "void";
            }},
        detail);
}
//----------------------------------------------------------------------------
size_t ResolvedType::getSize(size_t pointerBytes) const
{
    return std::visit(
        Utils::Overload{
            [](const Type::ResolvedType::Value &value)
            {
                return value.size;
            },
            [pointerBytes](const Type::ResolvedType::Pointer&)
            {
                return pointerBytes;
            },
            [](const Type::ResolvedType::Function&)->size_t
            {
                throw std::runtime_error("Function types do not have size");
            },
            [](std::monostate)->size_t
            {
                throw std::runtime_error("Void type does not have size");
            }},
            detail);
}
//----------------------------------------------------------------------------
// GeNN::Type::UnresolvedType
//----------------------------------------------------------------------------
ResolvedType UnresolvedType::resolve(const TypeContext &typeContext) const
{
    return std::visit(
         Utils::Overload{
             [](const Type::ResolvedType &resolved)
             {
                 return resolved;
             },
             [&typeContext](const std::string &name)
             {
                return Utils::parseNumericType(name, typeContext);
             }},
        detail);
}
//----------------------------------------------------------------------------
ResolvedType getPromotedType(const ResolvedType &type)
{
    // If a small integer type is used in an expression, it is implicitly converted to int which is always signed. 
    // This is known as the integer promotions or the integer promotion rule 
    // **NOTE** this is true because in our type system unsigned short is uint16 which can be represented in int32
    assert(type.isNumeric());
    if(type.getNumeric().rank < Int32.getNumeric().rank) {
        return Int32;
    }
    else {
        return type;
    }
}
//----------------------------------------------------------------------------
ResolvedType getCommonType(const ResolvedType &a, const ResolvedType &b)
{
    // If either type is double, common type is double
    assert(a.isNumeric());
    assert(b.isNumeric());
    const auto unqualifiedA = a.removeQualifiers();
    const auto unqualifiedB = b.removeQualifiers();
    if(unqualifiedA == Double || unqualifiedB == Double) {
        return Double;
    }
    // Otherwise, if either type is float, common type is float
    if(unqualifiedA == Float || unqualifiedB == Float) {
        return Float;
    }
    // Otherwise, must be an integer type
    else {
        // Promote both numeric types
        const ResolvedType aPromoted = getPromotedType(unqualifiedA);
        const ResolvedType bPromoted = getPromotedType(unqualifiedB);

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
            const ResolvedType signedOp = aPromoted.getNumeric().isSigned ? aPromoted : bPromoted;
            const ResolvedType unsignedOp = aPromoted.getNumeric().isSigned ? bPromoted : aPromoted;

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
//----------------------------------------------------------------------------
std::string writeNumeric(const NumericValue &value, const ResolvedType &type)
{
    const auto &numeric = type.getNumeric();
    std::visit(
        Utils::Overload{
            [&numeric](double value) 
            {
                // Set scientific formatting
                std::ostringstream os;
                os << std::scientific;

                // Set precision
                os << std::setprecision(numeric.maxDigits10);

                // Write value to stream
                os << value;

                // Return string stream contents with literal suffix
                return os.str() + numeric.literalSuffix;
            },
            // Otherwise, if it's an integer value, convert to string followed by literal suffix
            [&numeric](auto value) 
            {
                return std::to_string(value) + numeric.literalSuffix;
            }},
        value.get());
}
//----------------------------------------------------------------------------
void updateHash(const NumericValue &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.get(), hash);
}
//----------------------------------------------------------------------------
void updateHash(const ResolvedType::Numeric &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.rank, hash);
    updateHash(v.min, hash);
    updateHash(v.max, hash);
    updateHash(v.lowest, hash);
    Utils::updateHash(v.maxDigits10, hash);
    Utils::updateHash(v.isSigned, hash);
    Utils::updateHash(v.isIntegral, hash);
    Utils::updateHash(v.literalSuffix, hash);
}
//----------------------------------------------------------------------------
void updateHash(const ResolvedType::Value &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.size, hash);
    Utils::updateHash(v.numeric, hash);
}
//----------------------------------------------------------------------------
void updateHash(const ResolvedType::Pointer &v, boost::uuids::detail::sha1 &hash)
{
    updateHash(*v.valueType, hash);
}
//----------------------------------------------------------------------------
void updateHash(const ResolvedType::Function &v, boost::uuids::detail::sha1 &hash)
{
    updateHash(*v.returnType, hash);
    Utils::updateHash(v.argTypes, hash);
}
//----------------------------------------------------------------------------
void updateHash(const ResolvedType &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.qualifiers, hash);
    Utils::updateHash(v.detail, hash);
}
//----------------------------------------------------------------------------
void updateHash(const UnresolvedType &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.detail, hash);
}
}   // namespace GeNN::Type
