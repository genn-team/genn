#include "code_generator/backendBase.h"

// GeNN includes
#include "gennUtils.h"
#include "logging.h"

// Macro for simplifying defining type sizes
#define TYPE(T) {#T, sizeof(T)}

//--------------------------------------------------------------------------
// CodeGenerator::BackendBase
//--------------------------------------------------------------------------
CodeGenerator::BackendBase::BackendBase(const std::string &scalarType, const PreferencesBase &preferences)
:   m_PointerBytes(sizeof(char*)), m_TypeBytes{{TYPE(char), TYPE(wchar_t), TYPE(signed char), TYPE(short),
    TYPE(signed short), TYPE(short int), TYPE(signed short int), TYPE(int), TYPE(signed int), TYPE(long),
    TYPE(signed long), TYPE(long int), TYPE(signed long int), TYPE(long long), TYPE(signed long long), TYPE(long long int),
    TYPE(signed long long int), TYPE(unsigned char), TYPE(unsigned short), TYPE(unsigned short int), TYPE(unsigned),
    TYPE(unsigned int), TYPE(unsigned long), TYPE(unsigned long int), TYPE(unsigned long long),
    TYPE(unsigned long long int), TYPE(float), TYPE(double), TYPE(long double), TYPE(bool), TYPE(intmax_t),
    TYPE(uintmax_t), TYPE(int8_t), TYPE(uint8_t), TYPE(int16_t), TYPE(uint16_t), TYPE(int32_t), TYPE(uint32_t),
    TYPE(int64_t), TYPE(uint64_t), TYPE(int_least8_t), TYPE(uint_least8_t), TYPE(int_least16_t), TYPE(uint_least16_t),
    TYPE(int_least32_t), TYPE(uint_least32_t), TYPE(int_least64_t), TYPE(uint_least64_t), TYPE(int_fast8_t),
    TYPE(uint_fast8_t), TYPE(int_fast16_t), TYPE(uint_fast16_t), TYPE(int_fast32_t), TYPE(uint_fast32_t),
    TYPE(int_fast64_t), TYPE(uint_fast64_t)}}, m_Preferences(preferences)
{
    // Add scalar type
    addType("scalar", (scalarType == "float") ? sizeof(float) : sizeof(double));
}
//--------------------------------------------------------------------------
size_t CodeGenerator::BackendBase::getSize(const std::string &type) const
{
     // If type is a pointer, any pointer should have the same type
    if(Utils::isTypePointer(type)) {
        return m_PointerBytes;
    }
    // Otherwise
    else {
        // If type isn't found in dictionary, give a warning and return 0
        const auto typeSize = m_TypeBytes.find(type);
        if(typeSize == m_TypeBytes.cend()) {
            LOGW_CODE_GEN << "Unable to estimate size of type '" << type << "'";
            return 0;
        }
        // Otherwise, return its size
        else {
            return typeSize->second;
        }
    }
}
