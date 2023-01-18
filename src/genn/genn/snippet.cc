#include "snippet.h"

// GeNN includes
#include "logging.h"
#include "type.h"

//----------------------------------------------------------------------------
// GeNN::Snippet::Base::EGP
//----------------------------------------------------------------------------
namespace GeNN::Snippet
{
Base::EGP::EGP(const std::string &n, const std::string &t) 
:   name(n), type(Type::parseNumeric((t.back() == '*') ? t.substr(0, t.length() - 1) : t))
{
    // If type ends in a *, give warning as this is legacy syntax
    if(t.back() == '*') {
        LOGW_GENN << "Extra global parameters are now always arrays so * at end of type is no longer necessary";
    }
}
//----------------------------------------------------------------------------
bool Base::EGP::operator == (const EGP &other) const
{
    return ((name == other.name) && (type->getName() == other.type->getName()));
}

//----------------------------------------------------------------------------
// GeNN::Snippet::Base
//----------------------------------------------------------------------------
void Base::updateHash(boost::uuids::detail::sha1 &hash) const
{
    Utils::updateHash(getParamNames(), hash);
    Utils::updateHash(getDerivedParams(), hash);
    Utils::updateHash(getExtraGlobalParams(), hash);
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, double> &paramValues, const std::string &description) const
{
    const auto paramNames = getParamNames();
    Utils::validateParamNames(paramNames);
    Utils::validateVecNames(getDerivedParams(), "Derived parameter");
    Utils::validateVecNames(getExtraGlobalParams(), "Derived parameter");

    // If there are a different number of sizes than values, give error
    if(paramNames.size() != paramValues.size()) {
        throw std::runtime_error(description + " expected " + std::to_string(paramNames.size()) + " parameters but got " + std::to_string(paramValues.size()));
    }

    // Loop through names
    for(const auto &n : paramNames) {
        // If there is no values, give error
        if(paramValues.find(n) == paramValues.cend()) {
            throw std::runtime_error(description + " missing value for parameter: '" + n + "'");
        }
    }
}

//----------------------------------------------------------------------------
// Free functions
//----------------------------------------------------------------------------
void updateHash(const Base::EGP &e, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(e.name, hash);
    Utils::updateHash(e.type->getName(), hash);
}
//----------------------------------------------------------------------------
void updateHash(const Base::ParamVal &p, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(p.name, hash);
    Utils::updateHash(p.type, hash);
    Utils::updateHash(p.value, hash);
}
//----------------------------------------------------------------------------
void updateHash(const Base::DerivedParam &d, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(d.name, hash);
}
}   // namespace GeNN::Snippet
