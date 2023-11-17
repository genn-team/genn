#include "snippet.h"

// GeNN includes
#include "gennUtils.h"
#include "logging.h"

//----------------------------------------------------------------------------
// GeNN::Snippet::Base::EGP
//----------------------------------------------------------------------------
namespace GeNN::Snippet
{
Base::EGP::EGP(const std::string &n, const std::string &t) 
: name(n), type(Utils::handleLegacyEGPType(t))
{
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
void Base::validate(const std::unordered_map<std::string, Type::NumericValue> &paramValues, const std::string &description) const
{
    const auto paramNames = getParamNames();
    Utils::validateParamNames(paramNames);
    Utils::validateVecNames(getDerivedParams(), "Derived parameter");
    Utils::validateVecNames(getExtraGlobalParams(), "Extra global parameter");

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
    Type::updateHash(e.type, hash);
}
//----------------------------------------------------------------------------
void updateHash(const Base::ParamVal &p, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(p.name, hash);
    Type::updateHash(p.type, hash);
    Utils::updateHash(p.value, hash);
}
//----------------------------------------------------------------------------
void updateHash(const Base::DerivedParam &d, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(d.name, hash);
}
}   // namespace GeNN::Snippet
