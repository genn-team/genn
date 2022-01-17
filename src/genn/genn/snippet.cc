#include "snippet.h"

//----------------------------------------------------------------------------
// Snippet::Base
//----------------------------------------------------------------------------
void Snippet::Base::updateHash(boost::uuids::detail::sha1 &hash) const
{
    Utils::updateHash(getParamNames(), hash);
    Utils::updateHash(getDerivedParams(), hash);
    Utils::updateHash(getExtraGlobalParams(), hash);
}
//----------------------------------------------------------------------------
void Snippet::Base::validate(const std::unordered_map<std::string, double> &paramValues, const std::string &description) const
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