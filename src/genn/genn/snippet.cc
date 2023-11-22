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
    Utils::updateHash(getParams(), hash);
    Utils::updateHash(getDerivedParams(), hash);
    Utils::updateHash(getExtraGlobalParams(), hash);
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, Type::NumericValue> &paramValues, const std::string &description) const
{
    const auto params = getParams();
    Utils::validateVecNames(params, "Parameters");
    Utils::validateVecNames(getDerivedParams(), "Derived parameter");
    Utils::validateVecNames(getExtraGlobalParams(), "Extra global parameter");

    // If there are a different number of sizes than values, give error
    if(params.size() != paramValues.size()) {
        throw std::runtime_error(description + " expected " + std::to_string(params.size()) + " parameters but got " + std::to_string(paramValues.size()));
    }

    // Loop through names
    for(const auto &n : params) {
        // If there is no values, give error
        if(paramValues.find(n.name) == paramValues.cend()) {
            throw std::runtime_error(description + " missing value for parameter: '" + n.name + "'");
        }
    }
}

//----------------------------------------------------------------------------
// GeNN::Snippet::DynamicParameterContainer
//----------------------------------------------------------------------------
void DynamicParameterContainer::set(const std::string &name, bool value)
{
    // If we're setting, insert name into set
    // **NOTE** we don't care if it's there already
    if(value) {
        m_Dynamic.insert(name);
    }
    // Otherwise, remove
    // **NOTE** again, we don't care if it's there already
    else {
        m_Dynamic.erase(name);
    }
}
//----------------------------------------------------------------------------
bool DynamicParameterContainer::get(const std::string &name) const
{
    return (m_Dynamic.count(name) == 0) ? false : true;
}
//----------------------------------------------------------------------------
void DynamicParameterContainer::updateHash(boost::uuids::detail::sha1 &hash) const
{
    Utils::updateHash(m_Dynamic, hash);
}

//----------------------------------------------------------------------------
// Free functions
//----------------------------------------------------------------------------
void updateHash(const Base::Param &p, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(p.name, hash);
    Type::updateHash(p.type, hash);
}
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
    Type::updateHash(p.value, hash);
}
//----------------------------------------------------------------------------
void updateHash(const Base::DerivedParam &d, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(d.name, hash);
    Type::updateHash(d.type, hash);
}
}   // namespace GeNN::Snippet
