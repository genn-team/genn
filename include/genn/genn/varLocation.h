#pragma once

// Standard C++ includes
#include <algorithm>
#include <string>
#include <unordered_map>

// Standard C includes
#include <cstdint>

// GeNN includes
#include "gennUtils.h"

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
namespace GeNN
{
//!< Flags defining which memory space variables should be allocated in
enum class VarLocation : uint8_t
{
    HOST      = (1 << 0),
    DEVICE    = (1 << 1),
    ZERO_COPY = (1 << 2),

    HOST_DEVICE = HOST | DEVICE,
    HOST_DEVICE_ZERO_COPY = HOST | DEVICE | ZERO_COPY,
};

//----------------------------------------------------------------------------
// Operators
//----------------------------------------------------------------------------
inline bool operator & (VarLocation locA, VarLocation locB)
{
    return (static_cast<uint8_t>(locA) & static_cast<uint8_t>(locB)) != 0;
}


//----------------------------------------------------------------------------
// LocationContainer
//----------------------------------------------------------------------------
class LocationContainer
{
public:
    LocationContainer(VarLocation defaultLocation) : m_DefaultLocation(defaultLocation)
    {
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    VarLocation get(const std::string &name) const
    {
        const auto l = m_Locations.find(name);
        if(l == m_Locations.cend()) {
            return m_DefaultLocation;
        }
        else {
            return l->second;
        }
    }

    void set(const std::string &name, VarLocation location)
    {
        m_Locations[name] = location;
    }

    bool anyZeroCopy() const
    {
        return std::any_of(m_Locations.begin(), m_Locations.end(),
                           [](const auto &v){ return (v.second & VarLocation::ZERO_COPY); });
    }

    void updateHash(boost::uuids::detail::sha1 &hash) const
    {
        Utils::updateHash(m_DefaultLocation, hash);
        Utils::updateHash(m_Locations, hash);
    }
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    VarLocation m_DefaultLocation;

    std::unordered_map<std::string, VarLocation> m_Locations;
};
}   // namespace GeNN
