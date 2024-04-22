#pragma once

// Standard C++ includes
#include <algorithm>
#include <string>
#include <map>

// Standard C includes
#include <cstdint>

// GeNN includes
#include "gennUtils.h"

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
namespace GeNN
{
//! Flags defining attributes of var locations
enum class VarLocationAttribute : unsigned int
{
    HOST      = (1 << 0),   //!< Variable is located on the host
    DEVICE    = (1 << 1),   //!< Variable is located on the device
    ZERO_COPY = (1 << 2),   //!< Variable is located in zero-copy memory
};

//! Supported combination of VarLocationAttribute
enum class VarLocation : unsigned int
{
    //! Variable is only located on device. This can be used to save host memory.
    DEVICE = static_cast<unsigned int>(VarLocationAttribute::DEVICE),

    //! Variable is located on both host and device. This is the default.
    HOST_DEVICE = static_cast<unsigned int>(VarLocationAttribute::HOST) | static_cast<unsigned int>(VarLocationAttribute::DEVICE),

    //! Variable is shared between host and device using zero copy memory. 
    //! This can improve performance if data is frequently copied between host and device 
    //! but, on non cache-coherent architectures e.g. Jetson, can also reduce access speed.
    HOST_DEVICE_ZERO_COPY = static_cast<unsigned int>(VarLocationAttribute::HOST) | static_cast<unsigned int>(VarLocationAttribute::DEVICE) | static_cast<unsigned int>(VarLocationAttribute::ZERO_COPY),
};

//----------------------------------------------------------------------------
// Operators
//----------------------------------------------------------------------------
inline bool operator & (VarLocation locA, VarLocationAttribute locB)
{
    return (static_cast<unsigned int>(locA) & static_cast<unsigned int>(locB)) != 0;
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
                           [](const auto &v){ return (v.second & VarLocationAttribute::ZERO_COPY); });
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

    std::map<std::string, VarLocation> m_Locations;
};
}   // namespace GeNN
