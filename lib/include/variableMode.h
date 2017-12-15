#pragma once

// Standard C includes
#include <cstdint>

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
//!< Flags defining which memory space variables should be allocated in
enum class VarLocation : uint8_t
{
    HOST      = (1 << 0),
    DEVICE    = (1 << 1),
    ZERO_COPY = (1 << 2),
};

//!< Flags defining which device should be used to initialise variables
enum class VarInit : uint8_t
{
    HOST    = (1 << 3),
    DEVICE  = (1 << 4),
};

//!< Supported combinations of SynapticMatrixConnectivity and SynapticMatrixWeight
enum class VarMode : uint8_t
{
    LOC_DEVICE_INIT_DEVICE      = static_cast<uint8_t>(VarLocation::DEVICE) | static_cast<uint8_t>(VarInit::DEVICE),
    LOC_HOST_DEVICE_INIT_HOST   = static_cast<uint8_t>(VarLocation::HOST) | static_cast<uint8_t>(VarLocation::DEVICE) | static_cast<uint8_t>(VarInit::HOST),
    LOC_HOST_DEVICE_INIT_DEVICE = static_cast<uint8_t>(VarLocation::HOST) | static_cast<uint8_t>(VarLocation::DEVICE) | static_cast<uint8_t>(VarInit::DEVICE),
    LOC_ZERO_COPY_INIT_HOST     = static_cast<uint8_t>(VarLocation::ZERO_COPY) | static_cast<uint8_t>(VarInit::HOST),
    LOC_ZERO_COPY_INIT_DEVICE   = static_cast<uint8_t>(VarLocation::ZERO_COPY) | static_cast<uint8_t>(VarInit::DEVICE),
};

//----------------------------------------------------------------------------
// Operators
//----------------------------------------------------------------------------
inline bool operator & (VarMode mode, VarInit init)
{
    return (static_cast<uint8_t>(mode) & static_cast<uint8_t>(init)) != 0;
}

inline bool operator & (VarMode mode, VarLocation location)
{
    return (static_cast<uint8_t>(mode) & static_cast<uint8_t>(location)) != 0;
}