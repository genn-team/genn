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