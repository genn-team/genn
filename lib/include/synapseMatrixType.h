#pragma once

// GeNN includes
#include "synapseMatrixProperty.h"

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
//!< Supported combinations of SynapticMatrixConnectivity and SynapticMatrixWeight
enum class SynapseMatrixType : unsigned int
{
    SPARSE_GLOBALG       = SynapseMatrixConnectivity::SPARSE | SynapseMatrixWeight::GLOBAL,
    SPARSE_INDIVIDUALG   = SynapseMatrixConnectivity::SPARSE | SynapseMatrixWeight::INDIVIDUAL,
    DENSE_GLOBALG        = SynapseMatrixConnectivity::DENSE | SynapseMatrixWeight::GLOBAL,
    DENSE_INDIVIDUALG    = SynapseMatrixConnectivity::DENSE | SynapseMatrixWeight::INDIVIDUAL,
    BITMASK_GLOBALG      = SynapseMatrixConnectivity::BITMASK | SynapseMatrixWeight::GLOBAL,
};

//----------------------------------------------------------------------------
// Operators
//----------------------------------------------------------------------------
inline constexpr bool operator & (SynapseMatrixType type, SynapseMatrixConnectivity connType)
{
    return (static_cast<unsigned int>(type) & static_cast<unsigned int>(connType)) != 0;
}

inline constexpr bool operator & (SynapseMatrixType type, SynapseMatrixWeight weightType)
{
    return (static_cast<unsigned int>(type) & static_cast<unsigned int>(weightType)) != 0;
}