#pragma once

// GeNN includes
#include "synapseMatrixProperty.h"

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
//!< Supported combinations of SynapticMatrixConnectivity and SynapticMatrixWeight
enum class SynapseMatrixType : unsigned int
{
    SPARSE_GLOBAL_WEIGHT       = SynapseMatrixConnectivity::SPARSE | SynapseMatrixWeight::GLOBAL,
    SPARSE_INDIVIDUAL_WEIGHT   = SynapseMatrixConnectivity::SPARSE | SynapseMatrixWeight::INDIVIDUAL,
    DENSE_INDIVIDUAL_WEIGHT    = SynapseMatrixConnectivity::DENSE | SynapseMatrixWeight::INDIVIDUAL,
    BITMASK_GLOBAL_WEIGHT      = SynapseMatrixConnectivity::BITMASK | SynapseMatrixWeight::GLOBAL,
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