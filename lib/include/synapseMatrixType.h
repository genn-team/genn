#pragma once

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
//!< Flags defining differnet types of synaptic matrix connectivity
enum class SynapseMatrixConnectivity : unsigned int
{
    SPARSE  = (1 << 0),
    DENSE   = (1 << 1),
    BITMASK = (1 << 2),
    RAGGED  = (1 << 3),
    YALE    = (1 << 4),
};

//!< Flags defining different types of synaptic matrix connectivity
enum class SynapseMatrixWeight : unsigned int
{
    GLOBAL      = (1 << 5),
    INDIVIDUAL  = (1 << 6),
};

//!< Supported combinations of SynapticMatrixConnectivity and SynapticMatrixWeight
enum class SynapseMatrixType : unsigned int
{
    SPARSE_GLOBALG      = static_cast<unsigned int>(SynapseMatrixConnectivity::SPARSE) | static_cast<unsigned int>(SynapseMatrixConnectivity::YALE) |static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL),
    SPARSE_INDIVIDUALG  = static_cast<unsigned int>(SynapseMatrixConnectivity::SPARSE) | static_cast<unsigned int>(SynapseMatrixConnectivity::YALE)| static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL),
    DENSE_GLOBALG       = static_cast<unsigned int>(SynapseMatrixConnectivity::DENSE) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL),
    DENSE_INDIVIDUALG   = static_cast<unsigned int>(SynapseMatrixConnectivity::DENSE) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL),
    BITMASK_GLOBALG     = static_cast<unsigned int>(SynapseMatrixConnectivity::BITMASK) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL),
    RAGGED_GLOBALG      = static_cast<unsigned int>(SynapseMatrixConnectivity::SPARSE) | static_cast<unsigned int>(SynapseMatrixConnectivity::RAGGED) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL),
    RAGGED_INDIVIDUALG  = static_cast<unsigned int>(SynapseMatrixConnectivity::SPARSE) | static_cast<unsigned int>(SynapseMatrixConnectivity::RAGGED) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL),
};

//----------------------------------------------------------------------------
// Operators
//----------------------------------------------------------------------------
inline bool operator & (SynapseMatrixType type, SynapseMatrixConnectivity connType)
{
    return (static_cast<unsigned int>(type) & static_cast<unsigned int>(connType)) != 0;
}

inline bool operator & (SynapseMatrixType type, SynapseMatrixWeight weightType)
{
    return (static_cast<unsigned int>(type) & static_cast<unsigned int>(weightType)) != 0;
}