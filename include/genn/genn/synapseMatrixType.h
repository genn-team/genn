#pragma once

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
//!< Flags defining differnet types of synaptic matrix connectivity
enum class SynapseMatrixConnectivity : unsigned int
{
    DENSE   = (1 << 0),
    BITMASK = (1 << 1),
    SPARSE  = (1 << 2),
};

//!< Flags defining different types of synaptic matrix connectivity
enum class SynapseMatrixWeight : unsigned int
{
    GLOBAL          = (1 << 5),
    INDIVIDUAL      = (1 << 6),
    INDIVIDUAL_PSM  = (1 << 7),
};

//!< Supported combinations of SynapticMatrixConnectivity and SynapticMatrixWeight
enum class SynapseMatrixType : unsigned int
{
    DENSE_GLOBALG                   = static_cast<unsigned int>(SynapseMatrixConnectivity::DENSE) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL),
    DENSE_GLOBALG_INDIVIDUAL_PSM    = static_cast<unsigned int>(SynapseMatrixConnectivity::DENSE) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL_PSM),
    DENSE_INDIVIDUALG               = static_cast<unsigned int>(SynapseMatrixConnectivity::DENSE) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL_PSM),
    BITMASK_GLOBALG                 = static_cast<unsigned int>(SynapseMatrixConnectivity::BITMASK) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL),
    BITMASK_GLOBALG_INDIVIDUAL_PSM  = static_cast<unsigned int>(SynapseMatrixConnectivity::BITMASK) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL_PSM),
    SPARSE_GLOBALG                  = static_cast<unsigned int>(SynapseMatrixConnectivity::SPARSE) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL),
    SPARSE_GLOBALG_INDIVIDUAL_PSM   = static_cast<unsigned int>(SynapseMatrixConnectivity::SPARSE) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL_PSM),
    SPARSE_INDIVIDUALG              =  static_cast<unsigned int>(SynapseMatrixConnectivity::SPARSE) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL_PSM),
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

inline SynapseMatrixType operator | (SynapseMatrixWeight weightType, SynapseMatrixConnectivity connType)
{
    return static_cast<SynapseMatrixType>(static_cast<unsigned int>(weightType) | static_cast<unsigned int>(connType));
}