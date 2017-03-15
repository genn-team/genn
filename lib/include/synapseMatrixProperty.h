#pragma once

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
//!< Flags defining differnet types of synaptic matrix connectivity
enum class SynapseMatrixConnectivity : unsigned int
{
    SPARSE     = (1 << 0),
    DENSE      = (1 << 1),
    BITMASK    = (1 << 2),
};

//!< Flags defining different types of synaptic matrix connectivity
enum class SynapseMatrixWeight : unsigned int
{
    GLOBAL      = (1 << 3),
    INDIVIDUAL  = (1 << 4),
};

//----------------------------------------------------------------------------
// Operators
//----------------------------------------------------------------------------
//!< Operator to OR together connection and weight type into a unsigned int
inline constexpr unsigned int operator | (SynapseMatrixConnectivity connType, SynapseMatrixWeight weightType)
{
    return static_cast<unsigned int>(connType) | static_cast<unsigned int>(weightType);
}

//!< Operator to OR together weight and connection type into a unsigned int
inline constexpr unsigned int operator | (SynapseMatrixWeight weightType, SynapseMatrixConnectivity connType)
{
    return static_cast<unsigned int>(weightType) | static_cast<unsigned int>(connType);
}