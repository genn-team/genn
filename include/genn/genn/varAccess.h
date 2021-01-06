#pragma once


//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
//! Flags defining variable access models
enum class VarAccessMode : unsigned int
{
    READ_WRITE   = (1 << 0),    //! This variable is both read and written by the model
    READ_ONLY    = (1 << 1),    //! This variable is only read by the model
};

//! Flags defining how variables should be duplicated across multiple batches
enum class VarAccessDuplication : unsigned int
{
    DUPLICATE   = (1 << 2),     //! This variable should be duplicated in each batch
    SHARED      = (1 << 3),     //! This variable should be shared between batches
};

//! Supported combinations of SynapticMatrixConnectivity and SynapticMatrixWeight
enum class VarAccess : unsigned int
{
    READ_WRITE          = static_cast<unsigned int>(VarAccessMode::READ_WRITE) | static_cast<unsigned int>(VarAccessDuplication::DUPLICATE),
    READ_ONLY           = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDuplication::SHARED),
    READ_ONLY_DUPLICATE = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDuplication::DUPLICATE),
};

//----------------------------------------------------------------------------
// Operators
//----------------------------------------------------------------------------
inline bool operator & (VarAccess type, VarAccessMode mode)
{
    return (static_cast<unsigned int>(type) & static_cast<unsigned int>(mode)) != 0;
}

inline bool operator & (VarAccess type, VarAccessDuplication duplication)
{
    return (static_cast<unsigned int>(type) & static_cast<unsigned int>(duplication)) != 0;
}

inline bool operator & (VarAccessDuplication a, VarAccessDuplication b)
{
    return (static_cast<unsigned int>(a) & static_cast<unsigned int>(b)) != 0;
}

//----------------------------------------------------------------------------
// Helpers
//----------------------------------------------------------------------------
inline VarAccessMode getVarAccessMode(VarAccess type)
{
    return static_cast<VarAccessMode>(static_cast<unsigned int>(type) & 0x5);
}

inline VarAccessDuplication getVarAccessDuplication(VarAccess type)
{
    return static_cast<VarAccessDuplication>(static_cast<unsigned int>(type) & ~0x5);
}