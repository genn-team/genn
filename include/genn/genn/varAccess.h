#pragma once


//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
//! Flags defining attributes of var access models
//! **NOTE** Read-only and read-write are seperate flags rather than read and write so you can test mode & VarAccessMode::READ_ONLY
enum class VarAccessModeAttribute : unsigned int
{
    READ_ONLY   = (1 << 0), //! This variable is read only
    READ_WRITE  = (1 << 1), //! This variable is read-write
    REDUCE      = (1 << 2), //! This variable is a reduction target
    SUM         = (1 << 3), //! This variable's reduction operation is a summation
    MAX         = (1 << 4), //! This variable's reduction operation is a maximum
};

//! Supported combination of VarAccessModeAttribute
enum class VarAccessMode : unsigned int
{
    READ_WRITE  = static_cast<unsigned int>(VarAccessModeAttribute::READ_WRITE),
    READ_ONLY   = static_cast<unsigned int>(VarAccessModeAttribute::READ_ONLY),
    REDUCE_SUM  = static_cast<unsigned int>(VarAccessModeAttribute::REDUCE) | static_cast<unsigned int>(VarAccessModeAttribute::SUM),
    REDUCE_MAX  = static_cast<unsigned int>(VarAccessModeAttribute::REDUCE) | static_cast<unsigned int>(VarAccessModeAttribute::MAX),
};

//! Flags defining how variables should be duplicated across multiple batches
enum class VarAccessDuplication : unsigned int
{
    DUPLICATE       = (1 << 5), //! This variable should be duplicated in each batch
    SHARED          = (1 << 6), //! This variable should be shared between batches
    SHARED_NEURON   = (1 << 7)  //! This variable should be shared between neurons
};

//! Supported combinations of VarAccessMode and VarAccessDuplication
enum class VarAccess : unsigned int
{
    READ_WRITE              = static_cast<unsigned int>(VarAccessMode::READ_WRITE) | static_cast<unsigned int>(VarAccessDuplication::DUPLICATE),
    READ_ONLY               = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDuplication::SHARED),
    READ_ONLY_SHARED_NEURON = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDuplication::SHARED_NEURON),
    READ_ONLY_DUPLICATE     = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDuplication::DUPLICATE),
    REDUCE_BATCH_SUM        = static_cast<unsigned int>(VarAccessMode::REDUCE_SUM) | static_cast<unsigned int>(VarAccessDuplication::SHARED),
    REDUCE_BATCH_MAX        = static_cast<unsigned int>(VarAccessMode::REDUCE_MAX) | static_cast<unsigned int>(VarAccessDuplication::SHARED),
    REDUCE_NEURON_SUM       = static_cast<unsigned int>(VarAccessMode::REDUCE_SUM) | static_cast<unsigned int>(VarAccessDuplication::SHARED_NEURON),
    REDUCE_NEURON_MAX       = static_cast<unsigned int>(VarAccessMode::REDUCE_MAX) | static_cast<unsigned int>(VarAccessDuplication::SHARED_NEURON),
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

inline bool operator & (VarAccess type, VarAccessModeAttribute modeAttribute)
{
    return (static_cast<unsigned int>(type) & static_cast<unsigned int>(modeAttribute)) != 0;
}

inline bool operator & (VarAccessMode mode, VarAccessModeAttribute modeAttribute)
{
    return (static_cast<unsigned int>(mode) & static_cast<unsigned int>(modeAttribute)) != 0;
}

//----------------------------------------------------------------------------
// Helpers
//----------------------------------------------------------------------------
inline VarAccessMode getVarAccessMode(VarAccess type)
{
    return static_cast<VarAccessMode>(static_cast<unsigned int>(type) & 0x1F);
}

inline VarAccessDuplication getVarAccessDuplication(VarAccess type)
{
    return static_cast<VarAccessDuplication>(static_cast<unsigned int>(type) & ~0x1F);
}
