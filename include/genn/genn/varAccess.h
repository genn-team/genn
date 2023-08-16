#pragma once

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
namespace GeNN
{
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

//! Flags defining dimensions this variables has
enum class VarAccessDim : unsigned int
{
    NEURON      = (1 << 5),
    PRE_NEURON  = (1 << 6),
    POST_NEURON = (1 << 7),
    DELAY       = (1 << 8),
    BATCH       = (1 << 9),
};

//! Supported combinations of access mode and dimension for neuron variables
enum class NeuronVarAccess : unsigned int
{
    READ_WRITE              = static_cast<unsigned int>(VarAccessMode::READ_WRITE) | static_cast<unsigned int>(VarAccessDim::NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH),
    READ_ONLY               = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH),
    READ_ONLY_DUPLICATE     = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::NEURON),
    READ_ONLY_SHARED_NEURON = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::BATCH),
};

//! Supported combinations of access mode and dimension for synapse variables
/*enum class SynapseVarAccess : unsigned int
{
    // Synaptic variables
    READ_WRITE                  = static_cast<unsigned int>(VarAccessMode::READ_WRITE) | static_cast<unsigned int>(VarAccessDim::PRE_NEURON) | static_cast<unsigned int>(VarAccessDim::POST_NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH),
    READ_ONLY                   = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::PRE_NEURON) | static_cast<unsigned int>(VarAccessDim::POST_NEURON),
    READ_ONLY_DUPLICATE         = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::PRE_NEURON) | static_cast<unsigned int>(VarAccessDim::POST_NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH),

    // Presynaptic variables
    READ_WRITE_PRE              = static_cast<unsigned int>(VarAccessMode::READ_WRITE) | static_cast<unsigned int>(VarAccessDim::PRE_NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH),
    READ_ONLY_PRE               = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::PRE_NEURON),
    READ_ONLY_PRE_DUPLICATE     = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::PRE_NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH),

    // Postsynaptic variables
    READ_WRITE_POST             = static_cast<unsigned int>(VarAccessMode::READ_WRITE) | static_cast<unsigned int>(VarAccessDim::POST_NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH),
    READ_ONLY_POST              = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::POST_NEURON),
    READ_ONLY_POST_DUPLICATE    = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::POST_NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH), 
};

enum class CustomUpdateVarAccess : unsigned int
{
    // Variables with matching shape
    READ_WRITE,
    READ_ONLY,

    // Variables shared across batches
    READ_WRITE_SHARED,
    READ_ONLY_SHARED,


    READ_WRITE_PRE,

    // Reduction variables
    REDUCE_BATCH_SUM,
    REDUCE_BATCH_MAX,
    REDUCE_NEURON_SUM,
    REDUCE_NEURON_MAX,        
    REDUCE_PRE_NEURON_SUM,
    REDUCE_PRE_NEURON_MAX,       
    REDUCE_POST_NEURON_SUM,       
    REDUCE_POST_NEURON_MAX,       
}*/

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
inline bool operator & (unsigned int type, VarAccessMode mode)
{
    return (type & static_cast<unsigned int>(mode)) != 0;
}

inline bool operator & (unsigned int type, VarAccessDuplication duplication)
{
    return (type & static_cast<unsigned int>(duplication)) != 0;
}

inline bool operator & (unsigned int type, VarAccessModeAttribute modeAttribute)
{
    return (type & static_cast<unsigned int>(modeAttribute)) != 0;
}

inline bool operator & (VarAccessMode mode, VarAccessModeAttribute modeAttribute)
{
    return (static_cast<unsigned int>(mode) & static_cast<unsigned int>(modeAttribute)) != 0;
}

inline bool operator & (VarAccessMode a, VarAccessMode b)
{
    return (static_cast<unsigned int>(a) & static_cast<unsigned int>(b)) != 0;
}


//----------------------------------------------------------------------------
// Helpers
//----------------------------------------------------------------------------
inline VarAccessMode getVarAccessMode(unsigned int type)
{
    return static_cast<VarAccessMode>(type & 0x1F);
}

inline VarAccessDuplication getVarAccessDuplication(unsigned int type)
{
    return static_cast<VarAccessDuplication>(type & ~0x1F);
}
}   // namespace GeNN
