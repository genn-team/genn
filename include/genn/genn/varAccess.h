#pragma once

// Standard C++ includes
#include <variant>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
namespace GeNN
{
//! Flags defining attributes of var access models
//! Read-only and read-write are separate flags rather than read and write so you can test mode & VarAccessMode::READ_ONLY
enum class VarAccessModeAttribute : unsigned int
{
    READ_ONLY   = (1 << 0), //!< This variable can only be read from
    READ_WRITE  = (1 << 1), //!< This variable can be read from or written to
    REDUCE      = (1 << 2), //!< This variable is a reduction target
    SUM         = (1 << 3), //!< This variable's reduction operation is a summation
    MAX         = (1 << 4), //!< This variable's reduction operation is a maximum
    BROADCAST   = (1 << 5), //!< Writes to this variable get broadcast
};

//! Supported combination of VarAccessModeAttribute
enum class VarAccessMode : unsigned int
{
    //! This variable can be read from or written to
    READ_WRITE  = static_cast<unsigned int>(VarAccessModeAttribute::READ_WRITE),

    //! This variable can only be read from
    READ_ONLY   = static_cast<unsigned int>(VarAccessModeAttribute::READ_ONLY),

    //! This variable can only be broadcast i.e. written to
    BROADCAST   = static_cast<unsigned int>(VarAccessModeAttribute::BROADCAST),

    //! This variable is a target for a reduction with a sum operation
    REDUCE_SUM  = static_cast<unsigned int>(VarAccessModeAttribute::REDUCE) | static_cast<unsigned int>(VarAccessModeAttribute::SUM),

    //! This variable is a target for a reduction with a max operation
    REDUCE_MAX  = static_cast<unsigned int>(VarAccessModeAttribute::REDUCE) | static_cast<unsigned int>(VarAccessModeAttribute::MAX),
};

//! Flags defining dimensions this variables has
enum class VarAccessDim : unsigned int
{
    ELEMENT     = (1 << 5), //!< This variable stores separate values for each element i.e. neuron or synapse
    BATCH       = (1 << 6), //!< This variable stores separate values for each batch
};

//! Supported combinations of access mode and dimension for neuron and synapse variables
enum class VarAccess : unsigned int
{
    //! This variable can be read from and written to and stores separate values for each element and each batch
    READ_WRITE              = static_cast<unsigned int>(VarAccessMode::READ_WRITE) | static_cast<unsigned int>(VarAccessDim::ELEMENT) | static_cast<unsigned int>(VarAccessDim::BATCH),

    //! This variable can only be read from and stores separate values for each element but these are shared across batches
    READ_ONLY               = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::ELEMENT),

    //! This variable can only be read from and stores separate values for each element and each batch
    READ_ONLY_DUPLICATE     = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::ELEMENT) | static_cast<unsigned int>(VarAccessDim::BATCH),

    //! This variable can only be read from and stores separate values for each batch but these are shared across neurons
    READ_ONLY_SHARED_NEURON = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::BATCH),
};

//! Supported combinations of access mode and dimension for custom update variables.
/*! The axes are defined 'subtractively', i.e. VarAccessDim::BATCH indicates that this axis should be removed. */
enum class CustomUpdateVarAccess : unsigned int
{
    //! This variable can be read from and written to and has the same dimensions as whatever the custom update is attached to
    READ_WRITE                  = static_cast<unsigned int>(VarAccessMode::READ_WRITE),
    
    //! This variable can only be read from and has the same dimensions as whatever the custom update is attached to
    READ_ONLY                   = static_cast<unsigned int>(VarAccessMode::READ_ONLY),

    //! This variable has the same dimensions as whatever the custom update is attached to and writes to it get broadcast across delay slots
    BROADCAST_DELAY             = static_cast<unsigned int>(VarAccessMode::BROADCAST),

    /*! This variable can only be read from and has the same dimensions as whatever 
      the custom update is attached to aside from being shared across batches */
    READ_ONLY_SHARED            = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::BATCH),

    /*! This variable can only be read from and has the same dimensions as whatever 
      the custom update is attached to aside from being shared across neurons */
    READ_ONLY_SHARED_NEURON    = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::ELEMENT),

    //! This variable is a target for a reduction across batches using a sum operation
    REDUCE_BATCH_SUM            = static_cast<unsigned int>(VarAccessMode::REDUCE_SUM) | static_cast<unsigned int>(VarAccessDim::BATCH),
    
    //! This variable is a target for a reduction across batches using a max operation
    REDUCE_BATCH_MAX            = static_cast<unsigned int>(VarAccessMode::REDUCE_MAX) | static_cast<unsigned int>(VarAccessDim::BATCH),

    //! This variable is a target for a reduction across neurons using a sum operation
    REDUCE_NEURON_SUM          = static_cast<unsigned int>(VarAccessMode::REDUCE_SUM) | static_cast<unsigned int>(VarAccessDim::ELEMENT),

    //! This variable is a target for a reduction across neurons using a max operation
    REDUCE_NEURON_MAX          = static_cast<unsigned int>(VarAccessMode::REDUCE_MAX) | static_cast<unsigned int>(VarAccessDim::ELEMENT),
};

//----------------------------------------------------------------------------
// Operators
//----------------------------------------------------------------------------
inline bool operator & (VarAccessMode mode, VarAccessModeAttribute modeAttribute)
{
    return (static_cast<unsigned int>(mode) & static_cast<unsigned int>(modeAttribute)) != 0;
}

inline bool operator & (VarAccess mode, VarAccessModeAttribute modeAttribute)
{
    return (static_cast<unsigned int>(mode) & static_cast<unsigned int>(modeAttribute)) != 0;
}

inline bool operator & (CustomUpdateVarAccess mode, VarAccessModeAttribute modeAttribute)
{
    return (static_cast<unsigned int>(mode) & static_cast<unsigned int>(modeAttribute)) != 0;
}

inline bool operator & (VarAccessDim a, VarAccessDim b)
{
    return (static_cast<unsigned int>(a) & static_cast<unsigned int>(b)) != 0;
}

inline VarAccessDim operator | (VarAccessDim a, VarAccessDim b)
{
    return static_cast<VarAccessDim>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

//----------------------------------------------------------------------------
// Free functions
//----------------------------------------------------------------------------
inline VarAccessDim clearVarAccessDim(VarAccessDim a, VarAccessDim b)
{
    return static_cast<VarAccessDim>(static_cast<unsigned int>(a) & ~static_cast<unsigned int>(b));
}

//! Extract variable dimensions from its access enumeration
inline VarAccessDim getVarAccessDim(VarAccess v)
{
    return static_cast<VarAccessDim>(static_cast<unsigned int>(v) & ~0x3F);
}

//! Extract custom update variable dimensions from its access enumeration and dimensions of the custom update itself
inline VarAccessDim getVarAccessDim(CustomUpdateVarAccess v, VarAccessDim popDims)
{
    return clearVarAccessDim(popDims, static_cast<VarAccessDim>(static_cast<unsigned int>(v) & ~0x3F));
}

inline VarAccessMode getVarAccessMode(VarAccessMode v)
{
    return v;
}

inline VarAccessMode getVarAccessMode(VarAccess v)
{
    return static_cast<VarAccessMode>(static_cast<unsigned int>(v) & 0x3F);
}

inline VarAccessMode getVarAccessMode(CustomUpdateVarAccess v)
{
    return static_cast<VarAccessMode>(static_cast<unsigned int>(v) & 0x3F);
}
}   // namespace GeNN
