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

//! Flags defining dimensions this variables has
enum class VarAccessDim : unsigned int
{
    NEURON      = (1 << 5),
    PRE_NEURON  = (1 << 6),
    POST_NEURON = (1 << 7),
    BATCH       = (1 << 8),
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
enum class SynapseVarAccess : unsigned int
{
    // Synaptic variables
    READ_WRITE                  = static_cast<unsigned int>(VarAccessMode::READ_WRITE) | static_cast<unsigned int>(VarAccessDim::PRE_NEURON) | static_cast<unsigned int>(VarAccessDim::POST_NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH),
    READ_ONLY                   = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::PRE_NEURON) | static_cast<unsigned int>(VarAccessDim::POST_NEURON),
    READ_ONLY_DUPLICATE         = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::PRE_NEURON) | static_cast<unsigned int>(VarAccessDim::POST_NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH),

    // Presynaptic variables
    //READ_WRITE_PRE              = static_cast<unsigned int>(VarAccessMode::READ_WRITE) | static_cast<unsigned int>(VarAccessDim::PRE_NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH),
    //READ_ONLY_PRE               = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::PRE_NEURON),
    //READ_ONLY_PRE_DUPLICATE     = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::PRE_NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH),

    // Postsynaptic variables
    //READ_WRITE_POST             = static_cast<unsigned int>(VarAccessMode::READ_WRITE) | static_cast<unsigned int>(VarAccessDim::POST_NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH),
    //READ_ONLY_POST              = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::POST_NEURON),
    //READ_ONLY_POST_DUPLICATE    = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::POST_NEURON) | static_cast<unsigned int>(VarAccessDim::BATCH), 
};

//! Supported combinations of access mode and dimension for custom update variables
/*! The axes are defined 'subtractively' ie VarAccessDim::BATCH indicates that this axis should be removed */
enum class CustomUpdateVarAccess : unsigned int
{
    // Variables with same shape as groups custom update is attached to
    READ_WRITE                  = static_cast<unsigned int>(VarAccessMode::READ_WRITE),
    READ_ONLY                   = static_cast<unsigned int>(VarAccessMode::READ_ONLY),

    // Variables which will be shared across batches if custom update is batched
    READ_WRITE_SHARED           = static_cast<unsigned int>(VarAccessMode::READ_WRITE) | static_cast<unsigned int>(VarAccessDim::BATCH),
    READ_ONLY_SHARED            = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::BATCH),

    // Variables which will be shared across neurons if per-neuron
    READ_WRITE_SHARED_NEURON    = static_cast<unsigned int>(VarAccessMode::READ_WRITE) | static_cast<unsigned int>(VarAccessDim::NEURON),
    READ_ONLY_SHARED_NEURON     = static_cast<unsigned int>(VarAccessMode::READ_ONLY) | static_cast<unsigned int>(VarAccessDim::NEURON),

    // Reduction variables
    REDUCE_BATCH_SUM            = static_cast<unsigned int>(VarAccessMode::REDUCE_SUM) | static_cast<unsigned int>(VarAccessDim::BATCH),
    REDUCE_BATCH_MAX            = static_cast<unsigned int>(VarAccessMode::REDUCE_MAX) | static_cast<unsigned int>(VarAccessDim::BATCH),
    REDUCE_NEURON_SUM           = static_cast<unsigned int>(VarAccessMode::REDUCE_SUM) | static_cast<unsigned int>(VarAccessDim::NEURON),
    REDUCE_NEURON_MAX           = static_cast<unsigned int>(VarAccessMode::REDUCE_MAX) | static_cast<unsigned int>(VarAccessDim::NEURON),
};

//----------------------------------------------------------------------------
// Operators
//----------------------------------------------------------------------------
inline bool operator & (VarAccessMode mode, VarAccessModeAttribute modeAttribute)
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


inline VarAccessDim clearDim(VarAccessDim a, VarAccessDim b)
{
    return static_cast<VarAccessDim>(static_cast<unsigned int>(a) & ~static_cast<unsigned int>(b));
}

//----------------------------------------------------------------------------
// VarAccess
//----------------------------------------------------------------------------
//! Wrapper class encapsulating 
GENN_EXPORT class VarAccess
{
public:
    VarAccess()
    {}
    VarAccess(NeuronVarAccess n) : m_Access{n}
    {}
    VarAccess(SynapseVarAccess s) : m_Access{s}
    {}
    VarAccess(CustomUpdateVarAccess c) : m_Access{c}
    {}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    template<typename V>
    VarAccessDim getDims() const
    {
        // Extract value
        const unsigned int val = std::visit(
            Utils::Overload{
                // If access is set to default, use READ_WRITE mode of typed var access e.g. NeuronVarAcccess::READ_WRITE
                [](std::monostate) { return static_cast<unsigned int>(V::READ_WRITE); },
                // Otherwise, if stored type matches template type, use value
                [](V v) { return static_cast<unsigned int>(v); },
                // Otherwise, give error
                [](auto)->unsigned int { throw std::runtime_error("Invalid var access type"); }},
            m_Access);

        // Mask out dimension bits and cast to enum
        return static_cast<VarAccessDim>(val & ~0x1F);
    }

    template<typename V>
    bool isValid() const
    {
        return std::visit(
            Utils::Overload{
                [](std::monostate) { return true; },
                [](V) { return true; },
                [](auto) { return false; }},
            m_Access);
    }

    void updateHash(boost::uuids::detail::sha1 &hash) const
    {
        Utils::updateHash(m_Access, hash);
    }

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    operator VarAccessMode() const
    {
        // If access is set to default, access mode is always read-write otherwise mask out and cast access mode, bits
        return std::visit(
            Utils::Overload{
                [](std::monostate) { return VarAccessMode::READ_WRITE; },
                [](auto v) { return static_cast<VarAccessMode>(static_cast<unsigned int>(v) & 0x1F); }},
            m_Access);
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::variant<std::monostate, NeuronVarAccess, SynapseVarAccess, CustomUpdateVarAccess> m_Access;
};

}   // namespace GeNN
