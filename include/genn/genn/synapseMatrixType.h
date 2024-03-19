#pragma once

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
namespace GeNN
{
//! Flags defining how synaptic connectivity is represented
enum class SynapseMatrixConnectivity : unsigned int
{
    //! Connectivity is dense with a synapse between each pair or pre and postsynaptic neurons
    DENSE       = (1 << 0),

    //! Connectivity is sparse and stored using a bitmask.
    BITMASK     = (1 << 1),

    //! Connectivity is sparse and stored using a compressed sparse row data structure
    SPARSE      = (1 << 2),

    //! Connectivity is generated on the fly using a sparse connectivity initialisation snippet
    PROCEDURAL  = (1 << 3),

    //! Connectivity is generated on the fly using a Toeplitz connectivity initialisation snippet
    TOEPLITZ    = (1 << 4),
};

//! Flags defining how synaptic state variables are stored
enum class SynapseMatrixWeight : unsigned int
{
    //! Synaptic state variables are stored individually in memory
    INDIVIDUAL      = (1 << 6),

    //! Synaptic state is generated on the fly using a sparse connectivity initialisation snippet
    PROCEDURAL      = (1 << 7),

    //! Synaptic state variables are stored in a kernel which is shared between synapses in 
    //! a manner defined by either a Toeplitz or sparse connectivity initialisation snippet
    KERNEL          = (1 << 8) 
};

//! Supported combinations of SynapticMatrixConnectivity and SynapticMatrixWeight
enum class SynapseMatrixType : unsigned int
{
    //! Synaptic matrix is dense and synaptic state variables are stored individually in memory
    DENSE               = static_cast<unsigned int>(SynapseMatrixConnectivity::DENSE) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL),

    //! Synaptic matrix is dense and all synaptic state variables must either be constant or generated on the fly using their variable initialisation snippets
    DENSE_PROCEDURALG   = static_cast<unsigned int>(SynapseMatrixConnectivity::DENSE) | static_cast<unsigned int>(SynapseMatrixWeight::PROCEDURAL),

    //! Connectivity is stored as a bitmask.
    /*! For moderately sparse (>3%) connectivity, this uses the least memory. However, connectivity of this sort cannot 
        have any accompanying state variables. Which algorithm is used for propagating spikes through BITMASK connectivity can be hinted via
        SynapseGroup::ParallelismHint. */
    BITMASK             = static_cast<unsigned int>(SynapseMatrixConnectivity::BITMASK) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL),

    //! Connectivity is stored using a compressed sparse row data structure and synaptic state variables are stored individually in memory.
    /*! This is the most efficient choice for very sparse unstructured connectivity or if synaptic state variables are required.*/
    SPARSE              = static_cast<unsigned int>(SynapseMatrixConnectivity::SPARSE) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL),

    //! Sparse synaptic connectivity is generated on the fly using a sparse connectivity initialisation snippet and 
    //! all state variables must be either constant or generated on the fly using variable initialisation snippets.
    /*! Synaptic connectivity of this sort requires very little memory allowing extremely large models to be simulated on a single GPU. */
    PROCEDURAL          = static_cast<unsigned int>(SynapseMatrixConnectivity::PROCEDURAL) | static_cast<unsigned int>(SynapseMatrixWeight::PROCEDURAL),

    //! Sparse synaptic connectivity is generated on the fly using a sparse connectivity initialisation snippet and state variables are stored in a shared kernel
    PROCEDURAL_KERNELG  = static_cast<unsigned int>(SynapseMatrixConnectivity::PROCEDURAL) | static_cast<unsigned int>(SynapseMatrixWeight::KERNEL),

    //! Sparse structured connectivity is generated on the fly a Toeplitz connectivity initialisation snippet and state variables are stored in a shared kernel
    /*! This is the most efficient choice for convolution-like connectivity*/
    TOEPLITZ            = static_cast<unsigned int>(SynapseMatrixConnectivity::TOEPLITZ) | static_cast<unsigned int>(SynapseMatrixWeight::KERNEL),
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

//----------------------------------------------------------------------------
// Helpers
//----------------------------------------------------------------------------
// **THINK** these are kinda nasty as they can return things that aren't actually in the bit enums i.e. ORd together things
inline SynapseMatrixConnectivity getSynapseMatrixConnectivity(SynapseMatrixType type)
{
    return static_cast<SynapseMatrixConnectivity>(static_cast<unsigned int>(type) & 0x1F);
}

inline SynapseMatrixWeight getSynapseMatrixWeight(SynapseMatrixType type)
{
    return static_cast<SynapseMatrixWeight>(static_cast<unsigned int>(type) & ~0x1F);
}
}   // namespace GeNN
