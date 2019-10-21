#pragma once

// Standard C++ includes
#include <tuple>

// GeNN includes
#include "initSparseConnectivitySnippet.h"

// Forward declarations
namespace pugi
{
    class xml_node;
}

namespace filesystem
{
    class path;
}

enum class SynapseMatrixConnectivity : unsigned int;

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::FixedProbability
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
namespace Connectors
{
namespace FixedProbability
{
    SynapseMatrixConnectivity getMatrixConnectivity(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost);
    InitSparseConnectivitySnippet::Init getConnectivityInit(const pugi::xml_node &node);
}   // namespace FixedProbability

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::OneToOne
//----------------------------------------------------------------------------
namespace OneToOne
{
    SynapseMatrixConnectivity getMatrixConnectivity(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost);
    InitSparseConnectivitySnippet::Init getConnectivityInit(const pugi::xml_node &node);
}   // namespace OneToOne


//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::AllToAll
//----------------------------------------------------------------------------
namespace AllToAll
{
    SynapseMatrixConnectivity getMatrixConnectivity(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost);
}   // namespace AllToAll

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::List
//----------------------------------------------------------------------------
namespace List
{
    enum class DelayType
    {
        None,
        Homogeneous,
        Heterogeneous,
    };

    SynapseMatrixConnectivity getMatrixConnectivity(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost);
    std::tuple<unsigned int, DelayType, float> readMaxRowLengthAndDelay(const filesystem::path &basePath, const pugi::xml_node &node,
                                                                        unsigned int numPre, unsigned int numPost);
}   // namespace List
}   // namespace Connectors
}   // namespace SpineMLGenerator
