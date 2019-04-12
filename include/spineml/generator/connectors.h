#pragma once

// Standard C++ includes
#include <utility>

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

enum class SynapseMatrixType : unsigned int;

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::FixedProbability
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
namespace Connectors
{
namespace FixedProbability
{
    SynapseMatrixType getMatrixType(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost, bool globalG);
    InitSparseConnectivitySnippet::Init getConnectivityInit(const pugi::xml_node &node);
}   // namespace FixedProbability

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::OneToOne
//----------------------------------------------------------------------------
namespace OneToOne
{
    SynapseMatrixType getMatrixType(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost, bool globalG);
    InitSparseConnectivitySnippet::Init getConnectivityInit(const pugi::xml_node &node);
}   // namespace OneToOne


//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::AllToAll
//----------------------------------------------------------------------------
namespace AllToAll
{
    SynapseMatrixType getMatrixType(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost, bool globalG);
}   // namespace AllToAll

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::List
//----------------------------------------------------------------------------
namespace List
{
    SynapseMatrixType getMatrixType(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost, bool globalG);
    std::pair<unsigned int, float> readMaxRowLengthAndDelay(const filesystem::path &basePath, const pugi::xml_node &node,
                                                            unsigned int numPre, unsigned int numPost);
}   // namespace List
}   // namespace Connectors
}   // namespace SpineMLGenerator
