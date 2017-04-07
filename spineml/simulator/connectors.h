#pragma once

// Forward declarations
class SparseProjection;

namespace pugi
{
    class xml_node;
}

//------------------------------------------------------------------------
// SpineMLSimulator::Connectors
//------------------------------------------------------------------------
namespace SpineMLSimulator
{
namespace Connectors
{
    typedef void (*AllocateFn)(unsigned int);

    unsigned int fixedProbabilitySparse(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost,
                                        SparseProjection &sparseProjection, AllocateFn allocateFn);
}   // namespace Connectors
}   // namespace SpineMLSimulator