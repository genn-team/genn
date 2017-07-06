#pragma once

// Forward declarations
class SparseProjection;

namespace pugi
{
    class xml_node;
}

namespace filesystem
{
    class path;
}

//------------------------------------------------------------------------
// SpineMLSimulator::Connectors
//------------------------------------------------------------------------
namespace SpineMLSimulator
{
namespace Connectors
{
    typedef void (*AllocateFn)(unsigned int);

    unsigned int create(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost,
                        SparseProjection *sparseProjection, AllocateFn allocateFn, const filesystem::path &basePath);

}   // namespace Connectors
}   // namespace SpineMLSimulator