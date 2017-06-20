#pragma once

// Standard C++ includes
#include <fstream>
#include <memory>
#include <vector>

// SpineML simulator includes
#include "logOutput.h"

// Forward declarations
namespace pugi
{
    class xml_node;
}

namespace SpineMLSimulator
{
    class ModelProperty;
}

//----------------------------------------------------------------------------
// SpineMLSimulator::LogOutputAnalogue
//----------------------------------------------------------------------------
namespace SpineMLSimulator
{
class LogOutputAnalogue : public LogOutput
{
public:
    LogOutputAnalogue(const pugi::xml_node &node, double dt,
                      const filesystem::path &basePath,
                      const ModelProperty *modelProperty);

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    // Record any data required during this timestep
    virtual void record(double dt, unsigned int timestep) override;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ofstream m_File;

    const ModelProperty *m_ModelProperty;

    std::vector<unsigned int> m_Indices;
};
}   // namespace SpineMLSimulator