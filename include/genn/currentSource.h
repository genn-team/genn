#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "currentSourceModels.h"
#include "variableMode.h"

//------------------------------------------------------------------------
// CurrentSource
//------------------------------------------------------------------------
class CurrentSource
{
public:

    CurrentSource(const CurrentSource&) = delete;
    CurrentSource() = delete;

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Set location of neuron model state variable
    void setVarLocation(const std::string &varName, VarLocation loc);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const{ return m_Name; }

    //! Gets the current source model used by this group
    const CurrentSourceModels::Base *getCurrentSourceModel() const{ return m_CurrentSourceModel; }

    const std::vector<double> &getParams() const{ return m_Params; }
    const std::vector<Models::VarInit> &getVarInitialisers() const{ return m_VarInitialisers; }

    //! Get variable location for current source model state variable
    VarLocation getVarLocation(const std::string &varName) const;

    //! Get variable location for current source model state variable
    VarLocation getVarLocation(size_t index) const{ return m_VarLocation.at(index); }

protected:
    CurrentSource(const std::string &name, const CurrentSourceModels::Base *currentSourceModel,
                const std::vector<double> &params, const std::vector<Models::VarInit> &varInitialisers,
                VarLocation defaultVarLocation)
    :   m_Name(name), m_CurrentSourceModel(currentSourceModel), m_Params(params), m_VarInitialisers(varInitialisers),
        m_VarLocation(varInitialisers.size(), defaultVarLocation)
    {
    }

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void initInitialiserDerivedParams(double dt);

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_Name;

    const CurrentSourceModels::Base *m_CurrentSourceModel;
    std::vector<double> m_Params;
    std::vector<Models::VarInit> m_VarInitialisers;

    //!< Whether indidividual state variables of a neuron group should use zero-copied memory
    std::vector<VarLocation> m_VarLocation;
};
