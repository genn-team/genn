#pragma once

// Standard includes
#include <map>
#include <string>
#include <vector>

// GeNN includes
#include "currentSourceModels.h"
#include "gennExport.h"
#include "varLocation.h"

// Forward declarations
namespace GeNN
{
class NeuronGroupInternal;
}

//------------------------------------------------------------------------
// GeNN::CurrentSource
//------------------------------------------------------------------------
namespace GeNN
{
class GENN_EXPORT CurrentSource
{
public:
    CurrentSource(const CurrentSource&) = delete;
    CurrentSource() = delete;

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Set location of current source state variable.
    /*! This is ignored for simulations on hardware with a single memory space. */
    void setVarLocation(const std::string &varName, VarLocation loc);

    //! Set location of extra global parameter.
    /*! This is ignored for simulations on hardware with a single memory space. */
    void setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc);

    //! Set whether parameter is dynamic or not i.e. it can be changed at runtime
    void setParamDynamic(const std::string &paramName, bool dynamic = true);

    //! Set name of neuron input variable current source model will inject into.
    /*! This should either be 'Isyn' or the name of one of the target neuron's additional input variables. */
    void setTargetVar(const std::string &varName);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const{ return m_Name; }

    //! Gets the current source model used by this group
    const CurrentSourceModels::Base *getModel() const{ return m_Model; }

    const auto &getParams() const{ return m_Params; }
    const auto &getVarInitialisers() const{ return m_VarInitialisers; }
    const auto &getNeuronVarReferences() const{ return m_NeuronVarReferences;  }

    //! Get variable location for current source model state variable
    VarLocation getVarLocation(const std::string &varName) const{ return m_VarLocation.get(varName); }

    //! Get location of neuron model extra global parameter by name
    VarLocation getExtraGlobalParamLocation(const std::string &paramName) const{ return m_ExtraGlobalParamLocation.get(paramName); }

    //! Is parameter dynamic i.e. it can be changed at runtime
    bool isParamDynamic(const std::string &paramName) const{ return m_DynamicParams.get(paramName); }

    //! Get name of neuron input variable current source model will inject into.
    /*! This will either be 'Isyn' or the name of one of the target neuron's additional input variables. */
    const std::string &getTargetVar() const { return m_TargetVar; }

protected:
    CurrentSource(const std::string &name, const CurrentSourceModels::Base *model,
                  const std::map<std::string, Type::NumericValue> &params, const std::map<std::string, InitVarSnippet::Init> &varInitialisers,
                  const std::map<std::string, Models::VarReference> &neuronVarReferences, const NeuronGroupInternal *trgNeuronGroup, 
                  VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void finalise(double dt);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    const NeuronGroupInternal *getTrgNeuronGroup() const{ return m_TrgNeuronGroup; }

    const auto &getDerivedParams() const{ return m_DerivedParams; }

    bool isZeroCopyEnabled() const;

    //! Is var init code required for any variables in this current source?
    bool isVarInitRequired() const;

    //! Updates hash with current source.
    /*! \note this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getHashDigest(const NeuronGroup *ng) const;

    //! Updates hash with current source initialisation.
    /*! \note this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getInitHashDigest(const NeuronGroup *ng) const;

    boost::uuids::detail::sha1::digest_type getVarLocationHashDigest() const;

    const std::vector<Transpiler::Token> getInjectionCodeTokens() const{ return m_InjectionCodeTokens; }
    
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //! Unique name of current source
    std::string m_Name;

    //! Current source model used for this source
    const CurrentSourceModels::Base *m_Model;

    //! Values of current source parameters
    std::map<std::string, Type::NumericValue> m_Params;
    std::map<std::string, Type::NumericValue> m_DerivedParams;

    //! Initialisers for current source variables
    std::map<std::string, InitVarSnippet::Init> m_VarInitialisers;
    std::map<std::string, Models::VarReference> m_NeuronVarReferences;

    const NeuronGroupInternal *m_TrgNeuronGroup;

    //! Location of individual state variables.
    /*! This is ignored for simulations on hardware with a single memory space. */
    LocationContainer m_VarLocation;

    //! Location of extra global parameters
    LocationContainer m_ExtraGlobalParamLocation;

    //! Data structure tracking whether parameters are dynamic or not
    Snippet::DynamicParameterContainer m_DynamicParams;

    //! Name of neuron input variable current source will inject into.
    /*! This should either be 'Isyn' or the name of one of the target neuron's additional input variables. */
    std::string m_TargetVar;

    //! Tokens produced by scanner from injection code
    std::vector<Transpiler::Token> m_InjectionCodeTokens;
};
}   // namespace GeNN
