#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// GeNN includes
#include "currentSourceModels.h"
#include "gennExport.h"
#include "variableMode.h"

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
    //! Set location of current source state variable
    void setVarLocation(const std::string &varName, VarLocation loc);

    //! Set location of extra global parameter
    /*! This is ignored for simulations on hardware with a single memory space
        and only applies to extra global parameters which are pointers. */
    void setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc);

    //! Set name of neuron input variable current source model will inject into
    /*! This should either be 'Isyn' or the name of one of the target neuron's additional input variables. */
    void setTargetVar(const std::string &varName);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const{ return m_Name; }

    //! Gets the current source model used by this group
    const CurrentSourceModels::Base *getCurrentSourceModel() const{ return m_CurrentSourceModel; }

    const std::unordered_map<std::string, Type::NumericValue> &getParams() const{ return m_Params; }
    const std::unordered_map<std::string, InitVarSnippet::Init> &getVarInitialisers() const{ return m_VarInitialisers; }
    const std::unordered_map<std::string, Models::VarReference> &getNeuronVarReferences() const{ return m_NeuronVarReferences;  }

    //! Get variable location for current source model state variable
    VarLocation getVarLocation(const std::string &varName) const;

    //! Get variable location for current source model state variable
    VarLocation getVarLocation(size_t index) const{ return m_VarLocation.at(index); }

    //! Get location of neuron model extra global parameter by name
    /*! This is only used by extra global parameters which are pointers*/
    VarLocation getExtraGlobalParamLocation(const std::string &paramName) const;

    //! Get location of neuron model extra global parameter by omdex
    /*! This is only used by extra global parameters which are pointers*/
    VarLocation getExtraGlobalParamLocation(size_t index) const{ return m_ExtraGlobalParamLocation.at(index); }

    //! Get name of neuron input variable current source model will inject into
    /*! This will either be 'Isyn' or the name of one of the target neuron's additional input variables. */
    const std::string &getTargetVar() const { return m_TargetVar; }

protected:
    CurrentSource(const std::string &name, const CurrentSourceModels::Base *currentSourceModel,
                  const std::unordered_map<std::string, Type::NumericValue> &params, const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers,
                  const std::unordered_map<std::string, Models::VarReference> &neuronVarReferences, const NeuronGroupInternal *trgNeuronGroup, 
                  VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void finalise(double dt);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    const NeuronGroupInternal *getTrgNeuronGroup() const{ return m_TrgNeuronGroup; }

    const std::unordered_map<std::string, Type::NumericValue> &getDerivedParams() const{ return m_DerivedParams; }

    bool isZeroCopyEnabled() const;

    //! Is var init code required for any variables in this current source?
    bool isVarInitRequired() const;

    //! Updates hash with current source
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Updates hash with current source initialisation
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getInitHashDigest() const;

    boost::uuids::detail::sha1::digest_type getVarLocationHashDigest() const;

    const std::vector<Transpiler::Token> getInjectionCodeTokens() const{ return m_InjectionCodeTokens; }
    
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_Name;

    const CurrentSourceModels::Base *m_CurrentSourceModel;
    std::unordered_map<std::string, Type::NumericValue> m_Params;
    std::unordered_map<std::string, Type::NumericValue> m_DerivedParams;
    std::unordered_map<std::string, InitVarSnippet::Init> m_VarInitialisers;
    std::unordered_map<std::string, Models::VarReference> m_NeuronVarReferences;

    const NeuronGroupInternal *m_TrgNeuronGroup;

    //! Location of individual state variables
    std::vector<VarLocation> m_VarLocation;

    //! Location of extra global parameters
    std::vector<VarLocation> m_ExtraGlobalParamLocation;

    //! Name of neuron input variable current source will inject into
    /*! This should either be 'Isyn' or the name of one of the target neuron's additional input variables. */
    std::string m_TargetVar;

    //! Tokens produced by scanner from injection code
    std::vector<Transpiler::Token> m_InjectionCodeTokens;
};
}   // namespace GeNN
