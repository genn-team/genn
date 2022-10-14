#pragma once

// GeNN includes
#include "gennExport.h"
#include "models.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(TYPE, NUM_PARAMS, NUM_VARS, NUM_PRE_VARS, NUM_POST_VARS,   \
                                                 NUM_VAR_REFS, NUM_PRE_VAR_REFS, NUM_POST_VAR_REFS)         \
    DECLARE_SNIPPET(TYPE, NUM_PARAMS);                                                                      \
    typedef Models::VarInitContainerBase<NUM_VARS> VarValues;                                               \
    typedef Models::VarInitContainerBase<NUM_PRE_VARS> PreVarValues;                                        \
    typedef Models::VarInitContainerBase<NUM_POST_VARS> PostVarValues;                                      \
    typedef Models::WUVarReferenceContainerBase<NUM_VAR_REFS> VarReferences;                                \
    typedef Models::VarReferenceContainerBase<NUM_PRE_VAR_REFS> PreVarReferences;                           \
    typedef Models::VarReferenceContainerBase<NUM_POST_VAR_REFS> PostVarReferences

#define SET_PRE_VARS(...) virtual VarVec getPreVars() const override{ return __VA_ARGS__; }
#define SET_POST_VARS(...) virtual VarVec getPostVars() const override{ return __VA_ARGS__; }

#define SET_VAR_REFS(...) virtual VarRefVec getVarRefs() const override{ return __VA_ARGS__; }
#define SET_PRE_VAR_REFS(...) virtual VarRefVec getPreVarRefs() const override{ return __VA_ARGS__; }
#define SET_POST_VAR_REFS(...) virtual VarRefVec getPostVarRefs() const override{ return __VA_ARGS__; }

#define SET_ROW_UPDATE_CODE(ROW_UPDATE_CODE) virtual std::string getRowUpdateCode() const override{ return ROW_UPDATE_CODE; }
#define SET_HOST_UPDATE_CODE(HOST_UPDATE_CODE) virtual std::string getHostUpdateCode() const override{ return HOST_UPDATE_CODE; }

//----------------------------------------------------------------------------
// CustomConnectivityUpdateModels::Base
//----------------------------------------------------------------------------
namespace CustomConnectivityUpdateModels
{
//! Base class for all current source models
class GENN_EXPORT Base : public Models::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets names and types (as strings) of state variables that are common
    //! across all synapses coming from the same presynaptic neuron
    virtual VarVec getPreVars() const { return {}; }

    //! Gets names and types (as strings) of state variables that are common
    //! across all synapses going to the same postsynaptic neuron
    virtual VarVec getPostVars() const { return {}; }

    //! Gets names and types (as strings) of synapse variable references
    virtual VarRefVec getVarRefs() const { return {}; }

    //! Gets names and types (as strings) of presynaptic variable references
    virtual VarRefVec getPreVarRefs() const { return {}; }

    //! Gets names and types (as strings) of postsynaptic variable references
    virtual VarRefVec getPostVarRefs() const { return {}; }

    //! Gets the code that performs a row-wise update 
    virtual std::string getRowUpdateCode() const { return ""; }

    //! Gets the code that performs host update 
    virtual std::string getHostUpdateCode() const { return ""; }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Find the index of a named presynaptic variable
    size_t getPreVarIndex(const std::string &varName) const
    {
        return getNamedVecIndex(varName, getPreVars());
    }

    //! Find the index of a named postsynaptic variable
    size_t getPostVarIndex(const std::string &varName) const
    {
        return getNamedVecIndex(varName, getPostVars());
    }

    //! Update hash from model
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Validate names of parameters etc
    void validate() const;
};
}   // CustomConnectivityUpdateModels
