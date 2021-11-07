#include "code_generator/customUpdateGroupMerged.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

using namespace CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
template<typename C, typename R>
void genCustomUpdate(CodeStream &os, Substitutions &baseSubs, const C &cg, 
                     const ModelSpecMerged &modelMerged, const std::string &index,
                     R getVarRefIndex)
{
    Substitutions updateSubs(&baseSubs);

    const CustomUpdateModels::Base *cm = cg.getArchetype().getCustomUpdateModel();
    const auto varRefs = cm->getVarRefs();

    // Loop through variables
    for(const auto &v : cm->getVars()) {
        if(v.access & VarAccessMode::READ_ONLY) {
            os << "const ";
        }
        os << v.type << " l" << v.name;
        
        // If this isn't a reduction, read value from memory
        // **NOTE** by not initialising these variables for reductions, 
        // compilers SHOULD emit a warning if user code doesn't set it to something
        if(!(v.access & VarAccessModeAttribute::REDUCE)) {
            os << " = group->" << v.name << "[";
            os << cg.getVarIndex(modelMerged.getModel().getBatchSize(),
                                 getVarAccessDuplication(v.access),
                                 updateSubs[index]);
            os << "]";
        }
        os << ";" << std::endl;
    }

    // Loop through variable references
    for(size_t i = 0; i < varRefs.size(); i++) {
        if(varRefs[i].access == VarAccessMode::READ_ONLY) {
            os << "const ";
        }
       
        os << varRefs[i].type << " l" << varRefs[i].name;
        
        // If this isn't a reduction, read value from memory
        // **NOTE** by not initialising these variables for reductions, 
        // compilers SHOULD emit a warning if user code doesn't set it to something
        if(!(varRefs[i].access & VarAccessModeAttribute::REDUCE)) {
            os << " = " << "group->" << varRefs[i].name << "[";
            os << getVarRefIndex(cg.getArchetype().getVarReferences().at(i),
                                 updateSubs[index]);
            os << "]";
        }
        os << ";" << std::endl;
    }
    
    updateSubs.addVarNameSubstitution(cm->getVars(), "", "l");
    updateSubs.addVarNameSubstitution(cm->getVarRefs(), "", "l");
    updateSubs.addParamValueSubstitution(cm->getParamNames(), cg.getArchetype().getParams(),
                                         [&cg](size_t i) { return cg.isParamHeterogeneous(i);  },
                                         "", "group->");
    updateSubs.addVarValueSubstitution(cm->getDerivedParams(), cg.getArchetype().getDerivedParams(),
                                       [&cg](size_t i) { return cg.isDerivedParamHeterogeneous(i);  },
                                       "", "group->");
    updateSubs.addVarNameSubstitution(cm->getExtraGlobalParams(), "", "group->");

    std::string code = cm->getUpdateCode();
    updateSubs.applyCheckUnreplaced(code, "custom update : merged" + std::to_string(cg.getIndex()));
    code = ensureFtype(code, modelMerged.getModel().getPrecision());
    os << code;

    // Write read/write variables back to global memory
    for(const auto &v : cm->getVars()) {
        if(v.access & VarAccessMode::READ_WRITE) {
            os << "group->" << v.name << "[";
            os << cg.getVarIndex(modelMerged.getModel().getBatchSize(),
                                 getVarAccessDuplication(v.access),
                                 updateSubs[index]);
            os << "] = l" << v.name << ";" << std::endl;
        }
    }

    // Write read/write variable references back to global memory
    for(size_t i = 0; i < varRefs.size(); i++) {
        if(varRefs[i].access == VarAccessMode::READ_WRITE) {
            os << "group->" << varRefs[i].name << "[";
            os << getVarRefIndex(cg.getArchetype().getVarReferences().at(i),
                                 updateSubs[index]);
            os << "] = l" << varRefs[i].name << ";" << std::endl;
        }
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateGroupMerged::name = "CustomUpdate";
//----------------------------------------------------------------------------
CustomUpdateGroupMerged::CustomUpdateGroupMerged(size_t index, const std::string &precision, const std::string&, const BackendBase &backend,
                                                 const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups)
:   GroupMerged<CustomUpdateInternal>(index, precision, groups)
{
    addField("unsigned int", "size",
             [](const CustomUpdateInternal &c, size_t) { return std::to_string(c.getSize()); });
    
    // If some variables are delayed, add delay pointer
    if(getArchetype().getDelayNeuronGroup() != nullptr) {
        addField("unsigned int*", "spkQuePtr", 
                 [&backend](const CustomUpdateInternal &cg, size_t) 
                 { 
                     return backend.getScalarAddressPrefix() + "spkQuePtr" + cg.getDelayNeuronGroup()->getName(); 
                 });
    }

    // Add heterogeneous custom update model parameters
    const CustomUpdateModels::Base *cm = getArchetype().getCustomUpdateModel();
    addHeterogeneousParams<CustomUpdateGroupMerged>(
        cm->getParamNames(), "",
        [](const CustomUpdateInternal &cg) { return cg.getParams(); },
        &CustomUpdateGroupMerged::isParamHeterogeneous);

    // Add heterogeneous weight update model derived parameters
    addHeterogeneousDerivedParams<CustomUpdateGroupMerged>(
        cm->getDerivedParams(), "",
        [](const CustomUpdateInternal &cg) { return cg.getDerivedParams(); },
        &CustomUpdateGroupMerged::isDerivedParamHeterogeneous);

    // Add variables to struct
    addVars(cm->getVars(), backend.getDeviceVarPrefix());

    // Add variable references to struct
    addVarReferences(cm->getVarRefs(), backend.getDeviceVarPrefix(),
                    [](const CustomUpdateInternal &cg) { return cg.getVarReferences(); });

    // Add EGPs to struct
    this->addEGPs(cm->getExtraGlobalParams(), backend.getDeviceVarPrefix());
}
//----------------------------------------------------------------------------
bool CustomUpdateGroupMerged::isParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const CustomUpdateInternal &cg) { return cg.getParams(); });
}
//----------------------------------------------------------------------------    
bool CustomUpdateGroupMerged::isDerivedParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const CustomUpdateInternal &cg) { return cg.getDerivedParams(); });
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getHashDigest(), hash);

    // Update hash with each group's custom update size
    updateHash([](const CustomUpdateInternal &cg) { return cg.getSize(); }, hash);

    // Update hash with each group's parameters, derived parameters and variable references
    updateHash([](const CustomUpdateInternal &cg) { return cg.getParams(); }, hash);
    updateHash([](const CustomUpdateInternal &cg) { return cg.getDerivedParams(); }, hash);
    updateHash([](const CustomUpdateInternal &cg) { return cg.getVarReferences(); }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomUpdateGroupMerged::generateCustomUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    genCustomUpdate(os, popSubs, *this, modelMerged, "id",
                    [this, &modelMerged](const Models::VarReference &varRef, const std::string &index)
                    {
                        return getVarRefIndex(varRef.getDelayNeuronGroup() != nullptr,
                                              modelMerged.getModel().getBatchSize(),
                                              getVarAccessDuplication(varRef.getVar().access),
                                              index);
                    });
}
//----------------------------------------------------------------------------
std::string CustomUpdateGroupMerged::getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    // If variable is shared, the batch size is one or this custom update isn't batched, batch offset isn't required
    return ((varDuplication == VarAccessDuplication::SHARED || batchSize == 1 || !getArchetype().isBatched()) ? "" : "batchOffset + ") + index;
}
//----------------------------------------------------------------------------
std::string CustomUpdateGroupMerged::getVarRefIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    // If delayed, variable is shared, the batch size is one or this custom update isn't batched, batch delay offset isn't required
    if(delay) {
        return ((varDuplication == VarAccessDuplication::SHARED || batchSize == 1 || !getArchetype().isBatched()) ? "delayOffset + " : "batchDelayOffset + ") + index;
    }
    else {
        return getVarIndex(batchSize, varDuplication, index);
    }    
}

// ----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateWUGroupMergedBase
//----------------------------------------------------------------------------
bool CustomUpdateWUGroupMergedBase::isParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const CustomUpdateWUInternal &cg) { return cg.getParams(); });
}
//----------------------------------------------------------------------------
bool CustomUpdateWUGroupMergedBase::isDerivedParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const CustomUpdateWUInternal &cg) { return cg.getDerivedParams(); });
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateWUGroupMergedBase::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getHashDigest(), hash);

    // Update hash with sizes of pre and postsynaptic neuron groups
    updateHash([](const CustomUpdateWUInternal &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getSrcNeuronGroup()->getNumNeurons();
               }, hash);

    updateHash([](const CustomUpdateWUInternal &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getTrgNeuronGroup()->getNumNeurons();
               }, hash);

    // Update hash with each group's parameters, derived parameters and variable referneces
    updateHash([](const CustomUpdateWUInternal &cg) { return cg.getParams(); }, hash);
    updateHash([](const CustomUpdateWUInternal &cg) { return cg.getDerivedParams(); }, hash);
    updateHash([](const CustomUpdateWUInternal &cg) { return cg.getVarReferences(); }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
std::string CustomUpdateWUGroupMergedBase::getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    return ((varDuplication == VarAccessDuplication::SHARED || batchSize == 1 || !getArchetype().isBatched()) ? "" : "batchOffset + ") + index;
}
//----------------------------------------------------------------------------
std::string CustomUpdateWUGroupMergedBase::getVarRefIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    return ((varDuplication == VarAccessDuplication::SHARED || batchSize == 1 || !getArchetype().isBatched()) ? "" : "batchOffset + ") + index;
}
//----------------------------------------------------------------------------
CustomUpdateWUGroupMergedBase::CustomUpdateWUGroupMergedBase(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                             const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
:   GroupMerged<CustomUpdateWUInternal>(index, precision, groups)
{
    addField("unsigned int", "rowStride",
             [&backend](const CustomUpdateWUInternal &cg, size_t) 
             { 
                 const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                 return std::to_string(backend.getSynapticMatrixRowStride(*sgInternal)); 
             });
    
    addField("unsigned int", "numSrcNeurons",
             [](const CustomUpdateWUInternal &cg, size_t) 
             {
                 const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                 return std::to_string(sgInternal->getSrcNeuronGroup()->getNumNeurons()); 
             });

    addField("unsigned int", "numTrgNeurons",
             [](const CustomUpdateWUInternal &cg, size_t)
             { 
                 const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                 return std::to_string(sgInternal->getTrgNeuronGroup()->getNumNeurons()); 
             });

    // If synapse group has sparse connectivity
    if(getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        addField(getArchetype().getSynapseGroup()->getSparseIndType() + "*", "ind", 
                 [&backend](const CustomUpdateWUInternal &cg, size_t) 
                 { 
                     return backend.getDeviceVarPrefix() + "ind" + cg.getSynapseGroup()->getName(); 
                 });

        // If the referenced synapse group requires synaptic remapping and matrix type is sparse, add field
        if(backend.isSynRemapRequired(*getArchetype().getSynapseGroup())) {
            addField("unsigned int*", "synRemap", 
                     [&backend](const CustomUpdateWUInternal &cg, size_t) 
                     { 
                         return backend.getDeviceVarPrefix() + "synRemap" + cg.getSynapseGroup()->getName(); 
                     });
        }
        // Otherwise, add row length
        else {
            addField("unsigned int*", "rowLength",
                     [&backend](const CustomUpdateWUInternal &cg, size_t) 
                     { 
                         return backend.getDeviceVarPrefix() + "rowLength" + cg.getSynapseGroup()->getName(); 
                     });
        }
    }

    // Add heterogeneous custom update model parameters
    const CustomUpdateModels::Base *cm = getArchetype().getCustomUpdateModel();
    addHeterogeneousParams<CustomUpdateWUGroupMerged>(
        cm->getParamNames(), "",
        [](const CustomUpdateWUInternal &cg) { return cg.getParams(); },
        &CustomUpdateWUGroupMergedBase::isParamHeterogeneous);

    // Add heterogeneous weight update model derived parameters
    addHeterogeneousDerivedParams<CustomUpdateWUGroupMerged>(
        cm->getDerivedParams(), "",
        [](const CustomUpdateWUInternal &cg) { return cg.getDerivedParams(); },
        &CustomUpdateWUGroupMergedBase::isDerivedParamHeterogeneous);

    // Add variables to struct
    addVars(cm->getVars(), backend.getDeviceVarPrefix());

    // Add variable references to struct
    const auto varRefs = cm->getVarRefs();
    addVarReferences(varRefs, backend.getDeviceVarPrefix(),
                    [](const CustomUpdateWUInternal &cg) { return cg.getVarReferences(); });

     // Loop through variables
    for(size_t v = 0; v < varRefs.size(); v++) {
        // If variable has a transpose 
        if(getArchetype().getVarReferences().at(v).getTransposeSynapseGroup() != nullptr) {
            // Add field with transpose suffix, pointing to transpose var
            addField(varRefs[v].type + "*", varRefs[v].name + "Transpose",
                     [&backend, v](const CustomUpdateWUInternal &g, size_t)
                     {
                         const auto varRef = g.getVarReferences().at(v);
                         return backend.getDeviceVarPrefix() + varRef.getTransposeVar().name + varRef.getTransposeTargetName();
                     });
            }
    }
    // Add EGPs to struct
    this->addEGPs(cm->getExtraGlobalParams(), backend.getDeviceVarPrefix());
}
// ----------------------------------------------------------------------------
// CustomUpdateWUGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateWUGroupMerged::name = "CustomUpdateWU";
//----------------------------------------------------------------------------
void CustomUpdateWUGroupMerged::generateCustomUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    genCustomUpdate(os, popSubs, *this, modelMerged, "id_syn",
                    [this, &modelMerged](const Models::WUVarReference &varRef, const std::string &index) 
                    {  
                        return getVarRefIndex(modelMerged.getModel().getBatchSize(),
                                              getVarAccessDuplication(varRef.getVar().access),
                                              index);
                    });
}

//----------------------------------------------------------------------------
// CustomUpdateTransposeWUGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateTransposeWUGroupMerged::name = "CustomUpdateTransposeWU";
//----------------------------------------------------------------------------
void CustomUpdateTransposeWUGroupMerged::generateCustomUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    genCustomUpdate(os, popSubs, *this, modelMerged, "id_syn",
                    [this, &modelMerged](const Models::WUVarReference &varRef, const std::string &index) 
                    {
                        return getVarRefIndex(modelMerged.getModel().getBatchSize(),
                                              getVarAccessDuplication(varRef.getVar().access),
                                              index);
                    });
}
