#include "code_generator/customConnectivityUpdateGroupMerged.h"

// Standard C++ includes
#include <list>

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
template<typename A, typename G>
void addPrivateVarPointerFields(EnvironmentGroupMergedField<G> &env, const std::string &arrayPrefix, const G &group)
{
    // Loop through variable references and add private pointer field 
    const A archetypeAdaptor(group.getArchetype());
    for(const auto &v : archetypeAdaptor.getDefs()) {
        const auto resolvedType = v.type.resolve(getGroup().getTypeContext());
        env.addField(resolvedType.createPointer(), "_" + v.name, v.name + fieldSuffix,
                     [arrayPrefix, v](const auto &g, size_t) 
                     { 
                         return arrayPrefix + v.name + A(g).getNameSuffix();
                     });
    }
}
//----------------------------------------------------------------------------
template<typename A, typename G>
void addPrivateVarRefPointerFields(EnvironmentGroupMergedField<G> &env, const std::string &arrayPrefix, const G &group)
{
    // Loop through variable references and add private pointer field 
    const A archetypeAdaptor(group.getArchetype());
    for(const auto &v : archetypeAdaptor.getDefs()) {
        const auto resolvedType = v.type.resolve(getGroup().getTypeContext());
        env.addField(resolvedType.createPointer(), "_" + v.name, v.name + fieldSuffix,
                     [arrayPrefix, v](const auto &g, size_t) 
                     { 
                         const auto varRef = A(g).getInitialisers().at(v.name);
                         return arrayPrefix + varRef.getVar().name + varRef.getTargetName(); 
                     });
    }
}
//----------------------------------------------------------------------------
template<typename A, typename G, typename I>
void addPrivateVarRefAccess(EnvironmentGroupMergedField<G> &env, const G &group, unsigned int batchSize, I getIndexFn)
{
    // Loop through variable references
    const A archetypeAdaptor(group.getArchetype());
    for(const auto &v : archetypeAdaptor.getDefs()) {
        // If model isn't batched or variable isn't duplicated
        const auto &varRef = archetypeAdaptor.getInitialisers().at(v.name);
        if(batchSize == 1 || !varRef.isDuplicated()) {
            // Add field with qualified type which indexes private pointer field
            const auto resolvedType = v.type.resolve(getGroup().getTypeContext());
            const auto qualifiedType = (v.access & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;
            env.add(qualifiedType, v.name, "$(_" + v.name + ")[" + getIndexFn(v.access, varRef));
        }
    }
}
//----------------------------------------------------------------------------
template<typename A, typename G>
void addPrivateVarRefAccess(EnvironmentGroupMergedField<G> &env, const G &group, unsigned int batchSize, const std::string &indexSuffix)
{
    addPrivateVarRefPointerFields(env, group, batchSize, [&indexSuffix](){ return indexSuffix; });
}
//----------------------------------------------------------------------------
template<typename A, typename G>
void addPrivateVarAccess(EnvironmentGroupMergedField<G> &env, const G &group, unsigned int batchSize, const std::string &indexSuffix)
{
    // Loop through variable references
    const A archetypeAdaptor(group.getArchetype());
    for(const auto &v : archetypeAdaptor.getDefs()) {
        // Add field with qualified type which indexes private pointer field
        const auto resolvedType = v.type.resolve(getGroup().getTypeContext());
        const auto qualifiedType = (v.access & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;
        env.add(qualifiedType, v.name, "$(_" + v.name + ")[" + getIndexFn(v.access, varRef));
    }
}
//----------------------------------------------------------------------------
template<typename V>
void addTypes(Transpiler::TypeChecker::EnvironmentBase &env, const std::vector<V> &vars, 
              const Type::TypeContext &typeContext, Transpiler::ErrorHandler::ErrorHandlerBase &errorHandle)
{
    for(const auto &v : vars) {
        const auto resolvedType = v.type.resolve(typeContext);
        const auto qualifiedType = (v.access & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;
        env.define(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, v.name, 0}, qualifiedType, errorHandler);
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// CodeGenerator::CustomConnectivityUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityUpdateGroupMerged::name = "CustomConnectivityUpdate";
//----------------------------------------------------------------------------
CustomConnectivityUpdateGroupMerged::CustomConnectivityUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext,
                                                                         const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups)
:   GroupMerged<CustomConnectivityUpdateInternal>(index, typeContext, groups)
{
    // Reserve vector of vectors to hold variables to update for all custom connectivity update groups, in archetype order
    m_SortedDependentVars.reserve(getGroups().size());

    // Loop through groups
    for(const auto &g : getGroups()) {
        // Get group's dependent variables
        const auto dependentVars = g.get().getDependentVariables();
        
        // Convert to list and sort
        // **NOTE** WUVarReferences are non-assignable so can't be sorted in a vector
        std::list<Models::WUVarReference> dependentVarsList(dependentVars.cbegin(), dependentVars.cend());
        dependentVarsList.sort([](const auto &a, const auto &b)
                               {  
                                   boost::uuids::detail::sha1 hashA;  
                                   Type::updateHash(a.getVar().type, hashA);
                                   Utils::updateHash(getVarAccessDuplication(a.getVar().access), hashA);

                                   boost::uuids::detail::sha1 hashB;
                                   Type::updateHash(b.getVar().type, hashB);
                                   Utils::updateHash(getVarAccessDuplication(b.getVar().access), hashB);

                                   return (hashA.get_digest() < hashB.get_digest());
                                });

        // Add vector for this groups update vars
        m_SortedDependentVars.emplace_back(dependentVarsList.cbegin(), dependentVarsList.cend());
    }
    
    // Check all vectors are the same size
    assert(std::all_of(m_SortedDependentVars.cbegin(), m_SortedDependentVars.cend(),
                       [this](const std::vector<Models::WUVarReference> &vars)
                       {
                           return (vars.size() == m_SortedDependentVars.front().size());
                       }));
    
    
    /*addField(Uint32, "rowStride",
            [&backend](const auto &cg, size_t) 
            { 
                const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                return std::to_string(backend.getSynapticMatrixRowStride(*sgInternal)); 
            });
    
    
    assert(getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE);
    addField(getArchetype().getSynapseGroup()->getSparseIndType().createPointer(), "ind", 
             [&backend](const auto &cg, size_t) 
             { 
                 return backend.getDeviceVarPrefix() + "ind" + cg.getSynapseGroup()->getName(); 
             });

    addField(Uint32.createPointer(), "rowLength",
             [&backend](const auto &cg, size_t) 
             { 
                 return backend.getDeviceVarPrefix() + "rowLength" + cg.getSynapseGroup()->getName(); 
             });

    // If some presynaptic variables are delayed, add delay pointer
    if (getArchetype().getPreDelayNeuronGroup() != nullptr) {
        addField(Uint32.createPointer(), "preSpkQuePtr", 
                 [&backend](const auto &cg, size_t) 
                 { 
                     return backend.getScalarAddressPrefix() + "spkQuePtr" + cg.getPreDelayNeuronGroup()->getName(); 
                 });
    }

    // If some postsynaptic variables are delayed, add delay pointer
    if (getArchetype().getPostDelayNeuronGroup() != nullptr) {
        addField(Uint32.createPointer(), "postSpkQuePtr", 
                 [&backend](const auto &cg, size_t) 
                 { 
                     return backend.getScalarAddressPrefix() + "spkQuePtr" + cg.getPostDelayNeuronGroup()->getName(); 
                 });
    }
    

    // Add variables to struct

    
    // Loop through sorted dependent variables
    for(size_t i = 0; i < getSortedArchetypeDependentVars().size(); i++) {
        auto resolvedType = getSortedArchetypeDependentVars().at(i).getVar().type.resolve(getTypeContext());
        addField(resolvedType.createPointer(), "_dependentVar" + std::to_string(i), 
                 [i, &backend, this](const auto&, size_t g) 
                 { 
                     const auto &varRef = m_SortedDependentVars[g][i];
                     return backend.getDeviceVarPrefix() + varRef.getVar().name + varRef.getTargetName(); 
                 });
    }*/
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdateGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getHashDigest(), hash);

    // Update hash with sizes of pre and postsynaptic neuron groups
    updateHash([](const auto &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getSrcNeuronGroup()->getNumNeurons();
               }, hash);

    updateHash([](const auto &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getTrgNeuronGroup()->getNumNeurons();
               }, hash);

    // Update hash with each group's parameters, derived parameters and variable references
    updateHash([](const auto &cg) { return cg.getParams(); }, hash);
    updateHash([](const auto &cg) { return cg.getDerivedParams(); }, hash);
    updateHash([](const auto &cg) { return cg.getVarReferences(); }, hash);
    updateHash([](const auto &cg) { return cg.getPreVarReferences(); }, hash);
    updateHash([](const auto &cg) { return cg.getPostVarReferences(); }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomConnectivityUpdateGroupMerged::generateUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged) const
{
    // Create new environment to add current source fields to neuron update group 
    EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> updateEnv(env, *this);

    // Add fields for number of pre and postsynaptic neurons
    updateEnv.addField(Type::Uint32.addConst(), "num_pre",
                       Type::Uint32, "numSrcNeurons", 
                       [](const auto &cg, size_t) 
                       { 
                           const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                           return std::to_string(sgInternal->getSrcNeuronGroup()->getNumNeurons());
                       });
    updateEnv.addField(Type::Uint32.addConst(), "num_post",
                       Type::Uint32, "numTrgNeurons", 
                       [](const auto &cg, size_t) 
                       { 
                           const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                           return std::to_string(sgInternal->getSrcNeuronGroup()->getNumNeurons());
                       });

    // Calculate index of start of row
    updateEnv.add(Type::Uint32.addConst(), "_row_start_idx", "rowStartIdx",
                  {updateEnv.addInitialiser("const unsigned int rowStartIdx = $(id_pre) * $(_row_stride);")});

    updateEnv.add(Type::Uint32.addConst(), "_syn_stride", "synStride",
                  {updateEnv.addInitialiser("const unsigned int synStride = $(num_pre) * $(_row_stride);")});

    // Substitute parameter and derived parameter names
    const auto *cm = getArchetype().getCustomConnectivityUpdateModel();
    updateEnv.addParams(cm->getParamNames(), "", &CustomConnectivityUpdateInternal::getParams, 
                        &CustomConnectivityUpdateGroupMerged::isParamHeterogeneous);
    updateEnv.addDerivedParams(cm->getDerivedParams(), "", &CustomConnectivityUpdateInternal::getDerivedParams, 
                               &CustomConnectivityUpdateGroupMerged::isDerivedParamHeterogeneous);
    updateEnv.addExtraGlobalParams(cm->getExtraGlobalParams(), backend.getDeviceVarPrefix());
    
    // Add presynaptic variables
    updateEnv.addVars<CustomConnectivityUpdatePreVarAdapter>(backend.getDeviceVarPrefix(), "$(id_pre)");

    // Loop through presynaptic variable references
    for(const auto &v : getArchetype().getCustomConnectivityUpdateModel()->getPreVarRefs()) {
        // If model isn't batched or variable isn't duplicated
        const auto &varRef = getArchetype().getPreVarReferences().at(v.name);
        if(modelMerged.getModel().getBatchSize() == 1 || !varRef.isDuplicated()) {
            // Determine index
            const std::string index = (varRef.getDelayNeuronGroup() != nullptr) ? "$(_pre_delay_offset) + $(id_pre)" : "$(id_pre)";
            
            // If variable access is read-only, qualify type with const
            const auto resolvedType = v.type.resolve(getTypeContext());
            const auto qualifiedType = (v.access & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;

            // Add field
            updateEnv.addField(qualifiedType, v.name,
                       resolvedType.createPointer(), v.name,
                       [&backend, v](const auto &g, size_t) 
                       { 
                           const auto varRef = g.getPreVarReferences().at(v.name);
                           return backend.getDeviceVarPrefix() + varRef.getVar().name + varRef.getTargetName(); 
                       },
                       index);
        }
    }

    // Add fields and private $(_XXX) substitutions for postsyanptic and synaptic variables and variables references as, 
    // while these can only be accessed by user code inside loop, they can be used directly by add/remove synapse functions
    addPrivateVarPointerFields<CustomConnectivityUpdateVarAdapter>(updateEnv, backend.getDeviceVarPrefix(), *this);
    addPrivateVarPointerFields<CustomConnectivityUpdatePostVarAdapter>(updateEnv, backend.getDeviceVarPrefix(), *this);
    addPrivateVarRefPointerFields<CustomConnectivityUpdateVarRefAdapter>(updateEnv, backend.getDeviceVarPrefix(), *this);
    addPrivateVarRefPointerFields<CustomConnectivityUpdatePostVarRefAdapter>(updateEnv, backend.getDeviceVarPrefix(), *this);

    // Add private fields for dependent variables
    for(size_t i = 0; i < getSortedArchetypeDependentVars().size(); i++) {
        auto resolvedType = getSortedArchetypeDependentVars().at(i).getVar().type.resolve(getTypeContext());
        updateEnv.addField(resolvedType.createPointer(), "_dependent_var_" + std::to_string(i), "dependentVar" + std::to_string(i),
                           [i, &backend, this](const auto&, size_t g) 
                           { 
                               const auto &varRef = m_SortedDependentVars[g][i];
                               return backend.getDeviceVarPrefix() + varRef.getVar().name + varRef.getTargetName(); 
                           });
    }

    
    // Get variables which will need to be manipulated when adding and removing synapses
    const auto ccuVars = cm->getVars();
    const auto ccuVarRefs = cm->getVarRefs();
    const auto &dependentVars = getSortedArchetypeDependentVars();
    std::vector<Type::ResolvedType> addSynapseTypes{Type::Uint32};
    addSynapseTypes.reserve(1 + ccuVars.size() + ccuVarRefs.size() + dependentVars.size());

    // Generate code to add a synapse to this row
    std::stringstream addSynapseStream;
    CodeStream addSynapse(addSynapseStream);
    {
        CodeStream::Scope b(addSynapse);

        // Assert that there is space to add synapse
        backend.genAssert(addSynapse, "$(_row_length)[$(id_pre)] < $(_row_stride)");

        // Calculate index to insert synapse
        addSynapse << "const unsigned newIdx = $(_row_start_idx) + $(_row_length)[$(id_pre)];" << std::endl;

        // Set postsynaptic target to parameter 0
        addSynapse << "$(_ind)[newIdx] = $(0);" << std::endl;
 
        // Use subsequent parameters to initialise new synapse's custom connectivity update model variables
        for (size_t i = 0; i < ccuVars.size(); i++) {
            addSynapse << "$(_" << ccuVars[i].name << ")[newIdx] = $(" << (1 + i) << ");" << std::endl;
            addSynapseTypes.push_back(ccuVars[i].type.resolve(getTypeContext()));
        }

        // Use subsequent parameters to initialise new synapse's variables referenced via the custom connectivity update
        for (size_t i = 0; i < ccuVarRefs.size(); i++) {
            // If model is batched and this variable is duplicated
            if ((modelMerged.getModel().getBatchSize() > 1) && getArchetype().getVarReferences().at(ccuVarRefs[i].name).isDuplicated()) 
            {
                // Copy parameter into a register (just incase it's e.g. a RNG call) and copy into all batches
                addSynapse << "const " << ccuVarRefs[i].type.resolve(getTypeContext()).getName() << " _" << ccuVarRefs[i].name << "Val = $(" << (1 + ccuVars.size() + i) << ");" << std::endl;
                addSynapse << "for(int b = 0; b < " << modelMerged.getModel().getBatchSize() << "; b++)";
                {
                    CodeStream::Scope b(addSynapse);
                    addSynapse << "$(_" << ccuVarRefs[i].name << ")[(b * $(_syn_stride)) + newIdx] = _" << ccuVarRefs[i].name << "Val;" << std::endl;
                }
            }
            // Otherwise, write parameter straight into var reference
            else {
                addSynapse << "$(_" << ccuVarRefs[i].name << ")[newIdx] = $(" << (1 + ccuVars.size() + i) << ");" << std::endl;
            }

            addSynapseTypes.push_back(ccuVarRefs[i].type.resolve(getTypeContext()));
        }
        
        // Loop through any other dependent variables
        for (size_t i = 0; i < dependentVars.size(); i++) {
            // If model is batched and this dependent variable is duplicated
            if ((modelMerged.getModel().getBatchSize() > 1) && dependentVars.at(i).isDuplicated())
            {
                // Loop through all batches and zero
                addSynapse << "for(int b = 0; b < " << modelMerged.getModel().getBatchSize() << "; b++)";
                {
                    CodeStream::Scope b(addSynapse);
                    addSynapse << "$(_dependent_var_" << i << ")[(b * $(_syn_stride)) + newIdx] = 0;" << std::endl;
                }
            }
            // Otherwise, zero var reference
            else {
                addSynapse << "$(_dependent_var_" << i << ")[newIdx] = 0;" << std::endl;
            }

            addSynapseTypes.push_back(dependentVars.at(i).getVar().type.resolve(getTypeContext()));
        }

        // Increment row length
        // **NOTE** this will also effect any forEachSynapse loop currently in operation
        addSynapse << "$(_row_length)[$(id_pre)]++;" << std::endl;
    }

    // Add function substitution with parameters to initialise custom connectivity update variables and variable references
    updateEnv.add(Type::ResolvedType::createFunction(Type::Void, addSynapseTypes), "add_synapse", addSynapseStream.str());

    // Generate code to remove a synapse from this row
    std::stringstream removeSynapseStream;
    CodeStream removeSynapse(removeSynapseStream);
    {
        CodeStream::Scope b(removeSynapse);

        // Calculate index we want to copy synapse from
        removeSynapse << "const unsigned lastIdx = $(_row_start_idx) + $(_row_length)[$(id_pre)] - 1;" << std::endl;

        // Copy postsynaptic target from end of row over synapse to be deleted
        removeSynapse << "$(_ind)[idx] = $(_ind)[lastIdx];" << std::endl;

        // Copy custom connectivity update variables from end of row over synapse to be deleted
        for (size_t i = 0; i < ccuVars.size(); i++) {
            removeSynapse << "$(_" << ccuVars[i].name << ")[idx] = $(_" << ccuVars[i].name << ")[lastIdx];" << std::endl;
        }
        
        // Loop through variable references
        for (size_t i = 0; i < ccuVarRefs.size(); i++) {
            // If model is batched and this variable is duplicated
            if ((modelMerged.getModel().getBatchSize() > 1) 
                && getArchetype().getVarReferences().at(ccuVarRefs[i].name).isDuplicated())
            {
                // Loop through all batches and copy custom connectivity update variable references from end of row over synapse to be deleted
                removeSynapse << "for(int b = 0; b < " << modelMerged.getModel().getBatchSize() << "; b++)";
                {
                    CodeStream::Scope b(addSynapse);
                    removeSynapse << "$(_" << ccuVarRefs[i].name << ")[(b * $(_syn_stride)) + idx] = ";
                    removeSynapse << "$(_" << ccuVarRefs[i].name << ")[(b * $(_syn_stride)) + lastIdx];" << std::endl;
                }
            }
            // Otherwise, copy custom connectivity update variable references from end of row over synapse to be deleted
            else {
                removeSynapse << "$(_" << ccuVarRefs[i].name << ")[idx] = $(_" << ccuVarRefs[i].name << ")[lastIdx];" << std::endl;
            }
        }
        
        // Loop through any other dependent variables
        for (size_t i = 0; i < dependentVars.size(); i++) {
            // If model is batched and this dependent variable is duplicated
            if ((modelMerged.getModel().getBatchSize() > 1) && dependentVars.at(i).isDuplicated()) {
                // Loop through all batches and copy dependent variable from end of row over synapse to be deleted
                removeSynapse << "for(int b = 0; b < " << modelMerged.getModel().getBatchSize() << "; b++)";
                {
                    CodeStream::Scope b(removeSynapse);
                    removeSynapse << "$(_dependent_var_" << i << ")[(b * $(_syn_stride)) + idx] = ";
                    removeSynapse << "$(_dependent_var_" << i << ")[(b * $(_syn_stride)) + lastIdx];" << std::endl;
                }
            }
            // Otherwise, copy dependent variable from end of row over synapse to be deleted
            else {
                removeSynapse << "$(_dependent_var_" << i << ")[idx] = $(_dependent_var_" << i << ")[lastIdx];" << std::endl;
            }
        }

        // Decrement row length
        // **NOTE** this will also effect any forEachSynapse loop currently in operation
        removeSynapse << "$(_row_length)[$(id_pre)]--;" << std::endl;

        // Decrement loop counter so synapse j will get processed
        removeSynapse << "j--;" << std::endl;
    }

    // Add function substitution with parameters to initialise custom connectivity update variables and variable references
    updateEnv.add(Type::ResolvedType::createFunction(Type::Void, {}), "remove_synapse", removeSynapseStream.str());

    // Pretty print code back to environment
    Transpiler::ErrorHandler errorHandler("Current source injection" + std::to_string(getIndex()));
    prettyPrintStatements(cm->getRowUpdateCode(), getTypeContext(), updateEnv, errorHandler, 
                          // Within for_each_synapse loops, define the following types
                          [this](auto &env, auto &errorHandler)
                          {
                              // Add typed indices
                              env.define(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, "id_post", 0}, Type::Uint32.addConst(), errorHandler);
                              env.define(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, "id_syn", 0}, Type::Uint32.addConst(), errorHandler);

                              // Add types for variables and variable references accessible within loop
                              // **TODO** filter
                              addTypes(env, getArchetype().getCustomConnectivityUpdateModel()->getVars(), getTypeContext(), errorHandler);
                              addTypes(env, getArchetype().getCustomConnectivityUpdateModel()->getPostVars(), getTypeContext(), errorHandler);
                              addTypes(env, getArchetype().getCustomConnectivityUpdateModel()->getVarRefs(), getTypeContext(), errorHandler);
                              addTypes(env, getArchetype().getCustomConnectivityUpdateModel()->getPostVarRefs(), getTypeContext(), errorHandler);
                          },
                          [&backend, &modelMerged, this](auto &env, auto generateBody)
                          {
                              EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> bodyEnv(env, *this);
                              bodyEnv.getStream() << printSubs("for(int j = 0; j < $(_row_length)[$(id_pre)]; j++)", bodyEnv);
                              {
                                  CodeStream::Scope b(bodyEnv.getStream());

                                  // Add postsynaptic and synaptic indices
                                  bodyEnv.add(Type::Uint32.addConst(), "id_post", "$(_ind)[$(_row_start_idx) + j]");
                                  bodyEnv.add(Type::Uint32.addConst(), "id_syn", "idx",
                                              {bodyEnv.addInitialiser("const unsigned int idx = $(_row_start_idx) + j;")});

                                  // Add postsynaptic and synaptic variables
                                  bodyEnv.addVars<CustomConnectivityUpdateVarAdapter>(backend.getDeviceVarPrefix(), "$(id_syn)");
                                  bodyEnv.addVars<CustomConnectivityUpdatePostVarAdapter>(backend.getDeviceVarPrefix(), "$(id_post)");

                                  // Add postsynaptic and synapse variable references, only exposing those that aren't batched
                                  addPrivateVarRefAccess<CustomConnectivityUpdateVarRefAdapter>(bodyEnv, *this, modelMerged.getModel().getBatchSize(), "$(id_syn)");
                                  addPrivateVarRefAccess<CustomConnectivityUpdatePostVarRefAdapter>(
                                      bodyEnv, *this, modelMerged.getModel().getBatchSize(), 
                                      [](VarAccessMode a, const Models::VarReference &varRef)
                                      { 
                                          if(varRef.getDelayNeuronGroup() != nullptr) {
                                              return "$(_post_delay_offset) + $(id_post)"; 
                                          }
                                          else {
                                              return "$(id_post)"; 
                                          }
                                       });
                                    
                                    // Generate body of for_each_synapse loop within this new environment
                                    generateBody(bodyEnv);
                              }
                          });
}
//----------------------------------------------------------------------------
bool CustomConnectivityUpdateGroupMerged::isParamHeterogeneous(const std::string &name) const
{
    return isParamValueHeterogeneous(name, [](const CustomConnectivityUpdateInternal &cg) { return cg.getParams(); });
}
//----------------------------------------------------------------------------
bool CustomConnectivityUpdateGroupMerged::isDerivedParamHeterogeneous(const std::string &name) const
{
    return isParamValueHeterogeneous(name, [](const CustomConnectivityUpdateInternal &cg) { return cg.getDerivedParams(); });
}

//----------------------------------------------------------------------------
// CodeGenerator::CustomConnectivityHostUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityHostUpdateGroupMerged::name = "CustomConnectivityHostUpdate";
//----------------------------------------------------------------------------
/*CustomConnectivityHostUpdateGroupMerged::CustomConnectivityHostUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext,
                                                                                 const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups)
:   CustomConnectivityUpdateGroupMergedBase(index, typeContext, groups)
{
    using namespace Type;

    // Add pre and postsynaptic variables
    const auto *cm = getArchetype().getCustomConnectivityUpdateModel();
    addVars(backend, cm->getPreVars(), &CustomConnectivityUpdateInternal::getPreVarLocation);
    addVars(backend, cm->getPostVars(), &CustomConnectivityUpdateInternal::getPostVarLocation);

    // Add host extra global parameters
    for(const auto &e : cm->getExtraGlobalParams()) {
        const auto resolvedType = e.type.resolve(getTypeContext());
        addField(resolvedType.createPointer(), e.name,
                 [e](const auto &g, size_t) { return e.name + g.getName(); },
                 GroupMergedFieldType::HOST_DYNAMIC);

        if(!backend.getDeviceVarPrefix().empty()) {
            addField(resolvedType.createPointer(), backend.getDeviceVarPrefix() + e.name,
                     [e, &backend](const auto &g, size_t)
                     {
                         return backend.getDeviceVarPrefix() + e.name + g.getName();
                     },
                     GroupMergedFieldType::DYNAMIC);
        }
    }
             
}*/
//----------------------------------------------------------------------------
void CustomConnectivityHostUpdateGroupMerged::generateUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged) const
{
    CodeStream::Scope b(os);
    os << "// merged custom connectivity host update group " << getIndex() << std::endl;
    os << "for(unsigned int g = 0; g < " << getGroups().size() << "; g++)";
    {
        CodeStream::Scope b(os);

        // Get reference to group
        os << "const auto *group = &mergedCustomConnectivityHostUpdateGroup" << getIndex() << "[g]; " << std::endl;

        // Create substitutions
        const auto *cm = getArchetype().getCustomConnectivityUpdateModel();
        Substitutions subs;
        subs.addVarSubstitution("rng", "hostRNG");
        subs.addVarSubstitution("num_pre", "group->numSrcNeurons");
        subs.addVarSubstitution("num_post", "group->numTrgNeurons");
        subs.addVarNameSubstitution(cm->getExtraGlobalParams(), "", "group->");
        subs.addVarNameSubstitution(cm->getPreVars(), "", "group->");
        subs.addVarNameSubstitution(cm->getPostVars(), "", "group->");
        subs.addParamValueSubstitution(cm->getParamNames(), getArchetype().getParams(),
                                       [this](const std::string &p) { return isParamHeterogeneous(p); },
                                       "", "group->");
        subs.addVarValueSubstitution(cm->getDerivedParams(), getArchetype().getDerivedParams(),
                                     [this](const std::string & p) { return isDerivedParamHeterogeneous(p); },
                                     "", "group->");

        // Loop through EGPs
        for(const auto &egp : cm->getExtraGlobalParams()) {
            const auto resolvedType = egp.type.resolve(getTypeContext());

            // Generate code to push this EGP with count specified by $(0)
            std::stringstream pushStream;
            CodeStream push(pushStream);
            backend.genVariableDynamicPush(push, resolvedType, egp.name,
                                           VarLocation::HOST_DEVICE, "$(0)", "group->");

            // Add substitution
            subs.addFuncSubstitution("push" + egp.name + "ToDevice", 1, pushStream.str());

            // Generate code to pull this EGP with count specified by $(0)
            std::stringstream pullStream;
            CodeStream pull(pullStream);
            backend.genVariableDynamicPull(pull, resolvedType, egp.name,
                                           VarLocation::HOST_DEVICE, "$(0)", "group->");

            // Add substitution
            subs.addFuncSubstitution("pull" + egp.name + "FromDevice", 1, pullStream.str());
        }

        addVarPushPullFuncSubs(backend, subs, cm->getPreVars(), "group->numSrcNeurons",
                               &CustomConnectivityUpdateInternal::getPreVarLocation);
        addVarPushPullFuncSubs(backend, subs, cm->getPostVars(), "group->numTrgNeurons",
                               &CustomConnectivityUpdateInternal::getPostVarLocation);

        // Apply substitutons to row update code and write out
        std::string code = cm->getHostUpdateCode();
        subs.applyCheckUnreplaced(code, "custom connectivity host update : merged" + std::to_string(getIndex()));
        //code = ensureFtype(code, modelMerged.getModel().getPrecision());
        os << code;
    }
}
//----------------------------------------------------------------------------
bool CustomConnectivityHostUpdateGroupMerged::isParamHeterogeneous(const std::string &name) const
{
    return isParamValueHeterogeneous(name, [](const CustomConnectivityUpdateInternal &cg) { return cg.getParams(); });
}
//----------------------------------------------------------------------------
bool CustomConnectivityHostUpdateGroupMerged::isDerivedParamHeterogeneous(const std::string &name) const
{
    return isParamValueHeterogeneous(name, [](const CustomConnectivityUpdateInternal &cg) { return cg.getDerivedParams(); });
}
//----------------------------------------------------------------------------
void CustomConnectivityHostUpdateGroupMerged::addVarPushPullFuncSubs(const BackendBase &backend, Substitutions &subs, 
                                                                     const Models::Base::VarVec &vars, const std::string &count,
                                                                     VarLocation(CustomConnectivityUpdateInternal:: *getVarLocationFn)(const std::string&) const) const
{
    // Loop through variables
    for(const auto &v : vars) {
        const auto resolvedType = v.type.resolve(getTypeContext());

        // If var is located on the host
        const auto loc = std::invoke(getVarLocationFn, getArchetype(), v.name);
        if (loc & VarLocation::HOST) {
            // Generate code to push this variable
            std::stringstream pushStream;
            CodeStream push(pushStream);
            backend.genVariableDynamicPush(push, resolvedType, v.name,
                                           loc, count, "group->");

            // Add substitution
            subs.addFuncSubstitution("push" + v.name + "ToDevice", 0, pushStream.str());

            // Generate code to pull this variable
            // **YUCK** these EGP functions should probably just be called dynamic or something
            std::stringstream pullStream;
            CodeStream pull(pullStream);
            backend.genVariableDynamicPull(pull, resolvedType, v.name,
                                           loc, count, "group->");

            // Add substitution
            subs.addFuncSubstitution("pull" + v.name + "FromDevice", 0, pullStream.str());
        }
    }
}
//----------------------------------------------------------------------------
void CustomConnectivityHostUpdateGroupMerged::addVars(const BackendBase &backend, const Models::Base::VarVec &vars,
                                                      VarLocation(CustomConnectivityUpdateInternal:: *getVarLocationFn)(const std::string&) const)
{
    using namespace Type;

    // Loop through variables
    for(const auto &v : vars) {
        // If var is located on the host
        const auto resolvedType = v.type.resolve(getTypeContext());
        if (std::invoke(getVarLocationFn, getArchetype(), v.name) & VarLocation::HOST) {
            addField(resolvedType.createPointer(), v.name,
                    [v](const auto &g, size_t) { return v.name + g.getName(); },
                    GroupMergedFieldType::HOST);

            if(!backend.getDeviceVarPrefix().empty())  {
                // **TODO** I think could use addPointerField
                addField(resolvedType.createPointer(), backend.getDeviceVarPrefix() + v.name,
                         [v, &backend](const auto &g, size_t)
                         {
                             return backend.getDeviceVarPrefix() + v.name + g.getName();
                         });
            }
        }
    }
}
