#include "code_generator/customConnectivityUpdateGroupMerged.h"

// Standard C++ includes
#include <list>

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"
#include "code_generator/standardLibrary.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

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
                                   Type::updateHash(a.getVarType(), hashA);
                                   Utils::updateHash(a.getVarDims(), hashA);

                                   boost::uuids::detail::sha1 hashB;
                                   Type::updateHash(b.getVarType(), hashB);
                                   Utils::updateHash(b.getVarDims(), hashB);

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
    updateHash([](const auto& cg) { return cg.getEGPReferences(); }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomConnectivityUpdateGroupMerged::generateUpdate(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize)
{
    // Create new environment to add current source fields to neuron update group 
    EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> updateEnv(env, *this);

    // Calculate index of start of row
    const auto indexType = backend.getSynapseIndexType(*this);
    const auto indexTypeName = indexType.getName();
    updateEnv.add(Type::Uint32.addConst(), "_row_start_idx", "rowStartIdx",
                  {updateEnv.addInitialiser("const unsigned int rowStartIdx = $(id_pre) * $(_row_stride);")});

    updateEnv.add(indexType.addConst(), "_syn_stride", "synStride",
                  {updateEnv.addInitialiser("const " + indexTypeName + " synStride = (" + indexTypeName + ")$(num_pre) * $(_row_stride);")});

    // Expose row length and row stride
    updateEnv.add(Type::Uint32.addConst(), "row_length", "$(_row_length)[$(id_pre)]");
    updateEnv.add(Type::Uint32.addConst(), "row_stride", "$(_row_stride)");

    // Substitute parameter and derived parameter names
    const auto *cm = getArchetype().getModel();
    updateEnv.addParams(cm->getParams(), "", &CustomConnectivityUpdateInternal::getParams, 
                        &CustomConnectivityUpdateGroupMerged::isParamHeterogeneous,
                        &CustomConnectivityUpdateInternal::isParamDynamic);
    updateEnv.addDerivedParams(cm->getDerivedParams(), "", &CustomConnectivityUpdateInternal::getDerivedParams, 
                               &CustomConnectivityUpdateGroupMerged::isDerivedParamHeterogeneous);
    updateEnv.addExtraGlobalParams(cm->getExtraGlobalParams());
    updateEnv.addExtraGlobalParamRefs(cm->getExtraGlobalParamRefs());
    
    // Add presynaptic variables
    updateEnv.addVars<CustomConnectivityUpdatePreVarAdapter>("$(id_pre)");

    // Loop through presynaptic variable references
    for(const auto &v : getArchetype().getModel()->getPreVarRefs()) {
        // If model isn't batched or variable isn't duplicated
        const auto &varRef = getArchetype().getPreVarReferences().at(v.name);
        if(batchSize == 1 || !(varRef.getVarDims() & VarAccessDim::BATCH)) {
            // Determine index
            const std::string index = (varRef.getDelayNeuronGroup() != nullptr) ? "$(_pre_delay_offset) + $(id_pre)" : "$(id_pre)";
            
            // If variable access is read-only, qualify type with const
            const auto resolvedType = v.type.resolve(getTypeContext());
            const auto qualifiedType = (v.access & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;

            // Add field
            updateEnv.addField(qualifiedType, v.name,
                       resolvedType.createPointer(), v.name,
                       [v](const auto &runtime, const auto &g, size_t) 
                       { 
                           return g.getPreVarReferences().at(v.name).getTargetArray(runtime);
                       },
                       index);
        }
    }

    // Add fields and private $(_XXX) substitutions for postsyanptic and synaptic variables and variables references as, 
    // while these can only be accessed by user code inside loop, they can be used directly by add/remove synapse functions
    updateEnv.addVarPointers<CustomConnectivityUpdateVarAdapter>("", true);
    updateEnv.addVarPointers<CustomConnectivityUpdatePostVarAdapter>("", true);
    updateEnv.addVarRefPointers<CustomConnectivityUpdateVarRefAdapter>(true);
    updateEnv.addVarRefPointers<CustomConnectivityUpdatePostVarRefAdapter>(true);

    // Add private fields for dependent variables
    for(size_t i = 0; i < getSortedArchetypeDependentVars().size(); i++) {
        auto resolvedType = getSortedArchetypeDependentVars().at(i).getVarType().resolve(getTypeContext());
        updateEnv.addField(resolvedType.createPointer(), "_dependent_var_" + std::to_string(i), "dependentVar" + std::to_string(i),
                           [i, this](const auto &runtime, const auto&, size_t g) 
                           { 
                               return m_SortedDependentVars[g][i].getTargetArray(runtime);
                           });
    }

    
    // Get variables which will need to be manipulated when adding and removing synapses
    const auto ccuVars = cm->getVars();
    const auto ccuVarRefs = cm->getVarRefs();
    const auto &dependentVars = getSortedArchetypeDependentVars();
    std::vector<Type::ResolvedType> addSynapseTypes{Type::Uint32};
    addSynapseTypes.reserve(1 + ccuVars.size() + ccuVarRefs.size());

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
            const auto &varRef = getArchetype().getVarReferences().at(ccuVarRefs[i].name);
            if (batchSize > 1 && (varRef.getVarDims() & VarAccessDim::BATCH)) {
                // Copy parameter into a register (just incase it's e.g. a RNG call) and copy into all batches
                addSynapse << "const " << ccuVarRefs[i].type.resolve(getTypeContext()).getName() << " _" << ccuVarRefs[i].name << "Val = $(" << (1 + ccuVars.size() + i) << ");" << std::endl;
                addSynapse << "for(int b = 0; b < " << batchSize << "; b++)";
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
            if (batchSize > 1 && (dependentVars.at(i).getVarDims() & VarAccessDim::BATCH)) {
                // Loop through all batches and zero
                addSynapse << "for(int b = 0; b < " << batchSize << "; b++)";
                {
                    CodeStream::Scope b(addSynapse);
                    addSynapse << "$(_dependent_var_" << i << ")[(b * $(_syn_stride)) + newIdx] = 0;" << std::endl;
                }
            }
            // Otherwise, zero var reference
            else {
                addSynapse << "$(_dependent_var_" << i << ")[newIdx] = 0;" << std::endl;
            }
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
        removeSynapse << "$(_ind)[$(id_syn)] = $(_ind)[lastIdx];" << std::endl;

        // Copy custom connectivity update variables from end of row over synapse to be deleted
        for (size_t i = 0; i < ccuVars.size(); i++) {
            removeSynapse << "$(_" << ccuVars[i].name << ")[$(id_syn)] = $(_" << ccuVars[i].name << ")[lastIdx];" << std::endl;
        }
        
        // Loop through variable references
        for (size_t i = 0; i < ccuVarRefs.size(); i++) {
            // If model is batched and this variable is duplicated
            const auto &varRef = getArchetype().getVarReferences().at(ccuVarRefs[i].name);
            if (batchSize > 1 && (varRef.getVarDims() & VarAccessDim::BATCH)) {
                // Loop through all batches and copy custom connectivity update variable references from end of row over synapse to be deleted
                removeSynapse << "for(int b = 0; b < " << batchSize << "; b++)";
                {
                    CodeStream::Scope b(addSynapse);
                    removeSynapse << "$(_" << ccuVarRefs[i].name << ")[(b * $(_syn_stride)) + $(id_syn)] = ";
                    removeSynapse << "$(_" << ccuVarRefs[i].name << ")[(b * $(_syn_stride)) + $(id_syn)];" << std::endl;
                }
            }
            // Otherwise, copy custom connectivity update variable references from end of row over synapse to be deleted
            else {
                removeSynapse << "$(_" << ccuVarRefs[i].name << ")[$(id_syn)] = $(_" << ccuVarRefs[i].name << ")[lastIdx];" << std::endl;
            }
        }
        
        // Loop through any other dependent variables
        for (size_t i = 0; i < dependentVars.size(); i++) {
            // If model is batched and this dependent variable is duplicated
            if (batchSize > 1 && (dependentVars.at(i).getVarDims() & VarAccessDim::BATCH)) {
                // Loop through all batches and copy dependent variable from end of row over synapse to be deleted
                removeSynapse << "for(int b = 0; b < " << batchSize << "; b++)";
                {
                    CodeStream::Scope b(removeSynapse);
                    removeSynapse << "$(_dependent_var_" << i << ")[(b * $(_syn_stride)) + $(id_syn)] = ";
                    removeSynapse << "$(_dependent_var_" << i << ")[(b * $(_syn_stride)) + lastIdx];" << std::endl;
                }
            }
            // Otherwise, copy dependent variable from end of row over synapse to be deleted
            else {
                removeSynapse << "$(_dependent_var_" << i << ")[$(id_syn)] = $(_dependent_var_" << i << ")[lastIdx];" << std::endl;
            }
        }

        // Decrement row length
        // **NOTE** this will also effect any forEachSynapse loop currently in operation
        removeSynapse << "$(_row_length)[$(id_pre)]--;" << std::endl;

        // Decrement loop counter so synapse j will get processed
        removeSynapse << "j--;" << std::endl;
    }

    // Pretty print code back to environment
    Transpiler::ErrorHandler errorHandler("Custom connectivity update '" + getArchetype().getName() + "' row update code");
    prettyPrintStatements(getArchetype().getRowUpdateCodeTokens(), getTypeContext(), updateEnv, errorHandler, 
                          // Within for_each_synapse loops, define the following types
                          [&indexType, this](auto &env, auto &errorHandler)
                          {
                              // Add type of remove synapse function
                              env.define(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, "remove_synapse", 0}, Type::ResolvedType::createFunction(Type::Void, {}), errorHandler);

                              // Add typed indices
                              env.define(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, "id_post", 0}, Type::Uint32.addConst(), errorHandler);
                              env.define(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, "id_syn", 0}, indexType.addConst(), errorHandler);

                              // Add types for variables and variable references accessible within loop
                              // **NOTE** postsynaptic variables and variable references are read-only as many rows will access simultaneously
                              addTypes(env, getArchetype().getModel()->getVars(), errorHandler);
                              addTypes(env, getArchetype().getModel()->getPostVars(), errorHandler, true);
                              addTypes(env, getArchetype().getModel()->getVarRefs(), errorHandler);
                              addTypes(env, getArchetype().getModel()->getPostVarRefs(), errorHandler, true);
                          },
                          [batchSize, &backend, &indexType, &indexTypeName, &removeSynapseStream, this](auto &env, auto generateBody)
                          {
                              env.print("for(int j = 0; j < $(_row_length)[$(id_pre)]; j++)");
                              {
                                  CodeStream::Scope b(env.getStream());
                                  EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> bodyEnv(env, *this);

                                  // Add postsynaptic and synaptic indices
                                  bodyEnv.add(Type::Uint32.addConst(), "id_post", "$(_ind)[$(_row_start_idx) + j]");
                                  bodyEnv.add(indexType.addConst(), "id_syn", "idx",
                                              {bodyEnv.addInitialiser("const " + indexTypeName + " idx = (" + indexTypeName + ")$(_row_start_idx) + j;")});

                                  // Add postsynaptic and synaptic variables
                                  bodyEnv.addVars<CustomConnectivityUpdateVarAdapter>("$(id_syn)");
                                  bodyEnv.addVars<CustomConnectivityUpdatePostVarAdapter>("$(id_post)");

                                  // Add postsynaptic and synapse variable references, only exposing those that aren't batched
                                  addPrivateVarRefAccess<CustomConnectivityUpdateVarRefAdapter>(bodyEnv, batchSize, "$(id_syn)");
                                  addPrivateVarRefAccess<CustomConnectivityUpdatePostVarRefAdapter>(
                                      bodyEnv, batchSize, 
                                      [](VarAccessMode, const Models::VarReference &varRef)
                                      { 
                                          if(varRef.getDelayNeuronGroup() != nullptr) {
                                              return "$(_post_delay_offset) + $(id_post)"; 
                                          }
                                          else {
                                              return "$(id_post)"; 
                                          }
                                      });

                                    // Add function substitution with parameters to initialise custom connectivity update variables and variable references
                                    bodyEnv.add(Type::ResolvedType::createFunction(Type::Void, {}), "remove_synapse", removeSynapseStream.str());
                                    
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

// ----------------------------------------------------------------------------
// CustomConnectivityRemapUpdateGroupMerged
// ----------------------------------------------------------------------------
const std::string CustomConnectivityRemapUpdateGroupMerged::name = "CustomConnectivityRemapUpdate";
//----------------------------------------------------------------------------
void CustomConnectivityRemapUpdateGroupMerged::generateUpdate(const BackendBase &backend, EnvironmentExternalBase &env)
{
}
//----------------------------------------------------------------------------
// CodeGenerator::CustomConnectivityHostUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityHostUpdateGroupMerged::name = "CustomConnectivityHostUpdate";
//----------------------------------------------------------------------------
void CustomConnectivityHostUpdateGroupMerged::generateUpdate(const BackendBase &backend, EnvironmentExternalBase &env)
{
    // **THINK** would some additional standard library functions like memset be useful?
    EnvironmentLibrary rngEnv(env, StandardLibrary::getHostRNGFunctions(getScalarType()));
    CodeStream::Scope b(rngEnv.getStream());

    rngEnv.getStream() << "// merged custom connectivity host update group " << getIndex() << std::endl;
    rngEnv.getStream() << "for(unsigned int g = 0; g < " << getGroups().size() << "; g++)";
    {
        CodeStream::Scope b(rngEnv.getStream());

        // Get reference to group
        rngEnv.getStream() << "const auto *group = &mergedCustomConnectivityHostUpdateGroup" << getIndex() << "[g]; " << std::endl;

        // Create matching environment
        EnvironmentGroupMergedField<CustomConnectivityHostUpdateGroupMerged> groupEnv(rngEnv, *this);

        // Add fields for number of pre and postsynaptic neurons
        groupEnv.addField(Type::Uint32.addConst(), "num_pre",
                          Type::Uint32, "numSrcNeurons", 
                          [](const auto&, const auto &cg, size_t) 
                          { 
                              const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                              return sgInternal->getSrcNeuronGroup()->getNumNeurons();
                          });
        groupEnv.addField(Type::Uint32.addConst(), "num_post",
                          Type::Uint32, "numTrgNeurons", 
                          [](const auto&, const auto &cg, size_t) 
                          { 
                              const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                              return sgInternal->getSrcNeuronGroup()->getNumNeurons();
                          });

        // Expose row stride        
        groupEnv.addField(Type::Uint32.addConst(), "row_stride",
                          Type::Uint32, "rowStride", 
                          [&backend](const auto&, const auto &cg, size_t) { return backend.getSynapticMatrixRowStride(*cg.getSynapseGroup()); });

        // If synapse group connectivity is accessible from the host
        // **TODO** variable locations probably need hashing in host update group hash
        const auto loc = getArchetype().getSynapseGroup()->getSparseConnectivityLocation();
        if(loc & VarLocationAttribute::HOST) {
            // Add host pointer field for row length
            groupEnv.addField(Type::Uint32.createPointer(), "_row_length", "rowLength",
                              [](const auto &runtime, const auto &cg, size_t) 
                              { 
                                  return runtime.getArray(*cg.getSynapseGroup(), "rowLength");
                              },
                              "", GroupMergedFieldType::HOST);

            // Add substitution for direct access to field
            groupEnv.add(Type::Uint32.addConst().createPointer(), "row_length", "$(_row_length)");

            // If backend requires seperate device objects, add additional (private) field)
            if(backend.isArrayDeviceObjectRequired()) {
                groupEnv.addField(Type::Uint32.createPointer(), "_d_row_length", "d_rowLength",
                                  [](const auto &runtime, const auto &cg, size_t)
                                  {
                                      return runtime.getArray(*cg.getSynapseGroup(), "rowLength");
                                  });
            }

            // Generate code to pull this variable
            std::stringstream pullStream;
            CodeStream pull(pullStream);
            backend.genLazyVariableDynamicPull(pull, Type::Uint32, "row_length",
                                               loc, "$(num_pre)");

            // Add substitution
            groupEnv.add(Type::PushPull, "pullRowLengthFromDevice", pullStream.str());
        }



        // Substitute parameter and derived parameter names
        const auto *cm = getArchetype().getModel();
        groupEnv.addParams(cm->getParams(), "", &CustomConnectivityUpdateInternal::getParams, 
                           &CustomConnectivityHostUpdateGroupMerged::isParamHeterogeneous,
                           &CustomConnectivityUpdateInternal::isParamDynamic);
        groupEnv.addDerivedParams(cm->getDerivedParams(), "", &CustomConnectivityUpdateInternal::getDerivedParams, 
                                  &CustomConnectivityHostUpdateGroupMerged::isDerivedParamHeterogeneous);

        // Loop through EGPs
        for(const auto &egp : cm->getExtraGlobalParams()) {
            // If EGP is located on the host
            const auto loc = getArchetype().getExtraGlobalParamLocation(egp.name);
            if(loc & VarLocationAttribute::HOST) {
                // Add pointer field to allow user code to access
                const auto resolvedType = egp.type.resolve(getTypeContext());
                assert(!resolvedType.isPointer());
                const auto pointerType = resolvedType.createPointer();

                // Add field for host pointer
                groupEnv.addField(pointerType, "_" + egp.name, egp.name,
                                  [egp](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, egp.name); },
                                  "", GroupMergedFieldType::HOST_DYNAMIC);

                // Add substitution for direct access to field
                // **NOTE** we don't just directly expose the field as genLazyVariableDynamicXXX functions expect underscoring
                groupEnv.add(pointerType, egp.name, "$(_" + egp.name + ")");

                // If backend requires seperate device objects, add additional (private) field)
                if(backend.isArrayDeviceObjectRequired()) {
                    groupEnv.addField(pointerType, "_d_" + egp.name, "d_" + egp.name,
                                      [egp](const auto &runtime, const auto &g, size_t)
                                      {
                                          return runtime.getArray(g, egp.name);
                                      },
                                      "", GroupMergedFieldType::DYNAMIC);
                }

                // Generate code to push this EGP with count specified by $(0)
                std::stringstream pushStream;
                CodeStream push(pushStream);
                backend.genLazyVariableDynamicPush(push, resolvedType, egp.name, loc, "$(0)");

                // Generate code to pull this EGP with count specified by $(0)
                std::stringstream pullStream;
                CodeStream pull(pullStream);
                backend.genLazyVariableDynamicPull(pull, resolvedType, egp.name, loc, "$(0)");

                // Add substitutions
                groupEnv.add(Type::AllocatePushPullEGP, "push" + egp.name + "ToDevice", pushStream.str());
                groupEnv.add(Type::AllocatePushPullEGP, "pull" + egp.name + "FromDevice", pullStream.str());
            }
        }

        // Add pre and postsynaptic variables along with push and pull functions
        // **TODO** why not pre and post var-references
        addVars<CustomConnectivityUpdatePreVarAdapter>(groupEnv, "$(num_pre)", backend);
        addVars<CustomConnectivityUpdatePostVarAdapter>(groupEnv, "$(num_post)", backend);

        // Pretty print code back to environment
        Transpiler::ErrorHandler errorHandler("Custom connectivity '" + getArchetype().getName() + "' host update code");
        prettyPrintStatements(getArchetype().getHostUpdateCodeTokens(), getTypeContext(), groupEnv, errorHandler);
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
