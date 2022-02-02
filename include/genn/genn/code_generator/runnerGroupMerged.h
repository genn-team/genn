#pragma once

// Standard C++ includes
#include <variant>

// GeNN code generator includes
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// CodeGenerator::RunnerGroupMergedBase
//----------------------------------------------------------------------------
namespace CodeGenerator
{
template<typename G, typename M>
class RunnerGroupMergedBase : public GroupMerged<G>
{
public:
    enum PointerFieldFlags
    {
        POINTER_FIELD_MANUAL_ALLOC  = (1 << 0), //! This (dynamic) pointer field will be manually allocated
        POINTER_FIELD_PUSH_PULL     = (1 << 1), //! Generate push and pull functions for this field
        POINTER_FIELD_GET           = (1 << 2), //! Generate getter function for this field
        POINTER_FIELD_STATE         = (1 << 3), //! Should this field be included in 'state'
        POINTER_FIELD_CONNECTIVITY  = (1 << 4), //! Should this field be included in 'connectivity'
        POINTER_FIELD_PUSH_PULL_GET = POINTER_FIELD_PUSH_PULL | POINTER_FIELD_GET,
    };

    typedef std::function<std::string(const G &)> GetFieldValueFunc;
    typedef std::tuple<VarLocation, VarAccessDuplication, std::string, unsigned int> PointerField;
    typedef std::tuple<std::string, std::string, std::variant<GetFieldValueFunc, PointerField, std::string>> Field;

    
    RunnerGroupMergedBase(size_t index, const std::vector<std::reference_wrapper<const G>> groups)
    :   GroupMerged<G>(index, groups)
    {

    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Get group fields
    const std::vector<Field> &getFields() const{ return m_Fields; }
    
    //! Does this group provide functions to push state
    bool hasPushStateFunction() const{ return anyAccesibleFieldsWithFlags(POINTER_FIELD_PUSH_PULL | POINTER_FIELD_STATE); }
    
    //! Get group fields, sorted into order they will appear in struct
    std::vector<Field> getSortedFields(const BackendBase &backend) const
    {
        // Make a copy of fields and sort so largest come first. This should mean that due
        // to structure packing rules, significant memory is saved and estimate is more precise
        auto sortedFields = m_Fields;
        std::sort(sortedFields.begin(), sortedFields.end(),
                  [&backend](const Field &a, const Field &b)
                  {
                      return (backend.getSize(std::get<0>(a)) > backend.getSize(std::get<0>(b)));
                  });
        return sortedFields;
    }

    //! Generate declaration of struct to hold this merged group
    void generateRunner(const BackendBase &backend, CodeStream &definitionsFunc, 
                        CodeStream &runnerVarDecl,  CodeStream &runnerMergedRunnerStructAlloc, 
                        CodeStream &runnerVarAlloc, CodeStream &runnerVarFree, CodeStream &runnerPushFunc, 
                        CodeStream &runnerPullFunc, CodeStream &runnerGetterFunc, CodeStream &runnerAllocateFunc,
                        CodeStream &runnerFreeFunc, unsigned int batchSize, MemAlloc &memAlloc) const
    {
        const auto sortedFields = getSortedFields(backend);
        runnerVarDecl << "struct Merged" << M::name << "Group" << this->getIndex() << std::endl;
        {
            // Loop through fields and write to structure
            CodeStream::Scope b(runnerVarDecl);
            for(const auto &f : sortedFields) {
                // If it's a pointer
                if(std::holds_alternative<PointerField>(std::get<2>(f))) {
                    // If variable is present on host
                    const auto pointerField = std::get<PointerField>(std::get<2>(f));
                    const VarLocation loc = std::get<0>(pointerField);
                    if(loc & VarLocation::HOST) {
                        // Add field to struct
                        runnerVarDecl << std::get<0>(f) << " " << std::get<1>(f) << ";" << std::endl;

                        // If backend has device prefix, add additional field with prefix and overriden type
                        if(!backend.getHostVarPrefix().empty()) {
                            runnerVarDecl << backend.getMergedGroupFieldHostType(std::get<0>(f)) << " " << backend.getHostVarPrefix() << std::get<1>(f) << ";" << std::endl;
                        }
                    }

                    // If backend has device prefix, add additional field with prefix and overriden type
                    if((loc & VarLocation::DEVICE) && !backend.getDeviceVarPrefix().empty()) {
                        runnerVarDecl << backend.getMergedGroupFieldHostType(std::get<0>(f)) << " " << backend.getDeviceVarPrefix() << std::get<1>(f) << ";" << std::endl;
                    }
                }
                // Otherwise, if it's a scalar field e.g. numbers of neurons
                else if(std::holds_alternative<GetFieldValueFunc>(std::get<2>(f))) {
                    // Add field to struct
                    runnerVarDecl << std::get<0>(f) << " " << std::get<1>(f) << ";" << std::endl;
                }
                // Otherwise, it's a host device scalar
                else {
                    runnerVarDecl << std::get<0>(f) << " " << std::get<1>(f) << ";" << std::endl;
                    if(backend.isDeviceScalarRequired()) {
                        runnerVarDecl << backend.getMergedGroupFieldHostType(std::get<0>(f) + "*") << " " << backend.getDeviceVarPrefix() << std::get<1>(f) << ";" << std::endl;
                    }
                }
            }
            runnerVarDecl << std::endl;
        }
        runnerVarDecl << ";" << std::endl;

        // Declare array of groups
        runnerVarDecl << "Merged" << M::name << "Group" << this->getIndex() << " merged" << M::name << "Group" << this->getIndex() << "[" << this->getGroups().size() << "];" << std::endl;

        // Loop through groups
        for(size_t g = 0; g < this->getGroups().size(); g++) {
            // Loop through fields
            runnerMergedRunnerStructAlloc << "merged" << M::name << "Group" << this->getIndex() << "[" << g << "] = {";
            for(size_t fieldIndex = 0; fieldIndex < sortedFields.size(); fieldIndex++) {
                // If field's a pointer
                const auto &f = sortedFields.at(fieldIndex);
                if(std::holds_alternative<PointerField>(std::get<2>(f))) {
                    // If variable is present on host
                    const auto pointerField = std::get<PointerField>(std::get<2>(f));
                    const VarLocation loc = std::get<0>(pointerField);
                    if(loc & VarLocation::HOST) {
                        runnerMergedRunnerStructAlloc << "nullptr, ";

                        // If backend has device prefix, add additional field with prefix and overriden type
                        if(!backend.getHostVarPrefix().empty()) {
                            // **TODO** some sort of OpenCL initialiser
                            runnerMergedRunnerStructAlloc << "nullptr, ";
                        }
                    }

                    if((loc & VarLocation::DEVICE) && !backend.getDeviceVarPrefix().empty()) {
                        // **TODO** some sort of OpenCL initialiser
                        runnerMergedRunnerStructAlloc << "nullptr, ";
                    }
                }
                // Otherwise, if it's a scalar field e.g. numbers of neurons
                else if(std::holds_alternative<GetFieldValueFunc>(std::get<2>(f))) {
                    const auto getValueFn = std::get<GetFieldValueFunc>(std::get<2>(f));
                    runnerMergedRunnerStructAlloc << getValueFn(this->getGroups().at(g)) << ", ";
                }
                // Otherwise, it's a host device scalar
                else {
                    runnerMergedRunnerStructAlloc << std::get<std::string>(std::get<2>(f)) << ", ";
                    if(backend.isDeviceScalarRequired()) {
                        // **TODO** some sort of OpenCL initialiser
                        runnerMergedRunnerStructAlloc << "nullptr, ";
                    }
                }

            }
            runnerMergedRunnerStructAlloc << "};" << std::endl;
        }

        // Generate push, pull and getter functions
        genFieldFuncs(backend, runnerPushFunc, runnerPullFunc, runnerGetterFunc, 
                     runnerAllocateFunc, runnerFreeFunc, definitionsFunc);
        
        // Generate memory allocation code
        genAllocMem(backend, runnerVarAlloc, batchSize, memAlloc);

        // Generate memory free code
        genFreeMem(backend, runnerVarFree);
    }

protected:
    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void addField(const std::string &type, const std::string &name, 
                  GetFieldValueFunc getFieldValue)
    {
        // Add field to data structure
        assert(!Utils::isTypePointer(type));
        m_Fields.emplace_back(type, name, getFieldValue);
    }

    void addField(const std::string &type, const std::string &name, const std::string &hostValue)
    {
        // Add field to data structure
        assert(!Utils::isTypePointer(type));
        m_Fields.emplace_back(type, name, hostValue);
    }

    void addField(const std::string &type, const std::string &name, VarLocation loc, unsigned int flags,
                  const std::string &count = "", VarAccessDuplication duplication = VarAccessDuplication::DUPLICATE)
    {
        // Add field to data structure
        assert(!Utils::isTypePointer(type));
        m_Fields.emplace_back(type + "*", name, std::make_tuple(loc, duplication, count, flags));
    }

    template<typename E>
    void addEGPs(const Snippet::Base::EGPVec &egps, const std::string &suffix, E getEGPLocFunc)
    {
        // Loop through EGPs
        for(const auto &egp : egps) {
            // If EGP is a pointer, add field
            if(Utils::isTypePointer(egp.type)) {
                addField(Utils::getUnderlyingType(egp.type), egp.name + suffix,
                         getEGPLocFunc(egp.name),
                         POINTER_FIELD_PUSH_PULL_GET);
            }
        }
    }

    void addEGPs(const Snippet::Base::EGPVec &egps, const std::string &suffix = "")
    {
        addEGPs(egps, suffix, [](const std::string &) { return VarLocation::HOST_DEVICE; });
    }
    
    void genPushPointer(const BackendBase &backend, CodeStream &os, const std::string &varName, const std::string &groupIndex) const
    {
        // Push updated pointer to all destinations
        os << "for(unsigned int m = start" << varName << M::name << "Group" << this->getIndex() << "[" << groupIndex <<"]; m < end" << varName << M::name << "Group" << this->getIndex() << "[" << groupIndex << "]; m++)";
        {
            CodeStream::Scope b(os);
            os << "update" << varName << M::name << "Group" << this->getIndex() << "MergedGroups[m](group->" << backend.getDeviceVarPrefix() << varName << ");" << std::endl;
        }
    }
    
    //! Return true if there are any pointer fields with the specified flags set
    bool anyAccesibleFieldsWithFlags(unsigned int flags) const
    {
        return std::any_of(getFields().cbegin(), getFields().cend(),
                           [flags](const Field &f)
                           {
                               if(std::holds_alternative<PointerField>(std::get<2>(f))) {
                                   const auto pointerField = std::get<PointerField>(std::get<2>(f));
                                   const auto loc = std::get<0>(pointerField);
                                   const unsigned int fieldFlags = std::get<3>(pointerField);
                                   return (((fieldFlags & flags) == flags)
                                           && (loc & VarLocation::HOST) && (loc & VarLocation::DEVICE));
                               }
                               else {
                                   return false;
                               }
                           });
    }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------    
    void genFieldGroupPushFuncs(const std::vector<std::string> &fieldNames, const std::string &prefix,
                                CodeStream &runnerPushFunc, CodeStream &definitions) const
    {
        // If there are any fields in list
        if(!fieldNames.empty()) {
            // Define group push functions
            definitions << "EXPORT_FUNC void push" << prefix << "ToDevice(unsigned int i, bool uninitialisedOnly = false);" << std::endl;

            // Implement push function
            runnerPushFunc << "void push" << prefix << "ToDevice(unsigned int i, bool uninitialisedOnly)";
            {
                CodeStream::Scope a(runnerPushFunc);
                for(const auto &s : fieldNames) {
                    runnerPushFunc << "push" << s << "ToDevice(i, uninitialisedOnly);" << std::endl;
                }
            }
            runnerPushFunc << std::endl;
        }
    }
    
    void genFieldFuncs(const BackendBase &backend, CodeStream &runnerPushFunc, CodeStream &runnerPullFunc,
                       CodeStream &runnerGetterFunc, CodeStream &runnerAllocateFunc, CodeStream &runnerFreeFunc, CodeStream &definitions) const
    {
        // Loop through fields
        std::vector<std::string> stateFields;
        std::vector<std::string> connectivityFields;
        const std::string mergedGroupName = M::name + "Group" + std::to_string(this->getIndex());
        for(const auto &f : m_Fields) {
            // If this is pointer field
            if(std::holds_alternative<PointerField>(std::get<2>(f))) {
                const auto pointerField = std::get<PointerField>(std::get<2>(f));

                // If field should have a push and pull function generated and it can be pushed and pulled
                // **TODO** auto initialise and handle unitialisedOnly in backend
                const auto loc = std::get<0>(pointerField);
                const std::string &fieldCount = std::get<2>(pointerField);
                const unsigned int flags = std::get<3>(pointerField);
                const std::string name = std::get<1>(f) + mergedGroupName;
                const std::string count = fieldCount.empty() ? "count" : fieldCount;
                const std::string group = "merged" + mergedGroupName + "[i]";
                if((loc & VarLocation::HOST) && (loc & VarLocation::DEVICE))
                {
                    if(flags & POINTER_FIELD_PUSH_PULL) {
                        if(fieldCount.empty()) {
                            definitions << "EXPORT_FUNC void push" << name << "ToDevice(unsigned int i, unsigned int count);" << std::endl;
                            runnerPushFunc << "void push" << name << "ToDevice(unsigned int i, unsigned int count)";
                        }
                        else {
                            definitions << "EXPORT_FUNC void push" << name << "ToDevice(unsigned int i, bool uninitialisedOnly = false);" << std::endl;
                            runnerPushFunc << "void push" << name << "ToDevice(unsigned int i, bool uninitialisedOnly)";
                        }
                        {
                            CodeStream::Scope a(runnerPushFunc);
                            runnerPushFunc << "auto *group = &" << group << ";" << std::endl;
                            backend.genFieldPush(runnerPushFunc, std::get<0>(f), std::get<1>(f), loc, count);
                        }
                        runnerPushFunc << std::endl;

                        if(fieldCount.empty()) {
                            definitions << "EXPORT_FUNC void pull" << name << "FromDevice(unsigned int i, unsigned int count);" << std::endl;
                            runnerPullFunc << "void pull" << name << "FromDevice(unsigned int i, unsigned int count)";
                        }
                        else {
                            definitions << "EXPORT_FUNC void pull" << name << "FromDevice(unsigned int i);" << std::endl;
                            runnerPullFunc << "void pull" << name << "FromDevice(unsigned int i)";
                        }
                        {
                            CodeStream::Scope a(runnerPullFunc);
                            runnerPullFunc << "auto *group = &" << group << ";" << std::endl;
                            backend.genFieldPull(runnerPullFunc, std::get<0>(f), std::get<1>(f), loc, count);
                        }
                        runnerPullFunc << std::endl;
                        
                        // If this isn't a dynamic file
                        if(!fieldCount.empty()) {
                            // If state flag is set, add to vector of state fields
                            if(flags & POINTER_FIELD_STATE) {
                                stateFields.push_back(name);
                            }
                            
                            // If connectivity flag is set, add to vector of connectivity fields
                            if(flags & POINTER_FIELD_CONNECTIVITY) {
                                connectivityFields.push_back(name);
                            }
                        }
                    }
                    
                    if(flags & POINTER_FIELD_GET) {
                        definitions << "EXPORT_FUNC " << std::get<0>(f) << " get" << name << "(unsigned int i);" << std::endl;
                        runnerGetterFunc << std::get<0>(f) << " get" << name << "(unsigned int i)";
                        {
                            CodeStream::Scope a(runnerGetterFunc);
                            runnerGetterFunc << "return " << group << "." << std::get<1>(f) << ";" << std::endl;
                        }
                    }
                }
                
                // If field is dynamically allocated and doesn't have manual allocation flag set
                if(((flags & POINTER_FIELD_MANUAL_ALLOC) == 0) && fieldCount.empty()) {
                    // Define allocate and free functions
                    definitions << "EXPORT_FUNC void allocate" << name << "(unsigned int i, unsigned int count);" << std::endl;
                    definitions << "EXPORT_FUNC void free" << name << "(unsigned int i);" << std::endl;
                    
                    // Implement allocate function
                    runnerAllocateFunc << "void allocate" << name << "(unsigned int i, unsigned int count)";
                    {
                        CodeStream::Scope a(runnerAllocateFunc);
                        runnerAllocateFunc << "auto *group = &" << group << ";" << std::endl;
                        backend.genFieldAllocation(runnerAllocateFunc, std::get<0>(f), std::get<1>(f), loc, count);
                        
                        // Generate code to push updated pointer to all destinations
                        genPushPointer(backend, runnerAllocateFunc, std::get<1>(f), "i");
                    }
                    
                    // Allocate free function
                    runnerFreeFunc << "void free" << name << "(unsigned int i)";
                    {
                        CodeStream::Scope a(runnerFreeFunc);
                        runnerFreeFunc << "auto *group = &" << group << ";" << std::endl;
                        backend.genFieldFree(runnerFreeFunc, std::get<1>(f), loc);
                    }
                }
            }
        }
        
        // Generate push and pull functions for state if required
        genFieldGroupPushFuncs(stateFields, mergedGroupName + "State", runnerPushFunc, definitions);
        genFieldGroupPushFuncs(connectivityFields, mergedGroupName + "Connectivity", runnerPushFunc, definitions);
    }

    void genAllocMem(const BackendBase &backend, CodeStream &runner, unsigned int batchSize, MemAlloc &memAlloc) const
    {
        CodeStream::Scope b(runner);
        runner << "// merged group " << this->getIndex() << std::endl;
        runner << "for(unsigned int g = 0; g < " << this->getGroups().size() << "; g++)";
        {
            CodeStream::Scope b(runner);

            // Get reference to group
            runner << "auto *group = &merged" << M::name << "Group" << this->getIndex() << "[g]; " << std::endl;

            // Loop through fields
            for(const auto &f : m_Fields) {
                // If this is pointer field
                if(std::holds_alternative<PointerField>(std::get<2>(f))) {
                    const auto pointerField = std::get<PointerField>(std::get<2>(f));

                    // If field has a count i.e. it's not allocated dynamically
                    const std::string &count = std::get<2>(pointerField);
                    if(!count.empty()) {
                        if(std::get<1>(pointerField) == VarAccessDuplication::SHARED) {
                            backend.genFieldAllocation(runner, std::get<0>(f), std::get<1>(f), std::get<0>(pointerField), count);
                        }
                        else {
                            backend.genFieldAllocation(runner, std::get<0>(f), std::get<1>(f), std::get<0>(pointerField),
                                                       std::to_string(batchSize) + " * " + count);
                        }
                    }
                }
                // Otherwise, if this is a scalar field and device scalars are required, allocate one entry array on device
                else if(std::holds_alternative<std::string>(std::get<2>(f))) {
                    if(backend.isDeviceScalarRequired()) {
                        backend.genFieldAllocation(runner, std::get<0>(f) + "*", std::get<1>(f), VarLocation::DEVICE, "1");
                    }
                }
            }
        }
    }

    void genFreeMem(const BackendBase &backend, CodeStream &runner) const
    {
        CodeStream::Scope b(runner);
        runner << "// merged group " << this->getIndex() << std::endl;
        runner << "for(unsigned int g = 0; g < " << this->getGroups().size() << "; g++)";
        {
            CodeStream::Scope b(runner);

            // Get reference to group
            runner << "auto *group = &merged" << M::name << "Group" << this->getIndex() << "[g]; " << std::endl;

            // Loop through fields
            for(const auto &f : m_Fields) {
                // If this is pointer field
                if(std::holds_alternative<PointerField>(std::get<2>(f))) {
                    const auto pointerField = std::get<PointerField>(std::get<2>(f));

                    // If field has a count i.e. it's not allocated dynamically
                    // **TODO** we could insert a NULL check here and free these at the same time
                    const std::string &count = std::get<2>(pointerField);
                    if(!count.empty()) {
                        backend.genFieldFree(runner, std::get<1>(f), std::get<0>(pointerField));
                    }
                }
                // Otherwise, if this is a scalar field and device scalars are required, free device array
                else if(std::holds_alternative<std::string>(std::get<2>(f))) {
                    if(backend.isDeviceScalarRequired()) {
                         backend.genFieldFree(runner, std::get<1>(f), VarLocation::DEVICE);
                    }
                }
            }
        }
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<Field> m_Fields;
};

//----------------------------------------------------------------------------
// CodeGenerator::NeuronRunnerGroupMerged
//----------------------------------------------------------------------------
class NeuronRunnerGroupMerged : public RunnerGroupMergedBase<NeuronGroupInternal, NeuronRunnerGroupMerged>
{
public:
    NeuronRunnerGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                            const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void genRecordingBufferAlloc(const BackendBase &backend, CodeStream &runner, unsigned int batchSize) const;
    void genRecordingBufferPull(const BackendBase &backend, CodeStream &runner, unsigned int batchSize) const;

    //! Generate code to update host spike queue pointer
    void genSpikeQueuePtrUpdate(CodeStream &os) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseRunnerGroupMerged
//----------------------------------------------------------------------------
class SynapseRunnerGroupMerged : public RunnerGroupMergedBase<SynapseGroupInternal, SynapseRunnerGroupMerged>
{
public:
    SynapseRunnerGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                             const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Does this group provide functions to push connectivity
    bool hasPushConnectivityFunction() const{ return anyAccesibleFieldsWithFlags(POINTER_FIELD_PUSH_PULL | POINTER_FIELD_CONNECTIVITY); }
    
    //! Generate code to update host dendritic delay pointer
    void genDenDelayPtrUpdate(CodeStream &os) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// CodeGenerator::CurrentSourceRunnerGroupMerged
//----------------------------------------------------------------------------
class CurrentSourceRunnerGroupMerged : public RunnerGroupMergedBase<CurrentSourceInternal, CurrentSourceRunnerGroupMerged>
{
public:
    CurrentSourceRunnerGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                   const std::vector<std::reference_wrapper<const CurrentSourceInternal>> &groups);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateRunnerGroupMergedBase
//----------------------------------------------------------------------------
template<typename G, typename M>
class CustomUpdateRunnerGroupMergedBase : public RunnerGroupMergedBase<G, M>
{
public:
    CustomUpdateRunnerGroupMergedBase(size_t index, const std::vector<std::reference_wrapper<const G>> &groups)
    :   RunnerGroupMergedBase<G, M>(index, groups)
    {
        // Add extra global parmeters
        // **TODO** missing location
        this->addEGPs(this->getArchetype().getCustomUpdateModel()->getExtraGlobalParams(), "");

        // Loop through variables
        const auto &varInit = this->getArchetype().getVarInitialisers();
        for(const auto &var : this->getArchetype().getCustomUpdateModel()->getVars()) {
            this->addField(var.type, var.name, this->getArchetype().getVarLocation(var.name),
                           this->POINTER_FIELD_PUSH_PULL_GET | this->POINTER_FIELD_STATE, 
                           "group->size", getVarAccessDuplication(var.access));
            this->addEGPs(varInit.at(var.name).getSnippet()->getExtraGlobalParams(), var.name);
        }
    }
};

//----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateRunnerGroupMerged
//----------------------------------------------------------------------------
class CustomUpdateRunnerGroupMerged : public CustomUpdateRunnerGroupMergedBase<CustomUpdateInternal, CustomUpdateRunnerGroupMerged>
{
public:
    CustomUpdateRunnerGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                  const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateWURunnerGroupMerged
//----------------------------------------------------------------------------
class CustomUpdateWURunnerGroupMerged : public CustomUpdateRunnerGroupMergedBase<CustomUpdateWUInternal, CustomUpdateWURunnerGroupMerged>
{
public:
    CustomUpdateWURunnerGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                    const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

}   // namespace CodeGenerator
