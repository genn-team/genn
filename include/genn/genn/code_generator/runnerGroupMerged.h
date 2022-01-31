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
        POINTER_FIELD_PUSH_PULL_GET = POINTER_FIELD_PUSH_PULL | POINTER_FIELD_GET,
    };

    typedef std::function<std::string(const G &)> GetFieldValueFunc;
    typedef std::tuple<VarLocation, VarAccessDuplication, std::string, unsigned int> PointerField;
    typedef std::tuple<std::string, std::string, std::variant<GetFieldValueFunc, PointerField>> Field;

    
    RunnerGroupMergedBase(size_t index, const std::vector<std::reference_wrapper<const G>> groups)
    :   GroupMerged<G>(index, groups)
    {

    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Get group fields
    const std::vector<Field> &getFields() const{ return m_Fields; }

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
                        CodeStream &runnerPullFunc, CodeStream &runnerGetterFunc, 
                        unsigned int batchSize, MemAlloc &memAlloc) const
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
                else {
                    // Add field to struct
                    runnerVarDecl << std::get<0>(f) << " " << std::get<1>(f) << ";" << std::endl;
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
                // Otherwise, initialise with value
                else {
                    const auto getValueFn = std::get<GetFieldValueFunc>(std::get<2>(f));
                    runnerMergedRunnerStructAlloc << getValueFn(this->getGroups().at(g)) << ", ";
                }

            }
            runnerMergedRunnerStructAlloc << "};" << std::endl;
        }

        // Generate push, pull and getter functions
        genPushPullGet(backend, runnerPushFunc, runnerPullFunc, runnerGetterFunc, definitionsFunc);
        
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
                addField(egp.type, egp.name + suffix,
                         getEGPLocFunc(egp.name),
                         POINTER_FIELD_PUSH_PULL_GET);
            }
        }
    }

    void addEGPs(const Snippet::Base::EGPVec &egps, const std::string &suffix)
    {
        addEGPs(egps, suffix, [](const std::string &) { return VarLocation::HOST_DEVICE; });
    }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void genPushPullGet(const BackendBase &backend, CodeStream &runnerPushFunc, CodeStream &runnerPullFunc,
                        CodeStream &runnerGetterFunc, CodeStream &definitions) const
    {
        // Loop through fields
        for(const auto &f : m_Fields) {
            // If this is pointer field
            if(std::holds_alternative<PointerField>(std::get<2>(f))) {
                const auto pointerField = std::get<PointerField>(std::get<2>(f));

                // If field should have a push and pull function generated and it can be pushed and pulled
                // **TODO** auto initialise and handle unitialisedOnly in backend
                const auto loc = std::get<0>(pointerField);
                const std::string &fieldCount = std::get<2>(pointerField);
                const unsigned int flags = std::get<3>(pointerField);
                if((loc & VarLocation::HOST) && (loc & VarLocation::DEVICE))
                {
                    const std::string name = std::get<1>(f) + M::name + "Group" + std::to_string(this->getIndex());
                    const std::string count = fieldCount.empty() ? "count" : fieldCount;
                    const std::string group = "merged" + M::name + "Group" + std::to_string(this->getIndex()) + "[i]";

                    if(flags & POINTER_FIELD_PUSH_PULL) {
                        if(fieldCount.empty()) {
                            definitions << "EXPORT_FUNC void push" << name << "ToDevice(unsigned int i, unsigned int count;" << std::endl;
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
            }
        }
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
                           this->POINTER_FIELD_PUSH_PULL_GET, "group->size", getVarAccessDuplication(var.access));
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
