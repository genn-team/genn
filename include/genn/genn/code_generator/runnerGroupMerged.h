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
        POINTER_FIELD_PUSH_PULL_GET = (1 << 1), //! Generate push, pull and getter functions for this field
    };

    typedef std::function<std::string(const G &)> GetFieldValueFunc;
    typedef std::tuple<VarLocation, VarAccessDuplication, std::string, unsigned int> PointerField;
    typedef std::tuple<std::string, std::string, std::variant<GetFieldValueFunc, PointerField>> Field;

    
    RunnerGroupMergedBase(size_t index, const std::string &precision, const std::vector<std::reference_wrapper<const G>> groups, const BackendBase &backend)
    :   GroupMerged<G>(index, groups)
    {

    }

    void genPushPullGet(const BackendBase &backend, CodeStream &runner, CodeStream &definitions, const ModelSpecMerged &modelMerged) const
    {
        const unsigned int batchSize = modelMerged.getModel().getBatchSize();
        
        // Loop through fields
        for(const auto &f : getFields()) {
            // If this is pointer field
            if(std::holds_alternative<PointerField>(std::get<2>(f))) {
                const auto pointerField = std::get<PointerField>(std::get<2>(f));

                // If field should have a push and pull function generated and it can be pushed and pulled
                // **TODO** auto initialise and handle unitialisedOnly in backend
                const auto loc = std::get<0>(pointerField);
                const std::string &fieldCount = std::get<2>(pointerField);
                const unsigned int flags = std::get<3>(pointerField);
                if((flags & POINTER_FIELD_PUSH_PULL_GET) && (loc & VarLocation::HOST) &&
                   (loc & VarLocation::DEVICE)) 
                {
                    const std::string name = std::get<0>(f) + T::name + "Group" + std::to_string(getIndex());
                    const std::string count = fieldCount.empty() ? "count" : fieldCount;
                    const std::string group = "merged" + T::name + "Group" + std::to_string(getIndex()) + "[group]";
                    if(fieldCount.empty()) {
                        definitions << "EXPORT_FUNC void push" << name << "ToDevice(unsigned int group, unsigned int count;" << std::endl;
                        runner << "void push" << name << "ToDevice(unsigned int group, unsigned int count)";
                    }
                    else {
                        definitions << "EXPORT_FUNC void push" << name << "ToDevice(unsigned int group, bool uninitialisedOnly = false);" << std::endl;
                        runner << "void push" << name << "ToDevice(unsigned int group, bool uninitialisedOnly)";
                    }
                    {
                        CodeStream::Scope a(runner);
                        runner << "auto *group = &" << group ";" << std::endl;
                        backend.genExtraGlobalParamPush(push, std::get<0>(f), std::get<1>(f),
                                                        loc, count, "group->");
                    }
                    runner << std::endl;
                    
                    if(fieldCount.empty()) {
                        definitions << "EXPORT_FUNC void pull" << name << "FromDevice(unsigned int group, unsigned int count);" << std::endl;
                        runner << "void pull" << name << "FromDevice(unsigned int group, unsigned int count)";
                    }
                    else {
                        definitions << "EXPORT_FUNC void pull" << name << "FromDevice(unsigned int group);" << std::endl;
                        runner << "void pull" << name << "FromDevice(unsigned int group)";
                    }
                    {
                        CodeStream::Scope a(runner);
                        runner << "auto *group = &" << group ";" << std::endl;
                        backend.genExtraGlobalParamPull(push, std::get<0>(f), std::get<1>(f),
                                                        loc, count, "group->");
                    }
                    runner << std::endl;
                    
                    definitions << "EXPORT_FUNC " << std::get<0>(f) << "get" << name << "(unsigned int group);" << std::endl;
                    runner << std::get<0>(f) << "get" << name << "(unsigned int group)";
                    {
                        CodeStream::Scope a(runner);
                        runner << "return " << group << "." << std::get<1>(f) << ";" << std::endl;
                    }

                }
            }
        }
    }

    void genAllocMem(const BackendBase &backend, CodeStream &runner, const ModelSpecMerged &modelMerged, const MemAlloc &memAlloc) const
    {
        const unsigned int batchSize = modelMerged.getModel().getBatchSize();

        CodeStream::Scope b(runner);
        runner << "// merged group " << getIndex() << std::endl;
        runner << "for(unsigned int g = 0; g < " << getGroups().size() << "; g++)";
        {
            CodeStream::Scope b(runner);

            // Get reference to group
            runner << "auto *group = &merged" << T::name << "Group" << getIndex() << "[g]; " << std::endl;

            // Loop through fields
            for(const auto &f : getFields()) {
                // If this is pointer field
                if(std::holds_alternative<PointerField>(std::get<2>(f))) {
                    const auto pointerField = std::get<PointerField>(std::get<2>(f));

                    // If field has a count i.e. it's not allocated dynamically
                    const std::string &count = std::get<2>(pointerField);
                    if(!count.empty()) {
                        if(std::get<1>(pointerField) == VarAccessDuplication::SHARED) {
                            backend.genExtraGlobalParamAllocation(runner, std::get<0>(f), std::get<1>(f), std::get<0>(pointerField),
                                                                  count, "group->");
                        }
                        else {
                            backend.genExtraGlobalParamAllocation(runner, std::get<0>(f), std::get<1>(f), std::get<0>(pointerField),
                                                                  std::to_string(numCopies) + " * " + count, "group->");
                        }
                    }
                }
            }
        }
    }

    void genFreeMem(const BackendBase &backend, CodeStream &runner, const ModelSpecMerged &modelMerged, const MemAlloc &memAlloc) const
    {
        const unsigned int batchSize = modelMerged.getModel().getBatchSize();

        CodeStream::Scope b(runner);
        runner << "// merged group " << getIndex() << std::endl;
        runner << "for(unsigned int g = 0; g < " << getGroups().size() << "; g++)";
        {
            CodeStream::Scope b(runner);

            // Get reference to group
            runner << "auto *group = &merged" << T::name << "Group" << getIndex() << "[g]; " << std::endl;

            // Loop through fields
            for(const auto &f : getFields()) {
                // If this is pointer field
                if(std::holds_alternative<PointerField>(std::get<2>(f))) {
                    const auto pointerField = std::get<PointerField>(std::get<2>(f));

                    // If field has a count i.e. it's not allocated dynamically
                    // **TODO** we could insert a NULL check here and free these at the same time
                    const std::string &count = std::get<2>(pointerField);
                    if(!count.empty()) {
                        backend.genVariableFree(runner, std::get<1>(f), std::get<0>(pointerField));
                    }
                }
            }
        }
    }


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
    void generateStruct(CodeStream &os, const BackendBase &backend) const
    {
        os << "struct Merged" << T::name << "Group" << getIndex() << std::endl;
        {
            // Loop through fields and write to structure
            CodeStream::Scope b(os);
            const auto sortedFields = getSortedFields(backend);
            for(const auto &f : sortedFields) {
                // If it's a pointer
                if(std::holds_alternative<PointerField>(std::get<2>(f))) {
                    const VarLocation loc = std::get<0>(pointerField);

                    // If variable is present on host
                    if(loc & VarLocation::HOST) {
                        // Add field to struct
                        os << std::get<0>(f) << " " << std::get<1>(f) << ";" << std::endl;

                        // If backend has device prefix, add additional field with prefix and overriden type
                        if(!backend.getHostVarPrefix().empty()) {
                            os << backend.getMergedGroupFieldHostType(std::get<0>(f)) << " " << backend.getHostVarPrefix() << std::get<1>(f) << ";" << std::endl;
                        }
                    }

                    // If backend has device prefix, add additional field with prefix and overriden type
                    if((loc & VarLocation::DEVICE) && !backend.getDeviceVarPrefix().empty()) {
                        os << backend.getMergedGroupFieldHostType(std::get<0>(f)) << " " << backend.getDeviceVarPrefix() << std::get<1>(f) << ";" << std::endl;
                    }
                }
                else {
                    // Add field to struct
                    os << std::get<0>(f) << " " << std::get<1>(f) << ";" << std::endl;
                }
            }
            os << std::endl;
        }

        os << ";" << std::endl;
    }

protected:
    //------------------------------------------------------------------------
    // Protected API
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
    void genRecordingBufferAlloc(const BackendBase &backend, CodeStream &runner, const ModelSpecMerged &modelMerged) const;
    void genRecordingBufferPull(const BackendBase &backend, CodeStream &runner, const ModelSpecMerged &modelMerged) const;

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
    void genConnectivityPushPull(const BackendBase &backend, CodeStream &runner, const ModelSpecMerged &modelMerged) const;
};
}
