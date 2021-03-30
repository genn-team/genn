#include "code_generator/backendBase.h"

// GeNN includes
#include "gennUtils.h"
#include "logging.h"

// GeNN code generator includes
#include "code_generator/groupMerged.h"

// Macro for simplifying defining type sizes
#define TYPE(T) {#T, sizeof(T)}

//--------------------------------------------------------------------------
// CodeGenerator::BackendBase
//--------------------------------------------------------------------------
CodeGenerator::BackendBase::BackendBase(const std::string &scalarType, const PreferencesBase &preferences)
:   m_PointerBytes(sizeof(char*)), m_TypeBytes{{TYPE(char), TYPE(wchar_t), TYPE(signed char), TYPE(short),
    TYPE(signed short), TYPE(short int), TYPE(signed short int), TYPE(int), TYPE(signed int), TYPE(long),
    TYPE(signed long), TYPE(long int), TYPE(signed long int), TYPE(long long), TYPE(signed long long), TYPE(long long int),
    TYPE(signed long long int), TYPE(unsigned char), TYPE(unsigned short), TYPE(unsigned short int), TYPE(unsigned),
    TYPE(unsigned int), TYPE(unsigned long), TYPE(unsigned long int), TYPE(unsigned long long),
    TYPE(unsigned long long int), TYPE(float), TYPE(double), TYPE(long double), TYPE(bool), TYPE(intmax_t),
    TYPE(uintmax_t), TYPE(int8_t), TYPE(uint8_t), TYPE(int16_t), TYPE(uint16_t), TYPE(int32_t), TYPE(uint32_t),
    TYPE(int64_t), TYPE(uint64_t), TYPE(int_least8_t), TYPE(uint_least8_t), TYPE(int_least16_t), TYPE(uint_least16_t),
    TYPE(int_least32_t), TYPE(uint_least32_t), TYPE(int_least64_t), TYPE(uint_least64_t), TYPE(int_fast8_t),
    TYPE(uint_fast8_t), TYPE(int_fast16_t), TYPE(uint_fast16_t), TYPE(int_fast32_t), TYPE(uint_fast32_t),
    TYPE(int_fast64_t), TYPE(uint_fast64_t)}}, m_Preferences(preferences)
{
    // Add scalar type
    addType("scalar", (scalarType == "float") ? sizeof(float) : sizeof(double));
}
//--------------------------------------------------------------------------
size_t CodeGenerator::BackendBase::getSize(const std::string &type) const
{
     // If type is a pointer, any pointer should have the same type
    if(Utils::isTypePointer(type)) {
        return m_PointerBytes;
    }
    // Otherwise
    else {
        // If type isn't found in dictionary, give a warning and return 0
        const auto typeSize = m_TypeBytes.find(type);
        if(typeSize == m_TypeBytes.cend()) {
            LOGW_CODE_GEN << "Unable to estimate size of type '" << type << "'";
            return 0;
        }
        // Otherwise, return its size
        else {
            return typeSize->second;
        }
    }
}
//--------------------------------------------------------------------------
bool CodeGenerator::BackendBase::areSixtyFourBitSynapseIndicesRequired(const SynapseGroupMergedBase &sg) const
{
    // Loop through merged groups and calculate maximum number of synapses
    size_t maxSynapses = 0;
    for(const auto &g : sg.getGroups()) {
        const size_t numSynapses = (size_t)g.get().getSrcNeuronGroup()->getNumNeurons() * (size_t)getSynapticMatrixRowStride(g.get());
        maxSynapses = std::max(maxSynapses, numSynapses);
    }

    // Return true if any high bits are set
    return ((maxSynapses & 0xFFFFFFFF00000000ULL) != 0);
}
//-----------------------------------------------------------------------
void CodeGenerator::BackendBase::genNeuronIndexCalculation(CodeStream &os, const NeuronUpdateGroupMerged &ng, unsigned int batchSize) const
{
    // If batching is enabled, calculate batch offset
    if(batchSize > 1) {
        os << "const unsigned int batchOffset = group->numNeurons * batch;" << std::endl;
    }
            
    // If axonal delays are required
    if(ng.getArchetype().isDelayRequired()) {
        // We should READ from delay slot before spkQuePtr
        os << "const unsigned int readDelayOffset = (((*group->spkQuePtr + " << (ng.getArchetype().getNumDelaySlots() - 1) << ") % " << ng.getArchetype().getNumDelaySlots() << ") * group->numNeurons);" << std::endl;

        // And we should WRITE to delay slot pointed to be spkQuePtr
        os << "const unsigned int writeDelayOffset = (*group->spkQuePtr * group->numNeurons);" << std::endl;

        // If batching is also enabled
        if(batchSize > 1) {
            // Calculate current batch offset
            os << "const unsigned int batchDelayOffset = batchOffset * " << ng.getArchetype().getNumDelaySlots() << ";" << std::endl;

            // Calculate further offsets to include delay and batch
            os << "const unsigned int readBatchDelayOffset = readDelayOffset + batchDelayOffset;" << std::endl;
            os << "const unsigned int writeBatchDelayOffset = writeDelayOffset + batchDelayOffset;" << std::endl;
        }
    }
}
//-----------------------------------------------------------------------
void CodeGenerator::BackendBase::genSynapseIndexCalculation(CodeStream &os, const SynapseGroupMergedBase &sg, unsigned int batchSize) const
{
     // If batching is enabled
    if(batchSize > 1) {
        // Calculate batch offsets into pre and postsynaptic populations
        os << "const unsigned int preBatchOffset = group->numSrcNeurons * batch;" << std::endl;
        os << "const unsigned int postBatchOffset = group->numTrgNeurons * batch;" << std::endl;

        // Calculate batch offsets into synapse arrays, using 64-bit arithmetic if necessary
        if(areSixtyFourBitSynapseIndicesRequired(sg)) {
            os << "const uint64_t synBatchOffset = (uint64_t)preBatchOffset * (uint64_t)group->rowStride;" << std::endl;
        }
        else {
            os << "const unsigned int synBatchOffset = preBatchOffset * group->rowStride;" << std::endl;
        }
    }

    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
    if(sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
        const unsigned int numDelaySteps = sg.getArchetype().getDelaySteps();
        const unsigned int numSrcDelaySlots = sg.getArchetype().getSrcNeuronGroup()->getNumDelaySlots();

        os << "const unsigned int preDelaySlot = ";
        if(numDelaySteps == 0) {
            os << "*group->srcSpkQuePtr;" << std::endl;
        }
        else {
            os << "(*group->srcSpkQuePtr + " << (numSrcDelaySlots - numDelaySteps) << ") % " << numSrcDelaySlots <<  ";" << std::endl;
        }
        os << "const unsigned int preDelayOffset = preDelaySlot * group->numSrcNeurons;" << std::endl;

        if(batchSize > 1) {
            os << "const unsigned int preBatchDelaySlot = preDelaySlot + (batch * " << numSrcDelaySlots << ");" << std::endl;
            os << "const unsigned int preBatchDelayOffset = preDelayOffset + (preBatchOffset * " << numSrcDelaySlots << ");" << std::endl;
        }

        if(sg.getArchetype().getWUModel()->isPrevPreSpikeTimeRequired() || sg.getArchetype().getWUModel()->isPrevPreSpikeEventTimeRequired()) {
            os << "const unsigned int prePrevSpikeTimeDelayOffset = " << "((*group->srcSpkQuePtr + " << (numSrcDelaySlots - numDelaySteps - 1) << ") % " << numSrcDelaySlots << ")" << " * group->numSrcNeurons;" << std::endl;

            if(batchSize > 1) {
                os << "const unsigned int prePrevSpikeTimeBatchDelayOffset = prePrevSpikeTimeDelayOffset + (preBatchOffset * " << numSrcDelaySlots << ");" << std::endl;
            }
        }
    }

    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
    if(sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
        const unsigned int numBackPropDelaySteps = sg.getArchetype().getBackPropDelaySteps();
        const unsigned int numTrgDelaySlots = sg.getArchetype().getTrgNeuronGroup()->getNumDelaySlots();

        os << "const unsigned int postDelaySlot = ";
        if(numBackPropDelaySteps == 0) {
            os << "*group->trgSpkQuePtr;" << std::endl;
        }
        else {
            os << "(*group->trgSpkQuePtr + " << (numTrgDelaySlots - numBackPropDelaySteps) << ") % " << numTrgDelaySlots << ";" << std::endl;
        }
        os << "const unsigned int postDelayOffset = postDelaySlot * group->numTrgNeurons;" << std::endl;

        if(batchSize > 1) {
            os << "const unsigned int postBatchDelaySlot = postDelaySlot + (batch * " << numTrgDelaySlots << ");" << std::endl;
            os << "const unsigned int postBatchDelayOffset = postDelayOffset + (postBatchOffset * " << numTrgDelaySlots << ");" << std::endl;
        }

        if(sg.getArchetype().getWUModel()->isPrevPostSpikeTimeRequired()) {
            os << "const unsigned int postPrevSpikeTimeDelayOffset = " << "((*group->trgSpkQuePtr + " << (numTrgDelaySlots - numBackPropDelaySteps - 1) << ") % " << numTrgDelaySlots << ")" << " * group->numTrgNeurons;" << std::endl;
            
            if(batchSize > 1) {
                os << "const unsigned int postPrevSpikeTimeBatchDelayOffset = postPrevSpikeTimeDelayOffset + (postBatchOffset * " << numTrgDelaySlots << ");" << std::endl;
            }

        }
    }
}
//-----------------------------------------------------------------------
void CodeGenerator::BackendBase::genCustomUpdateIndexCalculation(CodeStream &os, const CustomUpdateGroupMerged &cu) const
{
    // If batching is enabled, calculate batch offset
    if(cu.getArchetype().isBatched()) {
        os << "const unsigned int batchOffset = group->size * batch;" << std::endl;
    }
            
    // If axonal delays are required
    if(cu.getArchetype().getDelayNeuronGroup() != nullptr) {
        // We should read from delay slot pointed to be spkQuePtr
        os << "const unsigned int delayOffset = (*group->spkQuePtr * group->size);" << std::endl;

        // If batching is also enabled, calculate offset including delay and batch
        if(cu.getArchetype().isBatched()) {
            os << "const unsigned int batchDelayOffset = delayOffset + (batchOffset * " << cu.getArchetype().getDelayNeuronGroup()->getNumDelaySlots() << ");" << std::endl;
        }
    }
}