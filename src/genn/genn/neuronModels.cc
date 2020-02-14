#include "neuronModels.h"

// Standard C++ includes
#include <sstream>

// Code generator includes
#include "code_generator/codeStream.h"

// Implement models
IMPLEMENT_MODEL(NeuronModels::RulkovMap);
IMPLEMENT_MODEL(NeuronModels::Izhikevich);
IMPLEMENT_MODEL(NeuronModels::IzhikevichVariable);
IMPLEMENT_MODEL(NeuronModels::LIF);
IMPLEMENT_MODEL(NeuronModels::SpikeSource);
IMPLEMENT_MODEL(NeuronModels::SpikeSourceArray);
IMPLEMENT_MODEL(NeuronModels::Poisson);
IMPLEMENT_MODEL(NeuronModels::PoissonNew);
IMPLEMENT_MODEL(NeuronModels::TraubMiles);
IMPLEMENT_MODEL(NeuronModels::TraubMilesFast);
IMPLEMENT_MODEL(NeuronModels::TraubMilesAlt);
IMPLEMENT_MODEL(NeuronModels::TraubMilesNStep);

//----------------------------------------------------------------------------
// NeuronModels::Base
//----------------------------------------------------------------------------
bool NeuronModels::Base::canBeMerged(const Base *other) const
{
    return (Models::Base::canBeMerged(other)
            && (getSimCode() == other->getSimCode())
            && (getThresholdConditionCode() == other->getThresholdConditionCode())
            && (getResetCode() == other->getResetCode())
            && (getSupportCode() == other->getSupportCode())
            && (isAutoRefractoryRequired() == other->isAutoRefractoryRequired())
            && (getAdditionalInputVars() == other->getAdditionalInputVars()));
}

//----------------------------------------------------------------------------
// NeuronModels::RK4Base
//----------------------------------------------------------------------------
std::string NeuronModels::RK4Base::getSimCode() const
{
    std::stringstream simCodeStream;
    CodeGenerator::CodeStream simCode(simCodeStream);
    
    const auto stateVars = getStateVars();
    const size_t lastStateVarIdx = stateVars.size() - 1;
    
    // Generate macros containing derivatives
    for(const auto &s : stateVars) {
        simCode << "#define D" << s.name << "(";
        for(size_t i = 0; i < stateVars.size(); i++) {
            simCode << stateVars[i].name;
            if(i != lastStateVarIdx) {
                simCode << ", ";
            }
        }
        simCode << ") " << s.derivative << std::endl;
    }
    
    // Generate preamble
    simCode << getSimCodePreamble() << std::endl;
    
    simCode << "// Calculate RK4 terms" << std::endl;
    
    // Generate code to calculate first term
    for(const auto &s : stateVars) {
        simCode << "const scalar " << s.name << "1 = D" << s.name << "(";
        for(size_t i = 0; i < stateVars.size(); i++) {
            simCode << "$(" << stateVars[i].name << ")";
            if(i != lastStateVarIdx) {
                simCode << ", ";
            }
        }
        simCode << ");" << std::endl;
    }
   
    // Generate code to calculate second term
    for(const auto &s : stateVars) {
        simCode << "const scalar " << s.name << "2 = D" << s.name << "(";
        for(size_t i = 0; i < stateVars.size(); i++) {
            simCode << "$(" << stateVars[i].name << ") + (DT * 0.5 * " << stateVars[i].name << "1)";
            if(i != lastStateVarIdx) {
                simCode << ", ";
            }
        }
        simCode << ");" << std::endl;
    }
    
    // Generate code to calculate third term
    for(const auto &s : stateVars) {
        simCode << "const scalar " << s.name << "3 = D" << s.name << "(";
        for(size_t i = 0; i < stateVars.size(); i++) {
            simCode << "$(" << stateVars[i].name << ") + (DT * 0.5 * " << stateVars[i].name << "2)";
            if(i != lastStateVarIdx) {
                simCode << ", ";
            }
        }
        simCode << ");" << std::endl;
    }
    
    // Generate code to calculate fouth term
    for(const auto &s : stateVars) {
        simCode << "const scalar " << s.name << "4 = D" << s.name << "(";
        for(size_t i = 0; i < stateVars.size(); i++) {
            simCode << "$(" << stateVars[i].name << ") + (DT * " << stateVars[i].name << "3)";
            if(i != lastStateVarIdx) {
                simCode << ", ";
            }
        }
        simCode << ");" << std::endl;
    }
    
    // Generate state update code
    for(const auto &s : stateVars) {
        // If there was a condition on updates, write if clause and open scope
        if(!s.updateCondition.empty()) {
            simCode << "if(" << s.updateCondition << ")";
            simCode << CodeGenerator::CodeStream::OB(1);
        }
        
        // Write state update
        simCode << "$(" << s.name << ") += (DT / 6.0) * (" << s.name << "1 + (2.0f * (" << s.name << "2 + " << s.name << "3)) + " << s.name << "4);" << std::endl;
        
        // If there was a condition on updates, close scope
        if(!s.updateCondition.empty()) {
            simCode << CodeGenerator::CodeStream::CB(1);
        }
    }
    // Generate postamble
    simCode << getSimCodePostamble() << std::endl;
    return simCodeStream.str();
}
