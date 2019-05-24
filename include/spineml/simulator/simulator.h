#pragma once

// Standard C++ includes
#include <map>
#include <set>
#include <string>
#include <tuple>

// SpineML simulator includes
#include "input.h"
#include "logOutput.h"
#include "modelProperty.h"


//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#ifdef _WIN32
    #define LIBRARY_HANDLE HMODULE
#else
    #define LIBRARY_HANDLE void*
#endif

//----------------------------------------------------------------------------
// SpineMLSimulator::Simulator
//----------------------------------------------------------------------------
namespace SpineMLSimulator
{
class Simulator
{
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    // Function pointer type for void function
    typedef void (*VoidFunction)(void);

    // Map from a component name to a set of event send and event receive port names
    typedef std::map<std::string, std::pair<std::set<std::string>, std::set<std::string>>> ComponentEventPorts;

public:
    Simulator();
    Simulator(const std::string &experimentXML, const std::string &overrideOutputPath = "");
    ~Simulator();

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void load(const std::string &experimentXML, const std::string &overrideOutputPath = "");
    void stepTime();

    double getDT() const{ return m_DT; }
    double getDurationMs() const{ return m_DurationMs; }

    double getInputMs() const{ return m_InputMs; }
    double getSimulateMs() const{ return m_SimulateMs; }
    double getLogMs() const{ return m_LogMs; }

    unsigned int calcNumTimesteps() const
    {
        return (unsigned int)std::ceil(getDurationMs() / getDT());
    }

private:
    //------------------------------------------------------------------------
    // Private functions
    //------------------------------------------------------------------------
    void *getLibrarySymbol(const char *name, bool allowMissing = false) const;

    std::tuple<unsigned int*, unsigned int*, unsigned int*, VoidFunction, VoidFunction> getNeuronPopSpikeVars(const std::string &popName) const;

    void addPropertiesAndSizes(const filesystem::path &basePath, const pugi::xml_node &node, const pugi::xml_node &modelNode, const std::string &geNNPopName, unsigned int popSize,
                               std::map<std::string, unsigned int> &sizes, const std::vector<unsigned int> *remapIndices = nullptr);

    std::unique_ptr<Input::Base> createInput(const pugi::xml_node &node, const std::map<std::string, unsigned int> &componentSizes,
                                             const std::map<std::string, std::string> &componentURLs, const ComponentEventPorts &componentEventPorts);

    std::unique_ptr<LogOutput::Base> createLogOutput(const pugi::xml_node &node, const filesystem::path &logPath,
                                                     const std::map<std::string, unsigned int> &componentSizes,
                                                     const std::map<std::string, std::string> &componentURLs,
                                                     const ComponentEventPorts &componentEventPorts);

    unsigned int getComponentSize(const std::string &componentName,
                                  const std::map<std::string, unsigned int> &componentSizes);

    bool isEventSendPort(const std::string &targetName, const std::string &portName,
                         const std::map<std::string, std::string> &componentURLs,
                         const ComponentEventPorts &componentEventPorts) const;

    bool isEventReceivePort(const std::string &targetName, const std::string &portName,
                            const std::map<std::string, std::string> &componentURLs,
                            const ComponentEventPorts &componentEventPorts) const;

    void addEventPorts(const filesystem::path &basePath, const pugi::xml_node &node,
                       std::map<std::string, std::string> &componentURLs,
                       ComponentEventPorts &componentEventPorts);

    template <typename T>
    std::tuple<T*, ModelProperty::Base::PushFunc, ModelProperty::Base::PullFunc> getStateVar(const std::string &stateVarName) const
    {
        // Get host statevar
        T *hostStateVar = reinterpret_cast<T*>(getLibrarySymbol(stateVarName.c_str(), true));

        // Get push and pull functions
        auto pushFunc = (ModelProperty::Base::PushFunc)getLibrarySymbol(("push" + stateVarName + "ToDevice").c_str(), true);
        auto pullFunc = (ModelProperty::Base::PullFunc)getLibrarySymbol(("pull" + stateVarName + "FromDevice").c_str(), true);

        // Return in tuple
        return std::make_tuple(hostStateVar, pushFunc, pullFunc);
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    LIBRARY_HANDLE m_ModelLibrary;
    VoidFunction m_StepTime;

    float *m_SimulationTime;

    unsigned long long *m_SimulationTimestep;

    // Timestep of simulation
    double m_DT;

    // Duration of simulation in ms
    double m_DurationMs;

    // Timing of various parts of simulation
    double m_InputMs;
    double m_SimulateMs;
    double m_LogMs;

    std::vector<std::unique_ptr<LogOutput::Base>> m_Loggers;
    std::vector<std::unique_ptr<Input::Base>> m_Inputs;
    std::map<std::string, std::map<std::string, std::unique_ptr<ModelProperty::Base>>> m_ComponentProperties;
};
}   // namespace SpineMLSimulator
