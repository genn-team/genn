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
public:
    Simulator();
    Simulator(const std::string &experimentXML, const std::string &overrideOutputPath = "");
    ~Simulator();

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Load model from XML file
    void load(const std::string &experimentXML, const std::string &overrideOutputPath = "");

    //! Advance simulation by one timestep
    void stepTime();

    //! Get the simulation timestep (in ms)
    double getDT() const{ return m_DT; }

    //! Get duration of simulation read from experiment in ms
    double getDurationMs() const{ return m_DurationMs; }

    //! Get the total times accumulated in each stage of the simulation
    double getInputMs() const{ return m_InputMs; }
    double getSimulateMs() const{ return m_SimulateMs; }
    double getLogMs() const{ return m_LogMs; }

    //! Calculate duration of simulation read from experiment in timesteps
    unsigned int calcNumTimesteps() const
    {
        return (unsigned int)std::ceil(getDurationMs() / getDT());
    }

private:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    //! Function pointer type for void function
    typedef void (*VoidFunction)(void);

    //! Map from a component name to a set of event send and event receive port names
    typedef std::map<std::string, std::pair<std::set<std::string>, std::set<std::string>>> ComponentEventPorts;

    //! Tuple containing variables and functions for accessing neuron population's spiking output
    /*! hostSpikeCount, *hostSpikes, spikeQueuePtr, pushFunc, pullFunc */
    typedef std::tuple<unsigned int*, unsigned int*, unsigned int*, VoidFunction, VoidFunction> NeuronPopSpikeVars;

    //! Tuple containing variables and functions for accesing component state variable
    /*! pointer, pushFunc, pullFunc */
    template<typename T>
    using ComponentStateVar = std::tuple<T*, ModelProperty::Base::PushFunc, ModelProperty::Base::PullFunc>;

    //------------------------------------------------------------------------
    // Private functions
    //------------------------------------------------------------------------
    //! Get a named symbol from the model library
    /*! if allowMissing is true, returns nullptr if symbol is not found, otherwise throws exception */
    void *getLibrarySymbol(const char *name, bool allowMissing = false) const;


    NeuronPopSpikeVars getNeuronPopSpikeVars(const std::string &popName) const;

    void addPropertiesAndSizes(const filesystem::path &basePath, const pugi::xml_node &node,
                               const pugi::xml_node &modelNode, const std::string &geNNPopName,
                               unsigned int popSize, std::map<std::string, unsigned int> &sizes,
                               const std::vector<unsigned int> *remapIndices = nullptr);

    //! Create the correct type of input object to simulate node.
    std::unique_ptr<Input::Base> createInput(const pugi::xml_node &node,
                                             const std::map<std::string, unsigned int> &componentSizes,
                                             const std::map<std::string, std::string> &componentURLs,
                                             const ComponentEventPorts &componentEventPorts);

    std::unique_ptr<LogOutput::Base> createLogOutput(const pugi::xml_node &node, const filesystem::path &logPath,
                                                     const std::map<std::string, unsigned int> &componentSizes,
                                                     const std::map<std::string, std::string> &componentURLs,
                                                     const ComponentEventPorts &componentEventPorts);

    //! Get size i.e. number of neurons/synapses in named component
    unsigned int getComponentSize(const std::string &componentName,
                                  const std::map<std::string, unsigned int> &componentSizes);

    //! Does the named target component have an event send port with this name?
    bool isEventSendPort(const std::string &targetName, const std::string &portName,
                         const std::map<std::string, std::string> &componentURLs,
                         const ComponentEventPorts &componentEventPorts) const;

    //! Does the named target component have an event receive port with this name?
    bool isEventReceivePort(const std::string &targetName, const std::string &portName,
                            const std::map<std::string, std::string> &componentURLs,
                            const ComponentEventPorts &componentEventPorts) const;

    //! If the model used by the component specified by node doesn't already have it's event ports cached,
    //! Parse component XML and population componentEventPorts
    void addEventPorts(const filesystem::path &basePath, const pugi::xml_node &node,
                       std::map<std::string, std::string> &componentURLs,
                       ComponentEventPorts &componentEventPorts);

    //! Helper function to get pointers to state variable as well as push and pull functions
    template <typename T>
    ComponentStateVar<T> getStateVar(const std::string &stateVarName) const
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
    //! Handle to model library
    LIBRARY_HANDLE m_ModelLibrary;

    //! Pointer to stepTime function in model library
    VoidFunction m_StepTime;

    //! Pointer to simulation time symbol in model library
    float *m_SimulationTime;

    //! Pointer to simulation timestep symbol in model library
    unsigned long long *m_SimulationTimestep;

    //! Timestep of simulation
    double m_DT;

    //! Duration of simulation in ms
    double m_DurationMs;

    //! Timing of various parts of simulation
    double m_InputMs;
    double m_SimulateMs;
    double m_LogMs;

    //! Vector of logging objects, updated at the end of each simulation time step
    std::vector<std::unique_ptr<LogOutput::Base>> m_Loggers;

    //! Vector of input objects, updated at the beginning of each simulation time step
    std::vector<std::unique_ptr<Input::Base>> m_Inputs;

    //! Map of model properties associated with each component
    std::map<std::string, std::map<std::string, std::unique_ptr<ModelProperty::Base>>> m_ComponentProperties;
};
}   // namespace SpineMLSimulator
