// Standard C++ includes
#include <iostream>
#include <map>
#include <set>
#include <string>

// Standard C includes
#include <cassert>
#include <cstdlib>

// POSIX C includes
extern "C"
{
#include <dlfcn.h>
}

// pugixml includes
#include "pugixml/pugixml.hpp"

//using namespace SpineMLSimulator;

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define LOAD_SYMBOL(LIBRARY, TYPE, NAME)                            \
    TYPE NAME = (TYPE)dlsym(LIBRARY, #NAME);                        \
    if(NAME == NULL) {                                              \
        throw std::runtime_error("Cannot find " #NAME " function"); \
    }

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
// Typedefines
typedef float scalar;
typedef void (*VoidFunction)(void);
typedef void (*AllocateFn)(unsigned int);
}   // Anonymous namespace

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    if(argc != 4) {
        std::cerr << "Expected model library and; network and experiment XML files passed as arguments" << std::endl;
        return EXIT_FAILURE;
    }

    void *modelLibrary = NULL;
    try
    {
        // Attempt to load model library
        modelLibrary = dlopen(argv[1], RTLD_NOW);

        // If it fails throw
        if(modelLibrary == NULL)
        {
            throw std::runtime_error("Unable to load library - error:" + std::string(dlerror()));
        }

        // Load statically-named symbols from library
        LOAD_SYMBOL(modelLibrary, VoidFunction, initialize);
        LOAD_SYMBOL(modelLibrary, VoidFunction, allocateMem);
        LOAD_SYMBOL(modelLibrary, VoidFunction, stepTimeCPU);
#ifndef CPU_ONLY
        LOAD_SYMBOL(modelLibrary, VoidFunction, stepTimeGPU);
#endif // CPU_ONLY

        // Load network document
        pugi::xml_document networkDoc;
        auto result = networkDoc.load_file(argv[2]);
        if(!result) {
            throw std::runtime_error("Unable to load XML file:" + std::string(argv[2]) + ", error:" + result.description());
        }

        // Get SpineML root
        auto spineML = networkDoc.child("SpineML");
        if(!spineML) {
            throw std::runtime_error("XML file:" + std::string(argv[2]) + " is not a SpineML network - it has no root SpineML node");
        }

        // Loop through populations once to build neuron populations
        for(auto population : spineML.children("Population")) {
            auto neuron = population.child("Neuron");
            if(!neuron) {
                throw std::runtime_error("'Population' node has no 'Neuron' node");
            }

            // Read basic population properties
            const auto *popName = neuron.attribute("name").value();
            const unsigned int popSize = neuron.attribute("size").as_int();
            std::cout << "Population " << popName << " consisting of ";
            std::cout << popSize << " neurons" << std::endl;

             for(auto param : neuron.children("Property")) {
                const auto *paramName = param.attribute("name").value();
                std::cout << paramName << std::endl;
             }

        }

    }
    catch(...)
    {
        // Close model library if loaded successfully
        if(modelLibrary)
        {
            dlclose(modelLibrary);
        }

        // Re-raise
        throw;
    }

    return 0;
}