#include "code_generator/generateNMakefile.h"

// Standard C++ includes
#include <string>

// GeNN includes
#include "modelSpec.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

//--------------------------------------------------------------------------
// GeNN::CodeGenerator
//--------------------------------------------------------------------------
void GeNN::CodeGenerator::generateNMakefile(std::ostream &os, const BackendBase &backend,
                                            const std::vector<std::string> &moduleNames)
{
    // CC isn't a standard suffix so needs specifying
    os << ".SUFFIXES: .cc" << std::endl;

    // List objects in makefile
    os << "OBJECTS = ";
    for(const auto &m : moduleNames) {
        os << m << ".obj ";
    }
    os << std::endl;

    // Generate make file preamble
    backend.genNMakefilePreamble(os);
    os << std::endl;

    
    // Add rule to build runner
    os << "all: runner.dll" << std::endl;
    os << std::endl;

    // Add rule to build DLL from objects
    // **NOTE** because e.g. CUDA requires seperate device linking dependencies not known
    backend.genNMakefileLinkRule(os);
    os << std::endl;

    // Generate compile rule build objects from source files
    backend.genNMakefileCompileRule(os);
    os << std::endl;

    // Add clean rule
    os << "clean:" << std::endl;
    os << "\tdel $(OBJECTS) runner.exp runner.lib runner.dll 2>nul" << std::endl;
}
