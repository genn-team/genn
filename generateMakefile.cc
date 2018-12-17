#include "generateMakefile.h"

// Standard C++ includes
#include <string>

// GeNN includes
#include "codeStream.h"
#include "modelSpec.h"

// NuGeNN includes
#include "backends/base.h"

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateMakefile(std::ostream &os, const Backends::Base &backend,
                                     const std::vector<std::string> moduleNames)
{
    //**TODO** deal with standard include paths e.g. MPI here

    // Generate make file preamble
    backend.genMakefilePreamble(os);

    // List objects in makefile
    os << "OBJECTS := ";
    for(const auto &m : moduleNames) {
        os << m << ".o ";
    }
    os << std::endl;

    // Apply substitution to generate dependency list
    os << "DEPS := $(OBJECTS:.o=.d)" << std::endl;
    os << std::endl;

    // Generate phony rules for all and clean
    os << ".PHONY: all clean" << std::endl;
    os << std::endl;

    // Add rule to build runner
    os << "all: librunner.so" << std::endl;
    os << std::endl;

    // Add rule to build shared library from objects
    os << "librunner.so: $(OBJECTS)" << std::endl;
    backend.genMakefileLinkRule(os);
    os << std::endl;

    // Include depencies
    os << "-include $(DEPS)" << std::endl;
    os << std::endl;

    // Generate compile rule build objects from source files
    backend.genMakefileCompileRule(os);
    os << std::endl;

    // Add dummy rule to handle missing .d files on first build
    os << "%.d: ;" << std::endl;
    os << std::endl;

    // Add clean rule
    os << "clean:" << std::endl;
    os << "\trm -f $(OBJECTS) $(DEPS) librunner.so" << std::endl;
}