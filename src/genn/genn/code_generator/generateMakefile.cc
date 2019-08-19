#include "code_generator/generateMakefile.h"

// Standard C++ includes
#include <string>

// GeNN includes
#include "modelSpec.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateMakefile(std::ostream &os, const BackendBase &backend,
                                     const std::vector<std::string> &moduleNames)
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
    // On Mac OS X add final step to recipe to make librunner relative to rpath
#ifdef __APPLE__
    os << "\t@install_name_tool -id \"@rpath/$@\" $@" << std::endl;
#endif
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
    os << "\t@rm -f $(OBJECTS) $(DEPS) librunner.so" << std::endl;
}
