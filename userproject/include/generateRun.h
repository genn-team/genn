#pragma once

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

// Standard C includes
#include <cstdlib>

// CLI11 includes
#include "CLI11.hpp"

//------------------------------------------------------------------------
// GenerateRunBase
//------------------------------------------------------------------------
class GenerateRunBase
{
public:
    GenerateRunBase(const std::string &projectName)
    :   m_App{"Generate run application for '" + projectName + "' user project"},
        m_Debug(false), m_CPUOnly(false), m_ScalarType("float"), m_ProjectName(projectName)
    {
        m_App.add_flag("--debug", m_Debug, "Whether to run in a debugger");
        m_App.add_flag("--cpu-only", m_CPUOnly, "Whether to build against single-threaded CPU backend");
        m_App.add_set("--ftype", m_ScalarType, {"float", "double"}, "What floating point type to use", true);
        m_App.add_option("outdir", m_OutDir, "Output directory", true)->required();
    }

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    // Write sizes header - override to write extra parameters - remember to call superclass first!
    virtual void writeSizes(std::ofstream &sizes) const
    {
        std::cout << "written:" << m_ScalarType << std::endl;
        std::string upperCaseScalarType;
        std::transform(m_ScalarType.begin(), m_ScalarType.end(), std::back_inserter(upperCaseScalarType), ::toupper);
        std::cout << "uppered:" << upperCaseScalarType << std::endl;
        sizes << "#pragma once" << std::endl;
        sizes << "#define _FTYPE " << "GENN_" << upperCaseScalarType << std::endl;
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    int getExitCode(const CLI::ParseError &e) {
        return m_App.exit(e);
    }

    void parseCommandLine(int argc, char **argv)
    {
        m_App.parse(argc, argv);
    }

    int buildAndRun() const
    {
        // build it
#ifdef _WIN32
        const std::string buildCmd = getBuildCommandWindows();
#else // UNIX
        const std::string buildCmd = getBuildCommandUnix();
#endif

        std::cout << buildCmd << std::endl;
        const int buildRetVal = system(buildCmd.c_str());
        if (buildRetVal != 0){
            std::cerr << "ERROR: Following call failed with status " << buildRetVal << ":" << std::endl << buildCmd << std::endl;
            std::cerr << "Exiting..." << std::endl;
            return EXIT_FAILURE;
        }

  // create output directory
#ifdef _WIN32
        _mkdir(m_OutDir.c_str());
#else // UNIX
        if (mkdir(m_OutDir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH) == -1) {
            std::cerr << "Directory cannot be created. It may exist already." << std::endl;
        }
#endif

        // run it!
        std::cout << "running test..." << std::endl;
#ifdef _WIN32
        const std::string runCmd = getRunCommandWindows();
#else // UNIX
        const std::string runCmd = getRunCommandUnix();
#endif
        const int runRetVal = system(runCmd.c_str());
        if (runRetVal != 0){
            std::cerr << "ERROR: Following call failed with status " << runRetVal << ":" << std::endl << runCmd << std::endl;
            std::cerr << "Exiting..." << std::endl;
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    CLI::App &getApp(){ return m_App; }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    std::string getBuildCommandUnix() const
    {
        std::string cmd = "cd model && genn-buildmodel.sh ";
        cmd += m_ProjectName + ".cc";
        if (m_Debug) {
            cmd += " -d";
        }
        if (m_CPUOnly) {
            cmd += " -c";
        }

        cmd += " && make clean all";
        const int retval = system(cmd.c_str());
        if (retval != 0){
            std::cerr << "ERROR: Following call failed with status " << retval << ":" << std::endl << cmd << std::endl;
            std::cerr << "Exiting..." << std::endl;
            exit(1);
        }

       return cmd;
    }

    std::string getBuildCommandWindows() const
    {
        std::string cmd = "cd model && genn-buildmodel.bat ";
        cmd += m_ProjectName + ".cc";
        if (m_Debug) {
            cmd += " -d";
        }
        if (m_CPUOnly) {
            cmd += " -c";
        }

        cmd += " && msbuild " + m_ProjectName + ".sln /t:" + m_ProjectName + " /p:Configuration=";
        if (m_Debug) {
            cmd += "Debug";
        }
        else {
            cmd += "Release";
        }
        return cmd;
    }

    std::string getRunCommandUnix() const
    {
        // **TODO** cpu-only debugger
        if (m_Debug) {
            return "cd model && cuda-gdb -tui --args " + m_ProjectName + " " + m_OutDir;
        }
        else {
            return "cd model && ./" + m_ProjectName + " " + m_OutDir;
        }
    }

    std::string getRunCommandWindows() const
    {
        if (m_Debug) {
            return "cd model && devenv /debugexe " + m_ProjectName + "_Debug.exe " + m_OutDir;
        }
        else {
            return "cd model && " + m_ProjectName + "_Release.exe " + m_OutDir;
        }
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    CLI::App m_App;
    bool m_Debug;
    bool m_CPUOnly;
    std::string m_ScalarType;
    std::string m_OutDir;
    const std::string m_ProjectName;
};
