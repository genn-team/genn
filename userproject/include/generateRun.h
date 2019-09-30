#pragma once

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

// Standard C includes
#include <cstdlib>


// Windows includes
#ifdef _WIN32
#include <direct.h>
#endif

// CLI11 includes
#include "../../include/genn/third_party/CLI11.hpp"

//------------------------------------------------------------------------
// GenerateRunBase
//------------------------------------------------------------------------
class GenerateRunBase
{
public:
    GenerateRunBase(const std::string &projectName)
    :   m_App{"Generate run application for '" + projectName + "' user project"},
        m_Debug(false), m_CPUOnly(false), m_Timing(false), m_ScalarType("float"), m_GPUDevice(-1), m_ProjectName(projectName)
    {
        m_App.add_flag("--debug", m_Debug, "Whether to run in a debugger");
        auto *cpuOnly = m_App.add_flag("--cpu-only", m_CPUOnly, "Whether to build against single-threaded CPU backend");
        m_App.add_flag("--timing", m_Timing, "Whether to use GeNN's timing mechanism to measure performance");
        m_App.add_set("--ftype", m_ScalarType, {"float", "double"}, "What floating point type to use", true);
        m_App.add_option("--gpu-device", m_GPUDevice, "What GPU device ID to use (-1 = select automatically)", true)->excludes(cpuOnly);
        m_App.add_option("experimentName", m_ExperimentName, "Experiment name")->required();
    }

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    // Write sizes header - override to write extra parameters - remember to call superclass first!
    virtual void writeSizes(std::ofstream &sizes) const
    {
        std::string upperCaseScalarType;
        std::transform(m_ScalarType.begin(), m_ScalarType.end(), std::back_inserter(upperCaseScalarType), ::toupper);

        sizes << "#pragma once" << std::endl;
        sizes << "#define _FTYPE " << "GENN_" << upperCaseScalarType << std::endl;
        sizes << "#define _TIMING " << m_Timing << std::endl;

        if(m_GPUDevice != -1) {
            sizes << "#define _GPU_DEVICE " << m_GPUDevice << std::endl;
        }
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

    int buildAndRun(std::initializer_list<std::string> runParams = {}) const
    {
        // create output directory
#ifdef _WIN32
        _mkdir(getOutDir().c_str());
#else // UNIX
        if (mkdir(getOutDir().c_str(), S_IRWXU | S_IRWXG | S_IXOTH) == -1) {
            std::cerr << "Directory cannot be created. It may exist already." << std::endl;
        }
#endif

        // Run any additional tools
        const int runToolsRetVal = runTools();
        if(runToolsRetVal != EXIT_SUCCESS) {
            return runToolsRetVal;
        }

        // build it
#ifdef _WIN32
        const std::string buildCmd = getBuildCommandWindows();
#else // UNIX
        const std::string buildCmd = getBuildCommandUnix();
#endif

        const int buildRetVal = system(buildCmd.c_str());
        if (buildRetVal != 0){
            std::cerr << "ERROR: Following call failed with status " << buildRetVal << ":" << std::endl << buildCmd << std::endl;
            std::cerr << "Exiting..." << std::endl;
            return EXIT_FAILURE;
        }

        // run it!
        std::cout << "running test..." << std::endl;
#ifdef _WIN32
        std::string runCmd = getRunCommandWindows();
#else // UNIX
        std::string runCmd = getRunCommandUnix();
#endif
        // Add out directory parameter
        runCmd += (" " + m_ExperimentName);

        // Add additional parameters
        for(const auto &p: runParams) {
            runCmd += (" " + p);
        }

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
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual int runTools() const
    {
        return EXIT_SUCCESS;
    }

    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    CLI::App &getApp(){ return m_App; }

    std::string getOutDir() const{ return m_ExperimentName + "_output"; }
    const std::string &getExperimentName() const{ return m_ExperimentName; }

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
            return "cd model && gdb -tui --args " + m_ProjectName;
        }
        else {
            return "cd model && ./" + m_ProjectName;
        }
    }

    std::string getRunCommandWindows() const
    {
        if (m_Debug) {
            return "cd model && devenv /debugexe " + m_ProjectName + "_Debug.exe";
        }
        else {
            return "cd model && " + m_ProjectName + "_Release.exe";
        }
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    CLI::App m_App;
    bool m_Debug;
    bool m_CPUOnly;
    bool m_Timing;
    std::string m_ScalarType;
    int m_GPUDevice;
    std::string m_ExperimentName;
    const std::string m_ProjectName;
};
