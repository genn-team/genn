#include "../include/generateRun.h"

class GenerateRun : public GenerateRunBase
{
public:
    GenerateRun()
    :   GenerateRunBase("MBody1"), m_Bitmask(false), m_DelayedSynapses(false), m_NumAL(100), m_NumKC(1000), m_NumLHI(20), m_NumDN(100), m_GScale(0.0025)
    {
        getApp().add_flag("--bitmask", m_Bitmask, "Whether to use bitmasks to represent sparse PN->KC connectivity");
        getApp().add_flag("--delayed-synapses", m_DelayedSynapses,  "Whether to simulate delays of (5 * DT) ms on KC->DN and of (3 * DT) ms on DN->DN synapse populations");
        getApp().add_option("--num-al", m_NumAL, "Number of neurons in the antennal lobe (AL), the input neurons to this model", true);
        getApp().add_option("--num-kc", m_NumKC, "Number of Kenyon cells (KC) in the \"hidden layer\"", true);
        getApp().add_option("--num-lhi", m_NumLHI, "Number of lateral horn interneurons, implementing gain control", true);
        getApp().add_option("--num-dn", m_NumDN, "Number of decision neurons (DN) in the output layer", true);
        getApp().add_option("--gscale", m_GScale, "Scaling of synaptic conductances", true);
    }

    //------------------------------------------------------------------------
    // GenerateRunBase virtuals
    //------------------------------------------------------------------------
    virtual void writeSizes(std::ofstream &sizes) const override
    {
        // Superclass
        GenerateRunBase::writeSizes(sizes);

        sizes << "#define _NAL " << m_NumAL << std::endl;
        sizes << "#define _NKC " << m_NumKC << std::endl;
        sizes << "#define _NLHI " << m_NumLHI << std::endl;
        sizes << "#define _NDN " << m_NumDN << std::endl;
        sizes << "#define _GScale " << m_GScale << std::endl;

        if(m_Bitmask) {
            sizes << "#define BITMASK" << std::endl;
        }

        if(m_DelayedSynapses) {
            sizes << "#define DELAYED_SYNAPSES" << std::endl;
        }
    }

protected:
    //------------------------------------------------------------------------
    // GenerateRunBase virtuals
    //------------------------------------------------------------------------
    virtual int runTools() const override
    {
        // generate input patterns
#ifdef _WIN32
        std::string cmd = "..\\tools\\gen_input_structured.exe ";
#else
        std::string cmd = "../tools/gen_input_structured ";
#endif
        // <nAL> <# classes> <# pattern/ input class> <prob. to be active> <perturbation prob. in class>
        // <'on' rate> <baseline rate>
        cmd +=  std::to_string(m_NumAL);
        cerr << "<outfile> 
        cmd += " 10 10 0.1 0.1 1000.0 0.2 ";   // p_perturb only sensible if >= 1/n_act (where n_act=p_act*nAL); this assumes nAL >= 100
        cmd += getOutDir() + "/" + getExperimentName() + ".inpat 2>&1 ";
#ifndef _WIN32
        cmd += "|tee " + getOutDir() + "/" + getExperimentName() + ".inpat.msg";
#endif // _WIN32

        const int retval = system(cmd.c_str());
        if (retval != 0){
            std::cerr << "ERROR: Following call failed with status " << retval << ":" << std::endl << cmd << std::endl;
            std::cerr << "Exiting..." << std::endl;
            return retval;
        }

        return EXIT_SUCCESS;
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    bool m_Bitmask;
    bool m_DelayedSynapses;
    unsigned int m_NumAL;
    unsigned int m_NumKC;
    unsigned int m_NumLHI;
    unsigned int m_NumDN;
    double m_GScale;
};

int main(int argc, char *argv[])
{
    // Parse command line
    GenerateRun generateRun;
    try {
        generateRun.parseCommandLine(argc, argv);
    }
    catch(const CLI::ParseError &e) {
        return generateRun.getExitCode(e);
    }

    // Write model sizes
    {
        std::ofstream sizes("model/sizes.h");
        generateRun.writeSizes(sizes);
    }

    // Build and run model
    return generateRun.buildAndRun();
}
