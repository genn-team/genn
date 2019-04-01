#include "../include/generateRun.h"

class GenerateRun : public GenerateRunBase
{
public:
    GenerateRun()
    :   GenerateRunBase("MBody1"), m_Bitmask(false), m_DelayedSynapses(false)
    {
        getApp().add_flag("--bitmask", m_Bitmask, "Whether to use bitmasks to represent sparse PN->KC connectivity");
        getApp().add_flag("--delayed-synapses", m_DelayedSynapses,  "Whether to simulate delays of (5 * DT) ms on KC->DN and of (3 * DT) ms on DN->DN synapse populations");
        getApp().add_option("numAL", m_NumAL, "Number of neurons in the antennal lobe (AL), the input neurons to this model")->required();
        getApp().add_option("numKC", m_NumKC, "Number of Kenyon cells (KC) in the \"hidden layer\"")->required();
        getApp().add_option("numLHI", m_NumLHI, "Number of lateral horn interneurons, implementing gain control")->required();
        getApp().add_option("numDN", m_NumDN, "Number of decision neurons (DN) in the output layer")->required();
        getApp().add_option("gscale", m_GScale, "Scaling of synaptic conductances")->required();
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
