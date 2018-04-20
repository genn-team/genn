// Standard C++ includes
#include <tuple>

// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpec.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
//--------------------------------------------------------------------------
// ConstNeuronVars
//--------------------------------------------------------------------------
class ConstNeuronVars : public ::testing::Test
{
protected:
    //--------------------------------------------------------------------------
    // Test virtuals
    //--------------------------------------------------------------------------
    virtual void SetUp()
    {
        GENN_PREFERENCES::autoInitSparseVars = false;

        NeuronModels::Izhikevich::ParamValues paramVals(
            0.02, 0.2, -65.0, 8.0);
        NeuronModels::Izhikevich::VarValues varVals(
            0.0, 0.0);

        m_NeuronGroups[0] = m_Model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
        m_NeuronGroups[1] = m_Model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    }

    //--------------------------------------------------------------------------
    // Protected API
    //--------------------------------------------------------------------------
    NNmodel &getModel() { return m_Model; }
    NeuronGroup *getNeuronGroup(size_t i) { return m_NeuronGroups[i]; }

private:
    //--------------------------------------------------------------------------
    // Private API
    //--------------------------------------------------------------------------
    NNmodel m_Model;
    NeuronGroup *m_NeuronGroups[2];
};

//--------------------------------------------------------------------------
// ConstNeuronSynapseVars
//--------------------------------------------------------------------------
class ConstNeuronSynapseVars : public ::testing::TestWithParam<std::tuple<SynapseMatrixType, bool>>
{
protected:
    //--------------------------------------------------------------------------
    // Test virtuals
    //--------------------------------------------------------------------------
    virtual void SetUp()
    {
        GENN_PREFERENCES::autoInitSparseVars = std::get<1>(GetParam());

        NeuronModels::Izhikevich::ParamValues paramVals(
            0.02, 0.2, -65.0, 8.0);
        NeuronModels::Izhikevich::VarValues varVals(
            0.0, 0.0);

        WeightUpdateModels::StaticPulse::VarValues wumVals(0.0);
        PostsynapticModels::ExpCond::ParamValues psmParamVals(5.0, -85.0);

        m_NeuronGroups[0] = m_Model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
        m_NeuronGroups[1] = m_Model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

        m_SynapseGroup = m_Model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCond>(
            "EE", std::get<0>(GetParam()), NO_DELAY,
            "Neurons0", "Neurons1",
            {}, wumVals,
            psmParamVals, {});
    }

    //--------------------------------------------------------------------------
    // Protected API
    //--------------------------------------------------------------------------
    NNmodel &getModel() { return m_Model; }
    NeuronGroup *getNeuronGroup(size_t i) { return m_NeuronGroups[i]; }
    SynapseGroup *getSynapseGroup() { return m_SynapseGroup; }

private:
    //--------------------------------------------------------------------------
    // Private API
    //--------------------------------------------------------------------------
    NNmodel m_Model;
    NeuronGroup *m_NeuronGroups[2];
    SynapseGroup *m_SynapseGroup;
};
}   // Anonymous namespace

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST_F(ConstNeuronVars, SpikesLocDeviceInitDevice)
{
    getNeuronGroup(0)->setSpikeVarMode(VarMode::LOC_DEVICE_INIT_DEVICE);

    ASSERT_FALSE(getModel().zeroCopyInUse());
    ASSERT_TRUE(getModel().isDeviceInitRequired(0));
    ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());

#ifndef CPU_ONLY
    ASSERT_FALSE(getModel().canRunOnCPU());
#else
    ASSERT_TRUE(getModel().canRunOnCPU());
#endif
}

TEST_F(ConstNeuronVars, SpikesLocHostDeviceInitHost)
{
    getNeuronGroup(0)->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_HOST);

    ASSERT_FALSE(getModel().zeroCopyInUse());
    ASSERT_FALSE(getModel().isDeviceInitRequired(0));
    ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());
    ASSERT_TRUE(getModel().canRunOnCPU());
}

TEST_F(ConstNeuronVars, SpikesLocHostDeviceInitDevice)
{
    getNeuronGroup(0)->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    ASSERT_FALSE(getModel().zeroCopyInUse());
    ASSERT_TRUE(getModel().isDeviceInitRequired(0));
    ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());
    ASSERT_TRUE(getModel().canRunOnCPU());
}

TEST_F(ConstNeuronVars, SpikesLocZeroCopyInitHost)
{
    getNeuronGroup(0)->setSpikeVarMode(VarMode::LOC_ZERO_COPY_INIT_HOST);

    ASSERT_TRUE(getModel().zeroCopyInUse());
    ASSERT_FALSE(getModel().isDeviceInitRequired(0));
    ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());
    ASSERT_TRUE(getModel().canRunOnCPU());
}

TEST_F(ConstNeuronVars, SpikesLocZeroCopyInitDevice)
{
    getNeuronGroup(0)->setSpikeVarMode(VarMode::LOC_ZERO_COPY_INIT_DEVICE);

    ASSERT_TRUE(getModel().zeroCopyInUse());
    ASSERT_TRUE(getModel().isDeviceInitRequired(0));
    ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());
    ASSERT_TRUE(getModel().canRunOnCPU());
}

TEST_F(ConstNeuronVars, VarLocDeviceInitDevice)
{
    getNeuronGroup(0)->setVarMode("U", VarMode::LOC_DEVICE_INIT_DEVICE);

    ASSERT_FALSE(getModel().zeroCopyInUse());
    ASSERT_TRUE(getModel().isDeviceInitRequired(0));
    ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());

#ifndef CPU_ONLY
    ASSERT_FALSE(getModel().canRunOnCPU());
#else
    ASSERT_TRUE(getModel().canRunOnCPU());
#endif
}

TEST_F(ConstNeuronVars, VarLocHostDeviceInitHost)
{
    getNeuronGroup(0)->setVarMode("U", VarMode::LOC_HOST_DEVICE_INIT_HOST);

    ASSERT_FALSE(getModel().zeroCopyInUse());
    ASSERT_FALSE(getModel().isDeviceInitRequired(0));
    ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());
    ASSERT_TRUE(getModel().canRunOnCPU());
}

TEST_F(ConstNeuronVars, VarLocHostDeviceInitDevice)
{
    getNeuronGroup(0)->setVarMode("U", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    ASSERT_FALSE(getModel().zeroCopyInUse());
    ASSERT_TRUE(getModel().isDeviceInitRequired(0));
    ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());
    ASSERT_TRUE(getModel().canRunOnCPU());
}

TEST_F(ConstNeuronVars, VarLocZeroCopyInitHost)
{
    getNeuronGroup(0)->setVarMode("U", VarMode::LOC_ZERO_COPY_INIT_HOST);

    ASSERT_TRUE(getModel().zeroCopyInUse());
    ASSERT_FALSE(getModel().isDeviceInitRequired(0));
    ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());
    ASSERT_TRUE(getModel().canRunOnCPU());
}

TEST_F(ConstNeuronVars, VarLocZeroCopyInitDevice)
{
    getNeuronGroup(0)->setVarMode("U", VarMode::LOC_ZERO_COPY_INIT_DEVICE);

    ASSERT_TRUE(getModel().zeroCopyInUse());
    ASSERT_TRUE(getModel().isDeviceInitRequired(0));
    ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());
    ASSERT_TRUE(getModel().canRunOnCPU());
}

TEST_P(ConstNeuronSynapseVars, VarLocDeviceInitDevice)
{
    getSynapseGroup()->setWUVarMode("g", VarMode::LOC_DEVICE_INIT_DEVICE);

    ASSERT_FALSE(getModel().zeroCopyInUse());

    if(std::get<1>(GetParam()) && std::get<0>(GetParam()) == SynapseMatrixType::SPARSE_INDIVIDUALG) {
        ASSERT_FALSE(getModel().isDeviceInitRequired(0));
        ASSERT_TRUE(getModel().isDeviceSparseInitRequired());
    }
    else if(std::get<0>(GetParam()) == SynapseMatrixType::DENSE_INDIVIDUALG) {
        ASSERT_TRUE(getModel().isDeviceInitRequired(0));
    }
    else {
        ASSERT_FALSE(getModel().isDeviceInitRequired(0));
        ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    }

    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());

#ifndef CPU_ONLY
    ASSERT_FALSE(getModel().canRunOnCPU());
#else
    ASSERT_TRUE(getModel().canRunOnCPU());
#endif
}

TEST_P(ConstNeuronSynapseVars, VarLocHostDeviceInitHost)
{
    getSynapseGroup()->setWUVarMode("g", VarMode::LOC_HOST_DEVICE_INIT_HOST);

    ASSERT_FALSE(getModel().zeroCopyInUse());
    ASSERT_FALSE(getModel().isDeviceInitRequired(0));
    ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());
    ASSERT_TRUE(getModel().canRunOnCPU());
}

TEST_P(ConstNeuronSynapseVars, VarLocHostDeviceInitDevice)
{
    getSynapseGroup()->setWUVarMode("g", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    ASSERT_FALSE(getModel().zeroCopyInUse());

    if(std::get<1>(GetParam()) && std::get<0>(GetParam()) == SynapseMatrixType::SPARSE_INDIVIDUALG) {
        ASSERT_FALSE(getModel().isDeviceInitRequired(0));
        ASSERT_TRUE(getModel().isDeviceSparseInitRequired());
    }
    else if(std::get<0>(GetParam()) == SynapseMatrixType::DENSE_INDIVIDUALG) {
        ASSERT_TRUE(getModel().isDeviceInitRequired(0));
    }
    else {
        ASSERT_FALSE(getModel().isDeviceInitRequired(0));
        ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    }

    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());
    ASSERT_TRUE(getModel().canRunOnCPU());
}

TEST_P(ConstNeuronSynapseVars, VarLocZeroCopyInitHost)
{
    getSynapseGroup()->setWUVarMode("g", VarMode::LOC_ZERO_COPY_INIT_HOST);

    ASSERT_TRUE(getModel().zeroCopyInUse());
    ASSERT_FALSE(getModel().isDeviceInitRequired(0));
    ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());
    ASSERT_TRUE(getModel().canRunOnCPU());
}

TEST_P(ConstNeuronSynapseVars, VarLocZeroCopyInitDevice)
{
    getSynapseGroup()->setWUVarMode("g", VarMode::LOC_ZERO_COPY_INIT_DEVICE);

    ASSERT_TRUE(getModel().zeroCopyInUse());

    if(std::get<1>(GetParam()) && std::get<0>(GetParam()) == SynapseMatrixType::SPARSE_INDIVIDUALG) {
        ASSERT_FALSE(getModel().isDeviceInitRequired(0));
        ASSERT_TRUE(getModel().isDeviceSparseInitRequired());
    }
    else if(std::get<0>(GetParam()) == SynapseMatrixType::DENSE_INDIVIDUALG) {
        ASSERT_TRUE(getModel().isDeviceInitRequired(0));
    }
    else {
        ASSERT_FALSE(getModel().isDeviceInitRequired(0));
        ASSERT_FALSE(getModel().isDeviceSparseInitRequired());
    }

    ASSERT_FALSE(getModel().isHostRNGRequired());
    ASSERT_FALSE(getModel().isDeviceRNGRequired());
    ASSERT_TRUE(getModel().canRunOnCPU());
}

//--------------------------------------------------------------------------
// Instatiations
//--------------------------------------------------------------------------
INSTANTIATE_TEST_CASE_P(SynapseMatrixTypes,
                        ConstNeuronSynapseVars,
                        ::testing::Combine(
                            ::testing::Values(true, false),
                            ::testing::Values(SynapseMatrixType::SPARSE_GLOBALG, SynapseMatrixType::SPARSE_INDIVIDUALG,
                                SynapseMatrixType::DENSE_GLOBALG, SynapseMatrixType::DENSE_INDIVIDUALG,
                                SynapseMatrixType::BITMASK_GLOBALG)));