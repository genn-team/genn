#include <functional>
#include <numeric>

// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "extra_global_parameters_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"

//----------------------------------------------------------------------------
// ExtraGlobalParametersTest
//----------------------------------------------------------------------------
class ExtraGlobalParametersTest : public SimulationTest
{
protected:
  //--------------------------------------------------------------------------
  // SimulationTest virtuals
  //--------------------------------------------------------------------------
  virtual void Init()
  {
    // Initialise neuron parameters
    for (int i = 0; i < 10; i++)
    {
      shiftpre[i] = i * 10.0f;
    }
  }
};

TEST_P(ExtraGlobalParametersTest, AcceptableError)
{
  float err = 0.0f;
  inputpre = 0.0f;
  for (int i = 0; i < (int)(20.0f / DT); i++)
  {
    // for all pre-synaptic neurons
    float x[10];
    for (int j= 0; j < 10; j++)
    {
        // generate expected values
        if (i > 0)
        {
          x[j]= (t - DT) + pow(t - DT, 2.0) + (j * 10);
        }
        else
        {
          x[j] = 0.0f;
        }
        /*if (write)
        {
            neurOs << xpre[glbSpkShiftpre+j] << " ";
            expNeurOs << x[j] << " ";
        }*/
    }

    // Add error for this time step to total
    err += std::inner_product(&x[0], &x[10],
                              &xpre[glbSpkShiftpre],
                              0.0,
                              std::plus<float>(),
                              [](double a, double b){ return abs(a - b); });

    //err += absDiff(x, xpre + glbSpkShiftpre, 10);
    /*if (write)
    {
        neurOs << endl;
        expNeurOs << endl;
    }*/

    // Update global
    inputpre = pow(t, 2.0);

    // Step simulation
    Step();

    /*if (fmod(t+5e-5, REPORT_TIME) < 1e-4)
    {
      cout << "\r" << t;
    }*/
  }
  /*cout << "\r";
  cout << "# done in " << timer->getElapsedTime() << " seconds" << endl;
  if (write)
  {
      timeOs << timer->getElapsedTime() << endl;
      timeOs.close();
      neurOs.close();
      expNeurOs.close();
  }*/

  // Check total error is less than some tolerance
  EXPECT_LT(err, 2e-2);
}

#ifndef CPU_ONLY
auto simulatorBackends = ::testing::Values(true, false);
#else
auto simulatorBackends = ::testing::Values(false);
#endif

INSTANTIATE_TEST_CASE_P(Backends,
                        ExtraGlobalParametersTest,
                        simulatorBackends);