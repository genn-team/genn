//--------------------------------------------------------------------------
//   Author:    James Turner
//  
//   Institute: Center for Computational Neuroscience and Robotics
//              University of Sussex
//              Falmer, Brighton BN1 9QJ, UK 
//  
//   email to:  J.P.Turner@sussex.ac.uk
//  
//--------------------------------------------------------------------------

#ifndef SYNDELAYSIM_HPP
#define SYNDELAYSIM_HPP

#define DT 1.0f
#define TOTAL_TIME 5000.0f
#define REPORT_TIME 1000.0f

class SynDelay
{
private:
  bool usingGPU;

public:
  SynDelay(bool usingGPU);
  ~SynDelay();
  void run(float t);
};

#endif // SYNDELAYSIM_HPP
