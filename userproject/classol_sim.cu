/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#include "classol_sim.h"

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    cerr << "usage: classol_sim <basename> <CPU=0, GPU=1" << endl;
    return 1;
  }
  int which= atoi(argv[2]);
  string name;
  name= toString(argv[1]) + toString(".time");
  ofstream timeos(name.c_str());  

  timer.startTimer();
  
  cerr << "# DT " << DT << endl;
  cerr << "# T_REPORT_TME " << T_REPORT_TME << endl;
  cerr << "# SYN_OUT_TME " << SYN_OUT_TME << endl;
  cerr << "# PATFTIME " << PATFTIME << endl; 
  cerr << "# PAT_FIRETIME " << PAT_FIRETIME << endl;
  cerr << "# PAT_TIME " << PAT_TIME << endl;
  cerr << "# PAT_SETTIME " << PAT_SETTIME << endl;
  cerr << "# TOTAL_TME " << TOTAL_TME << endl;
  
  name= toString(argv[1]) + toString(".out.st"); 
  ofstream os(name.c_str());

  //-----------------------------------------------------------------
  // build the neuronal circuitery
  classol locust;

  cerr << "# reading PN-KC synapses ..." << endl;
  name= toString(argv[1]) + toString(".pnkc");
  ifstream pnkcsis(name.c_str(), ios::binary);
  assert(pnkcsis.good());	   
  locust.read_pnkcsyns(pnkcsis);   
 
  cerr << "# reading PN-LHI synapses ..." << endl;
  name= toString(argv[1]) + toString(".pnlhi");
  ifstream pnlhisis(name.c_str(), ios::binary);
  assert(pnlhisis.good());	   
  locust.read_pnlhisyns(pnlhisis);   
  
  cerr << "# reading KC-DN synapses ..." << endl;
  name= toString(argv[1]) + toString(".kcdn");
  ifstream kcdnsis(name.c_str(), ios::binary);
  assert(kcdnsis.good());
  locust.read_kcdnsyns(kcdnsis);
   
  cerr << "# reading input patterns ..." << endl;
  name= toString(argv[1]) + toString(".inpat");
  ifstream patis(name.c_str(), ios::binary);
  assert(patis.good());
  locust.read_input_patterns(patis);
  locust.generate_baserates();

  if (which == GPU) {
    locust.allocate_device_mem_patterns();
  }
  locust.init(which);         // this includes copying g's for the GPU version

  cerr << "# neuronal circuitery built, start computation ..." << endl << endl;

  //------------------------------------------------------------------
  // output general parameters to output file and start the simulation

  cerr << "# We are running with fixed time step " << DT << endl;
  cerr << "# initial wait time execution ... " << endl;

  t= 0.0;
  void *devPtr;
//  float lastsynwrite= t;
  int done= 0;
  float last_t_report=  t;
//  locust.output_state(os, which);  
//  locust.output_spikes(os, which);  
  locust.run(DT, which);
//locust.output_state(os, which);  
//  float synwriteT= 0.0f;
//  int synwrite= 0;
unsigned int sum= 0;
	 os.precision(10);
  while (!done) 
  {
//   if (which == GPU) locust.getSpikesFromGPU();
//    if (which == GPU) locust.getSpikeNumbersFromGPU();
    locust.run(DT, which); // run next batch
    if (which == GPU) {  
      cudaGetSymbolAddress(&devPtr, "d_VDN");
      CUDA_SAFE_CALL(cudaMemcpy(VDN, devPtr, 10*sizeof(float), cudaMemcpyDeviceToHost));
    }
//    locust.sum_spikes();
//    locust.output_spikes(os, which);
//   locust.output_state(os, which);  // while outputting the current one ...
   os << t << " ";
   for (int i= 0; i < 10; i++) {
     os << VDN[i] << " ";
   }
   os << endl;
//      cudaThreadSynchronize();

    // report progress
    if (t - last_t_report >= T_REPORT_TME)
    {
      cerr << "time " << t << endl;
      last_t_report= t;
      //locust.output_state(os);
    }
    // output synapses occasionally
    // if (synwrite) {
    //   lastsynwrite= synwriteT;
    //   name= toString(argv[1]) + toString(".") + toString((int) synwriteT);
    //   name+= toString(".syn");
    //   ofstream kcdnsynos(name.c_str());
    //   locust.write_kcdnsyns(kcdnsynos);
    //   synwrite= 0;
    // }
    // if (t - lastsynwrite >= SYN_OUT_TME) {
    //   locust.get_kcdnsyns();
    //   synwrite= 1;
    //   synwriteT= t;
    // }
    done= (t >= TOTAL_TME);
  }
//  locust.output_state(os);
//    if (which == GPU) locust.getSpikesFromGPU();
//    locust.output_spikes(os, which);
  // if (synwrite) {
  //   lastsynwrite= t;
  //   name= toString(argv[1]) + toString(".") + toString((int) t);
  //   name+= toString(".syn");
  //   ofstream kcdnsynos(name.c_str());
  //   locust.write_kcdnsyns(kcdnsynos);
  //   synwrite= 0;
  // }

  timer.stopTimer();
  timeos << locust.sumPN << " ";
  timeos << locust.sumKC << " ";
  timeos << locust.sumLHI << " ";
  timeos << locust.sumDN << " ";
  timeos << " " << timer.getElapsedTime() << endl;

  return 0;
}
