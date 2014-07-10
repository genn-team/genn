/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Informatics
              University of Sussex 
              Brighton BN1 9QJ, UK
  
   email to:  t.nowotny@sussex.ac.uk
  
   initial version: 2014-06-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file VClampGA.cu

\brief Main entry point for the GeNN project demonstrating realtime fitting of a neuron with a GA running mostly on the GPU. 
*/
//--------------------------------------------------------------------------

#include "VClampGA.h"

//--------------------------------------------------------------------------
/*! \brief This function is the entry point for running the project
*/
//--------------------------------------------------------------------------

int main(int argc, char *argv[])
{
  if (argc != 4)
  {
    fprintf(stderr, "usage: VClampGA <basename> <CPU=0, GPU=1> <protocol> \n");
    return 1;
  }
  int which= atoi(argv[2]);
  int protocol= atoi(argv[3]);    
  string OutDir = toString(argv[1]) +"_output";
  string name;
  name= OutDir+ "/"+ toString(argv[1]) + toString(".time");
  FILE *timef= fopen(name.c_str(),"a");  

  timer.startTimer();
  write_para();
  
  name= OutDir+ "/"+ toString(argv[1]) + toString(".out.I"); 
  FILE *osf= fopen(name.c_str(),"w");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".out.err"); 
  FILE *ose= fopen(name.c_str(),"w");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".out.best"); 
  FILE *osb= fopen(name.c_str(),"w");

  //-----------------------------------------------------------------
  // build the neuronal circuitery

  NNmodel model;
  modelDefinition(model);
  allocateMem();
  initialize();
  var_reinit(1.0);         // this includes copying vars for the GPU version
  initexpHH();
  fprintf(stderr, "# neuronal circuitery built, start computation ... \n\n");
  
  double *theExp_p[7];
  theExp_p[0]= &gNaexp;
  theExp_p[1]= &ENaexp;
  theExp_p[2]= &gKexp;
  theExp_p[3]= &EKexp;
  theExp_p[4]= &glexp;
  theExp_p[5]= &Elexp;
  theExp_p[6]= &Cexp;

  //------------------------------------------------------------------
  // output general parameters to output file and start the simulation

  fprintf(stderr, "# We are running with fixed time step %f \n", DT);

  int done= 0, sn;
  unsigned int VSize= NPOP*sizeof(double);
  double lt, oldt;
  inputSpec I;
  initI(I);
  stepVGHH= I.baseV;
  int iTN= (int) (I.t/DT);

  t= 0.0;
  while (!done) 
  {
    truevar_init();
    truevar_initexpHH();
    lt= 0.0;
    sn= 0;	
    for (int iT= 0; iT < iTN; iT++) {
      oldt= lt;
      runexpHH(t); 
      if (which == GPU) {
	stepTimeGPU(t);
      }
      else {
	stepTimeCPU(t);
      }
      t+= DT;	
      lt+= DT;
      // if (which == 1) {       
      //   CHECK_CUDA_ERRORS(cudaMemcpy(VHH, d_VHH, VSize, cudaMemcpyDeviceToHost));
      // }
      // fprintf(osf,"%f %f %f ", t, stepVGHH, IsynGHH);
      // for (int i= 0; i < NPOP; i++) {
      //     fprintf(osf, "%f ", 1000.0*(stepVGHH-VHH[i]));
      // }
      // fprintf(osf, "\n");
      if ((sn < I.N) && (oldt < I.st[sn]) && (lt >= I.st[sn])) {
	stepVGHH= I.V[sn];
	sn++;
      }
    }
    if (which == 1) {
       CHECK_CUDA_ERRORS(cudaMemcpy(errHH, d_errHH, VSize, cudaMemcpyDeviceToHost));
    }   
    fprintf(ose,"%f ", t);
    for (int i= 0; i < NPOP; i++) {
       fprintf(ose, "%f ", errHH[i]);
    }
    fprintf(ose,"\n");
    fprintf(osb, "%f %f %f %f %f %f %f ", gNaexp, ENaexp, gKexp, EKexp, glexp, Elexp, Cexp);
    procreatePop(osb);
    if (protocol < 7) {
       *(theExp_p[protocol])=  myHH_ini[protocol+4]+40*sin(3.1415927*t/40000);
    }
    else {    
       if (protocol == 7) {
	  int pn= (int) (R.n()*6.0);
	  double fac;
	  if (pn%2 == 0) {
	       fac= 1+0.04*RG.n();
	   *(theExp_p[pn])*= fac;
	  }
	  else {
	     fac= RG.n();
	       *(theExp_p[pn])+= fac;
	  }
       }
    }
    cerr << "% " << t << endl;
    done= (t >= TOTALT);
  }
  timer.stopTimer();
  fprintf(timef,"%f \n",timer.getElapsedTime());  
  // close files 
  fclose(osf);
  fclose(ose);
  fclose(timef);
  fclose(osb);
  freeDeviceMem();
  cudaDeviceReset();
  return 0;
}
