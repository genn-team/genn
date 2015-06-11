/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file gen_kcdn_syns.cc

\brief This file is part of a tool chain for running the classol/MBody1 example model.

This file compiles to a tool to generate appropriate connectivity patterns between KCs and DNs (detector neurons) in the model. The connectivity is saved to file and can then be read by the classol method for reading this connectivity.
*/ 
//--------------------------------------------------------------------------


#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

#include "randomGen.h"
#include "gauss.h"

#include "randomGen.cc"

randomGen R, R2;
randomGauss RG;

int main(int argc, char *argv[])
{
  cout << "generating KCDN weights..." << endl;
  if (argc != 7)
  {
    cerr << "usage: gen_kcdn_syns <nMB> <nLobes> ";
    cerr << "<mean strength> ";
    cerr << "<strength jitter> <min strength> <outfile>" << endl;
    exit(1);
  }

  int nMB= atoi(argv[1]);
  int nLB= atoi(argv[2]);
  double kcdn_gsyn= atof(argv[3]);
  double kcdn_jitter= atof(argv[4]);
  double EPS= atof(argv[5]);
  ofstream os(argv[6],ios::binary);
  double gsyn;
  double *g= new double[nMB*nLB];

  cout << "# call was: ";
  for (int i= 0; i < argc; i++) cerr << argv[i] << " ";

  double psyn = 10000.0/nMB;
  cout << "psyn = " << psyn << endl;
  int smallctr = 0;
  for (int i= 0; i < nMB; i++) {
    for (int j= 0; j < nLB; j++) {
      if (psyn > 1.0){  
        gsyn= kcdn_gsyn+kcdn_jitter*RG.n();
      }
      else{
        if (R2.n()<psyn){
	    gsyn= kcdn_gsyn/psyn+kcdn_jitter*sqrt((double) nMB)/100.0*RG.n(); // scaling stddev by sqrt(n/10000)
        }
        else gsyn = 1.0e-25;
      }
      if (gsyn < EPS) {
        //cout <<  "Rand number smaller than min. at "<< i << " " << j << "... Setting to min..." << endl;
        gsyn= EPS;
        smallctr++;
      }
      g[i*nLB+j]= gsyn;
    }
  }

  cout << endl;
  os.write((char *)g, nMB*nLB*sizeof(double));
  os.close();
  delete[] g;
  cout << "KCDN weights are created for " << nMB << " KC's and " << nLB << " DN's with mean = " << kcdn_gsyn << ", stddev = " << kcdn_jitter << endl;
  cout.precision(3);
  if (smallctr >0 ) cout << smallctr << " (% " << fixed << 100*double(smallctr)/(nMB*nLB) << ") of the weights were too small or negative, so they are set to " << scientific << EPS << endl;
  cout << endl; 
  return 0;
}

