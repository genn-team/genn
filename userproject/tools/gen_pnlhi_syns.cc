/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file gen_pnlhi_syns.cc

\brief This file is part of a tool chain for running the classol/MBody1 example model.

This file compiles to a tool to generate appropriate connectivity patterns between PNs and LHIs (lateral horn interneurons) in the model. The connectivity is saved to file and can then be read by the classol method for reading this connectivity.
*/ 
//--------------------------------------------------------------------------


#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

int main(int argc, char *argv[])
{
  if (argc != 6)
  {
    cerr << "usage: gen_pnlhi_syns <nAL> <nLHI> ";
    cerr << "<pnlhi theta> ";
    cerr << "<pnlhi minact> ";
    cerr << "<outfile>" << endl;
    exit(1);
  }

  int nAL= atoi(argv[1]);
  int nLHI= atoi(argv[2]);
  double PNLHI_theta= atof(argv[3]);
  double PNLHI_minact= atof(argv[4]);
  ofstream os(argv[5], ios::binary);
  double *g= new double[nAL*nLHI];

  cerr << "# call was: ";
  for (int i= 0; i < argc; i++) cerr << argv[i] << " ";
  cerr << endl;

  for (int i= 0; i < nAL; i++) {
    for (int j= 0; j < nLHI; j++) {
      g[i*nLHI+j]= PNLHI_theta/(PNLHI_minact+j);
    }
  }
  
  os.write((char *)g, nAL*nLHI*sizeof(double));
  os.close();
  delete[] g;
  
  return 0;
}

