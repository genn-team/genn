/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Science
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file gen_pnkc_syns_indivID.cc

\brief This file is part of a tool chain for running the classol/MBody1 example model.

This file compiles to a tool to generate appropriate connectivity patterns between PNs and KCs in the model. In contrast to the gen_pnkc_syns.cc tool, here the output is in a format that is suited for the "INDIVIDUALID" method for specifying connectivity. The connectivity is saved to file and can then be read by the classol method for reading this connectivity.
*/ 
//--------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cstdint>

using namespace std;

#include "randomGen.h"
#include "gauss.h"
#include "simpleBit.h"

#include "randomGen.cc"

randomGen R;
randomGauss RG;

int main(int argc, char *argv[])
{
  if (argc != 5)
  {
    cerr << "usage: gen_pnkc_syns_indivID <nAL> <nMB> ";
    cerr << "<prob. of PN-KC syn> ";
    cerr << "<outfile>" << endl;
    exit(1);
  }

  int nAL= atoi(argv[1]);
  int nMB= atoi(argv[2]);
  double psyn= atof(argv[3]);
  ofstream os(argv[4], ios::binary);
  unsigned int size= nAL*nMB;
  size= (size/32+1)*sizeof(uint32_t);
  uint32_t *g= new uint32_t[size];

  cerr << "# call was: ";
  for (int i= 0; i < argc; i++) cerr << argv[i] << " ";
  cerr << endl;

  for (unsigned int i= 0; i < size; i++) {
    g[i]= 0;
  }
  
  for (int i= 0; i < nAL; i++) {
    for (int j= 0; j < nMB; j++) {
      if (R.n() < psyn) {
	setB(g[(i*nMB+j) >> 5], (i*nMB+j)&31);
	//	cerr << ((i*nMB+j) >> 5) << " " << (i*nMB+j)&31 << " ";
	//	cerr << g[(i*nMB+j) >> 5] << endl;
      }
    }
  }
  
  os.write((char *)g, size);
  os.close();
  delete[] g;
  
  return 0;
}

