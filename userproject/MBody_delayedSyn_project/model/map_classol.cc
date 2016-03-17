/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Science
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#ifndef _MAP_CLASSOL_CC_
#define _MAP_CLASSOL_CC_ //!< macro for avoiding multiple inclusion during compilation


//--------------------------------------------------------------------------
/*! \file map_classol.cc

\brief Implementation of the classol class.
*/
//--------------------------------------------------------------------------

#include "map_classol.h"
#include "MBody_delayedSyn_CODE/runner.cc"


//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

classol::classol()
{
  modelDefinition(model);
  p_pattern= new scalar[model.neuronN[0]*PATTERNNO];
  pattern= new uint64_t[model.neuronN[0]*PATTERNNO];
  baserates= new uint64_t[model.neuronN[0]];
  allocateMem();
  initialize();
  sumPN= 0;
  sumKC= 0;
  sumLHI= 0;
  sumDN= 0;
  offsetPN= 0;
}

//--------------------------------------------------------------------------
/*! \brief Method for initialising variables
 */
//--------------------------------------------------------------------------

void classol::init(unsigned int which //!< Flag defining whether GPU or CPU only version is run
		   )
{
  offsetPN= 0;
  if (which == CPU) {
    ratesPN= baserates;
  }
  if (which == GPU) {
    ratesPN= d_baserates;
#ifndef CPU_ONLY
    copyStateToDevice();
#endif
  }
}

//--------------------------------------------------------------------------
/*! \brief Method for allocating memory on the GPU device to hold the input patterns
 */
//--------------------------------------------------------------------------

#ifndef CPU_ONLY
void classol::allocate_device_mem_patterns()
{
  unsigned int size;

  // allocate device memory for input patterns
  size= model.neuronN[0]*PATTERNNO*sizeof(uint64_t);
  CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_pattern, size));
  fprintf(stdout, "allocated %lu elements for pattern.\n", size/sizeof(uint64_t));
  CHECK_CUDA_ERRORS(cudaMemcpy(d_pattern, pattern, size, cudaMemcpyHostToDevice));
  size= model.neuronN[0]*sizeof(uint64_t);
  CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_baserates, size));
  CHECK_CUDA_ERRORS(cudaMemcpy(d_baserates, baserates, size, cudaMemcpyHostToDevice)); 
}

//--------------------------------------------------------------------------
/*! \brief Methods for unallocating the memory for input patterns on the GPU device
*/
//--------------------------------------------------------------------------

void classol::free_device_mem()
{
  // clean up memory
  CHECK_CUDA_ERRORS(cudaFree(d_pattern));
  CHECK_CUDA_ERRORS(cudaFree(d_baserates));
}
#endif


//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

classol::~classol()
{
  free(pattern);
  free(baserates);
  freeMem();
}

void classol::importArray(scalar *dest, double *src, int sz) 
{
    for (int i= 0; i < sz; i++) {
	dest[i]= (scalar) src[i];
    }
}

void classol::exportArray(double *dest, scalar *src, int sz) 
{
    for (int i= 0; i < sz; i++) {
	dest[i]= (scalar) src[i];
    }
}

//--------------------------------------------------------------------------
/*! \brief Method for reading the connectivity between PNs and KCs from a file
 */
//--------------------------------------------------------------------------

void classol::read_pnkcsyns(FILE *f //!< File handle for a file containing PN to KC conductivity values
			    )
{
  // version 1
    int sz= model.neuronN[0]*model.neuronN[1];
    double *tmpg= new double[sz];
    fprintf(stdout, "%lu\n", sz * sizeof(double));
    unsigned int retval = fread(tmpg, 1, sz * sizeof(double),f);
    importArray(gPNKC, tmpg, sz);
  // version 2
  /*  unsigned int UIntSz= sizeof(unsigned int)*8;   // in bit!
  unsigned int logUIntSz= (int) (logf((scalar) UIntSz)/logf(2.0f)+1e-5f);
  unsigned int tmp= model.neuronN[0]*model.neuronN[1];
  unsigned size= (tmp >> logUIntSz);
  if (tmp > (size << logUIntSz)) size++;
  size= size*sizeof(unsigned int);
  is.read((char *)gPNKC, size);*/

  // general:
  //assert(is.good());
  fprintf(stdout,"read pnkc ... \n");
  fprintf(stdout, "%u bytes, values start with: \n", retval);
  for(int i= 0; i < 20; i++) {
    fprintf(stdout, "%f ", gPNKC[i]);
  }
  fprintf(stdout,"\n\n");
    delete[] tmpg;
}

//--------------------------------------------------------------------------
/*! \brief Method for writing the conenctivity between PNs and KCs back into file
 */
//--------------------------------------------------------------------------


void classol::write_pnkcsyns(FILE *f //!< File handle for a file to write PN to KC conductivity values to
			     )
{
    int sz= model.neuronN[0]*model.neuronN[1];
    double *tmpg= new double[sz];
    exportArray(tmpg, gPNKC, sz);
    fwrite(tmpg, model.neuronN[0] * model.neuronN[1] * sizeof(double), 1, f);
    fprintf(stdout, "wrote pnkc ... \n");
    delete[] tmpg;
}


//--------------------------------------------------------------------------
/*! \brief Method for reading the connectivity between PNs and LHIs from a file
 */
//--------------------------------------------------------------------------

void classol::read_pnlhisyns(FILE *f //!< File handle for a file containing PN to LHI conductivity values
			     )
{
    int sz= model.neuronN[0]*model.neuronN[2];
    double *tmpg= new double[sz];
    unsigned int retval = fread(tmpg, 1, sz * sizeof(double),  f);
    importArray(gPNLHI, tmpg, sz);
    fprintf(stdout,"read pnlhi ... \n");
    fprintf(stdout, "%u bytes, values start with: \n", retval);
    for(int i= 0; i < 20; i++) {
	fprintf(stdout, "%f ", gPNLHI[i]);
    }
    fprintf(stdout, "\n\n");
    delete[] tmpg;
}

//--------------------------------------------------------------------------
/*! \brief Method for writing the connectivity between PNs and LHIs to a file
 */
//--------------------------------------------------------------------------

void classol::write_pnlhisyns(FILE *f //!< File handle for a file to write PN to LHI conductivity values to
			      )
{
    int sz= model.neuronN[0]*model.neuronN[2];
    double *tmpg= new double[sz];
    exportArray(tmpg, gPNLHI, sz);
    fwrite(tmpg, sz * sizeof(double), 1, f);
    fprintf(stdout, "wrote pnlhi ... \n");
    delete[] tmpg;
}


//--------------------------------------------------------------------------
/*! \brief Method for reading the connectivity between KCs and DNs (detector neurons) from a file
 */
//--------------------------------------------------------------------------

void classol::read_kcdnsyns(FILE *f //!< File handle for a file containing KC to DN (detector neuron) conductivity values 
			    )
{
    int sz= model.neuronN[1]*model.neuronN[3];
    double *tmpg= new double[sz];   
    unsigned int retval =fread(tmpg, 1, sz *sizeof(double),f);
    importArray(gKCDN, tmpg, sz);
    fprintf(stdout, "read kcdn ... \n");
    fprintf(stdout, "%u bytes, values start with: \n", retval);
    for(int i= 0; i < 100; i++) {
	fprintf(stdout, "%f ", gKCDN[i]);
    }
    fprintf(stdout, "\n\n");
    int cnt= 0;
    for (int i= 0; i < model.neuronN[1]*model.neuronN[3]; i++) {
	if (gKCDN[i] < 2.0*SCALAR_MIN){
	    cnt++;
	    fprintf(stdout, "Too low conductance value %e detected and set to 2*SCALAR_MIN= %e, at index %d \n", gKCDN[i], 2*SCALAR_MIN, i);
	    gKCDN[i] = 2.0*SCALAR_MIN; //to avoid log(0)/0 below
	}
	scalar tmp = gKCDN[i] / myKCDN_p[5]*2.0 ;
	gRawKCDN[i]=  0.5 * log( tmp / (2.0 - tmp)) /myKCDN_p[7] + myKCDN_p[6];
    }
    cerr << "Total number of low value corrections: " << cnt << endl;
    delete[] tmpg;
}


//--------------------------------------------------------------------------
/*! \brief Method to write the connectivity between KCs and DNs (detector neurons) to a file 
 */
//--------------------------------------------------------------------------

void classol::write_kcdnsyns(FILE *f //!< File handle for a file to write KC to DN (detectore neuron) conductivity values to
			     )
{
    int sz= model.neuronN[1]*model.neuronN[3];
    double *tmpg= new double[sz];
     exportArray(tmpg, gKCDN, sz);   
    fwrite(tmpg, sz*sizeof(double),1,f);
    fprintf(stdout, "wrote kcdn ... \n");
    delete[] tmpg;
}

void classol::read_sparsesyns_par(int synInd, SparseProjection C, scalar* g, FILE *f_ind,FILE *f_indInG, FILE *f_g //!< File handle for a file containing sparse conductivity values
    )
{
    //allocateSparseArray(synInd,C.connN);
    int sz= C.connN;
    double *tmpg= new double[sz];
    int retval = fread(tmpg, 1, sz*sizeof(double),f_g);
    importArray(g, tmpg, sz);
    fprintf(stdout,"%d active synapses. \n", sz);
    retval = fread(C.indInG, 1, (model.neuronN[model.synapseSource[synInd]]+1)*sizeof(unsigned int),f_indInG);
    retval = fread(C.ind, 1, sz*sizeof(int),f_ind);
    // general:
    fprintf(stdout,"Read sparse projection indices ... \n");
    fprintf(stdout, "Size is %d for synapse group %d. Values start with: \n",C.connN, synInd);
    for(int i= 0; i < 100; i++) {
	fprintf(stdout, "%d, %d ", C.indInG[i], C.ind[i]);
    }
    fprintf(stdout,"\n\n");
    fprintf(stdout, "%d indices read. Index values start with: \n",C.connN);
    for(int i= 0; i < 100; i++) {
	fprintf(stdout, "%d ", C.ind[i]);
    }  
    fprintf(stdout,"\n\n");
    fprintf(stdout, "%d g indices read. Index in g array values start with: \n", model.neuronN[model.synapseSource[synInd]]+1);
    for(int i= 0; i < 100; i++) {
	fprintf(stdout, "%d ", C.indInG[i]);
    }  
    delete[] tmpg;
}

//--------------------------------------------------------------------------
/*! \brief Method for reading the input patterns from a file
 */
//--------------------------------------------------------------------------

void classol::read_input_patterns(FILE *f //!< File handle for a file containing input patterns
				  )
{
  // we use a predefined pattern number
    int sz= model.neuronN[0]*PATTERNNO;
    double *tmpp= new double[sz];
    unsigned int retval = fread(tmpp, 1, sz*sizeof(double),f);
    importArray(p_pattern, tmpp, sz);
    fprintf(stdout, "read patterns ... \n");
    fprintf(stdout, "%u bytes, input pattern values start with: \n", retval);
    for(int i= 0; i < 100; i++) {
	fprintf(stdout, "%f ", p_pattern[i]);
    }
    fprintf(stdout, "\n\n");
    convertRateToRandomNumberThreshold(p_pattern, pattern, model.neuronN[0]*PATTERNNO);
    delete[] tmpp;
}

//--------------------------------------------------------------------------
/*! \brief Method for calculating the baseline rates of the Poisson input neurons
 */
//--------------------------------------------------------------------------

void classol::generate_baserates()
{
  // we use a predefined pattern number
    uint64_t inputBase;
    convertRateToRandomNumberThreshold(&InputBaseRate, &inputBase, 1);
    for (int i= 0; i < model.neuronN[0]; i++) {
	baserates[i]= inputBase;
    }
    fprintf(stdout, "generated basereates ... \n");
    fprintf(stdout, "baserate value: %f ", InputBaseRate);
    fprintf(stdout, "\n\n");  
}

//--------------------------------------------------------------------------
/*! \brief Method for simulating the model for a given period of time on the GPU
 */
//--------------------------------------------------------------------------

#ifndef CPU_ONLY
void classol::runGPU(scalar runtime //!< Duration of time to run the model for 
		  )
{
  unsigned int pno;
  int riT= (int) (runtime/DT+1e-6);

  for (int i= 0; i < riT; i++) {
      unsigned int flags= 0;
      if (iT%patSetTime == 0) {
	  pno= (iT/patSetTime)%PATTERNNO;
	  ratesPN= d_pattern;
	  offsetPN= pno*model.neuronN[0];
	  flags= GENN_FLAGS::COPY;
      }
      if (iT%patSetTime == patFireTime) {
	  ratesPN= d_baserates;
	  offsetPN= 0;
	  flags= GENN_FLAGS::COPY;
      }
      stepTimeGPU(flags);
  }
}
#endif

//--------------------------------------------------------------------------
/*! \brief Method for simulating the model for a given period of time on the CPU
 */
//--------------------------------------------------------------------------

void classol::runCPU(scalar runtime //!< Duration of time to run the model for 
		  )
{
  unsigned int pno;
  int riT= (int) (runtime/DT);

  for (int i= 0; i < riT; i++) {
    if (iT%patSetTime == 0) {
      pno= (iT/patSetTime)%PATTERNNO;
      ratesPN= pattern;
      offsetPN= pno*model.neuronN[0];
    }
    if (iT%patSetTime == patFireTime) {
	ratesPN= baserates;
	offsetPN= 0;
    }

    stepTimeCPU();
  }
}
//--------------------------------------------------------------------------
// output functions

//--------------------------------------------------------------------------
/*! \brief Method for copying from device and writing out to file of the entire state of the model
 */
//--------------------------------------------------------------------------

void classol::output_state(FILE *f, //!< File handle for a file to write the model state to 
			   unsigned int which //!< Flag determining whether using GPU or CPU only
			   )
{
  if (which == GPU) 
#ifndef CPU_ONLY
    copyStateFromDevice();
#endif
  fprintf(f, "%f ", t);
  for (int i= 0; i < model.neuronN[0]; i++) {
    fprintf(f, "%f ", VPN[i]);
   }
  // for (int i= 0; i < model.neuronN[0]; i++) {
  //   os << seedsPN[i] << " ";
  // }
  // for (int i= 0; i < model.neuronN[0]; i++) {
  //   os << spikeTimesPN[i] << " ";
  // }
  //  os << glbSpkCnt0 << "  ";
  // for (int i= 0; i < glbSpkCntPN; i++) {
  //   os << glbSpkPN[i] << " ";
  // }
  // os << " * ";
  // os << glbSpkCntKC << "  ";
  // for (int i= 0; i < glbSpkCntKC; i++) {
  //   os << glbSpkKC[i] << " ";
  // }
  // os << " * ";
  // os << glbSpkCntLHI << "  ";
  // for (int i= 0; i < glbSpkCntLHI; i++) {
  //   os << glbSpkLHI[i] << " ";
  // }
  // os << " * ";
  // os << glbSpkCntDN << "  ";
  // for (int i= 0; i < glbSpkCntDN; i++) {
  //   os << glbSpkDN[i] << " ";
  // }
 //   os << " * ";
 // for (int i= 0; i < 20; i++) {
 //   os << VKC[i] << " ";
 // }
   for (int i= 0; i < model.neuronN[1]; i++) {
     fprintf(f, "%f ", VKC[i]);
   }
  //os << " * ";
  //for (int i= 0; i < model.neuronN[1]; i++) {
  //  os << inSynKC0[i] << " ";
  //}
  //  os << " * ";
  //  for (int i= 0; i < 20; i++) {
  //    os << inSynKC1[i] << " ";
  //  }
  //  os << " * ";
  //  for (int i= 0; i < 20; i++) {
  //    os << inSynLHI0[i] << " ";
  //  }
  //  os << " * ";
  //  for (int i= 0; i < model.neuronN[3]; i++) {
  //    os << inSynDN0[i] << " ";
  //  }
  //  os << endl;
  //  os << " * ";
  //  for (int i= 0; i < model.neuronN[3]; i++) {
  //    os << inSynDN1[i] << " ";
  //  }
  for (int i= 0; i < model.neuronN[2]; i++) {
    fprintf(f, "%f ", VLHI[i]);
  }
  for (int i= 0; i < model.neuronN[3]; i++) {
    fprintf(f, "%f ", VDN[i]);
  }
  fprintf(f,"\n");
}

//--------------------------------------------------------------------------
/*! \brief Method for copying all spikes of the last time step from the GPU
 
  This is a simple wrapper for the convenience function copySpikesFromDevice() which is provided by GeNN.
*/
//--------------------------------------------------------------------------

void classol::getSpikesFromGPU()
{
#ifndef CPU_ONLY
  copySpikesFromDevice();
#endif
}

//--------------------------------------------------------------------------
/*! \brief Method for copying the number of spikes in all neuron populations that have occurred during the last time step
 
This method is a simple wrapper for the convenience function copySpikeNFromDevice() provided by GeNN.
*/
//--------------------------------------------------------------------------

void classol::getSpikeNumbersFromGPU() 
{
#ifndef CPU_ONLY
  copySpikeNFromDevice();
#endif
}

//--------------------------------------------------------------------------
/*! \brief Method for writing the spikes occurred in the last time step to a file
 */
//--------------------------------------------------------------------------

void classol::output_spikes(FILE *f, //!< File handle for a file to write spike times to
			    unsigned int which //!< Flag determining whether using GPU or CPU only
			    )
{

  //    fprintf(stdout, "%f %f %f %f %f\n", t, glbSpkCntPN, glbSpkCntKC, glbSpkCntLHI,glbSpkCntDN);
  for (int i= 0; i < glbSpkCntPN[0]; i++) {
    fprintf(f, "%f %d\n", t, glbSpkPN[i]);
  }
  for (int i= 0; i < glbSpkCntKC[0]; i++) {
    fprintf(f,  "%f %d\n", t, model.sumNeuronN[0]+glbSpkKC[i]);
  }
  for (int i= 0; i < glbSpkCntLHI[0]; i++) {
    fprintf(f, "%f %d\n", t, model.sumNeuronN[1]+glbSpkLHI[i]);
  }
  for (int i= 0; i < glbSpkCntDN[0]; i++) {
    fprintf(f, "%f %d\n", t, model.sumNeuronN[2]+glbSpkDN[i]);
  }
}

//--------------------------------------------------------------------------
/*! \brief Method for summing up spike numbers
 */
//--------------------------------------------------------------------------

void classol::sum_spikes()
{
  sumPN+= glbSpkCntPN[0];
  sumKC+= glbSpkCntKC[0];
  sumLHI+= glbSpkCntLHI[0];
  sumDN+= glbSpkCntDN[0];
}

//--------------------------------------------------------------------------
/*! \brief Method for copying the synaptic conductances of the learning synapses between KCs and DNs (detector neurons) back to the CPU memory
 */
//--------------------------------------------------------------------------

void classol::get_kcdnsyns()
{
#ifndef CPU_ONLY
    CHECK_CUDA_ERRORS(cudaMemcpy(gKCDN, d_gKCDN, model.neuronN[1]*model.neuronN[3]*sizeof(scalar), cudaMemcpyDeviceToHost));
#endif
}



#endif	


