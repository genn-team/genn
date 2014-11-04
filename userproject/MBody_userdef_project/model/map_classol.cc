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
#include "MBody_userdef_CODE/runner.cc"

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

classol::classol()
{
  modelDefinition(model);
  pattern= new unsigned int[model.neuronN[0]*PATTERNNO];
  baserates= new unsigned int[model.neuronN[0]];
  allocateMem();
  initialize();
  sumPN= 0;
  sumKC= 0;
  sumLHI= 0;
  sumDN= 0;
}

//--------------------------------------------------------------------------
/*! \brief Method for initialising variables
 */
//--------------------------------------------------------------------------

void classol::init(unsigned int which //!< Flag defining whether GPU or CPU only version is run
		   )
{
  //initGRaw();
  //following is to initialise user-defined learn1synapse graw variable. Equivalent to initGRaw();
  //for (int i=0;i<gCKDN.connN;i++){ //sparse
 
  for (int i=0;i<gPNKC.connN;i++){ //alltoall
    gRawPNKC[i]=0.5; 
  }
 
  for (int i=0;i<model.neuronN[0]*model.neuronN[2];i++){ //alltoall
    gRawPNLHI[i]=0.2; 
  }

  for (int i=0;i<model.neuronN[1]*model.neuronN[3];i++){ //alltoall
    float tmp = gpKCDN[i] / (model.synapsePara[3][6]*2.0) - 1.0;
    gRawKCDN[i]=0.5 * log((1.0 + tmp) / (1.0 - tmp)) /model.synapsePara[3][8] + model.synapsePara[3][7]; 
  }
  
  initializeAllSparseArrays();

  if (which == CPU) {
    theRates= baserates;
  }
  if (which == GPU) {
    copyGToDevice(); 
    copyStateToDevice();
    theRates= d_baserates;
  }
}

//--------------------------------------------------------------------------
/*! \brief Method for allocating memory on the GPU device to hold the input patterns
 */
//--------------------------------------------------------------------------

void classol::allocate_device_mem_patterns()
{
  unsigned int size;

  // allocate device memory for input patterns
  size= model.neuronN[0]*PATTERNNO*sizeof(unsigned int);
  CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_pattern, size));
  fprintf(stdout, "allocated %lu elements for pattern.\n", size/sizeof(unsigned int));
  CHECK_CUDA_ERRORS(cudaMemcpy(d_pattern, pattern, size, cudaMemcpyHostToDevice));
  size= model.neuronN[0]*sizeof(unsigned int);
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


//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

classol::~classol()
{
  free(pattern);
  free(baserates);
  freeMem();
}


//--------------------------------------------------------------------------
/*! \brief Method for reading the connectivity between PNs and KCs from a file
 */
//--------------------------------------------------------------------------

void classol::read_pnkcsyns(FILE *f //!< File handle for a file containing PN to KC conductivity values
			    )
{
  // version 1
  fprintf(stdout, "%lu\n", model.neuronN[0]*model.neuronN[1]*sizeof(float));
  unsigned int retval = fread(gpPNKC, 1, model.neuronN[0]*model.neuronN[1]*sizeof(float),f);
  // version 2
  /*  unsigned int UIntSz= sizeof(unsigned int)*8;   // in bit!
  unsigned int logUIntSz= (int) (logf((float) UIntSz)/logf(2.0f)+1e-5f);
  unsigned int tmp= model.neuronN[0]*model.neuronN[1];
  unsigned size= (tmp >> logUIntSz);
  if (tmp > (size << logUIntSz)) size++;
  size= size*sizeof(unsigned int);
  is.read((char *)gpPNKC, size);*/

  // general:
  //assert(is.good());
  fprintf(stdout,"read pnkc ... \n");
  fprintf(stdout, "%u bytes, values start with: \n", retval);
  for(int i= 0; i < 20; i++) {
    fprintf(stdout, "%f ", gpPNKC[i]);
  }
  fprintf(stdout,"\n\n");
}

//--------------------------------------------------------------------------
/*! \brief Method for writing the conenctivity between PNs and KCs back into file
 */
//--------------------------------------------------------------------------


void classol::write_pnkcsyns(FILE *f //!< File handle for a file to write PN to KC conductivity values to
			     )
{
  fwrite(gpPNKC, model.neuronN[0] * model.neuronN[1] * sizeof(float), 1, f);
  fprintf(stdout, "wrote pnkc ... \n");
}


//--------------------------------------------------------------------------
/*! \brief Method for reading the connectivity between PNs and LHIs from a file
 */
//--------------------------------------------------------------------------

void classol::read_pnlhisyns(FILE *f //!< File handle for a file containing PN to LHI conductivity values
			     )
{
  unsigned int retval = fread(gpPNLHI, 1, model.neuronN[0] * model.neuronN[2] * sizeof(float),  f);
  fprintf(stdout,"read pnlhi ... \n");
  fprintf(stdout, "%u bytes, values start with: \n", retval);
  for(int i= 0; i < 20; i++) {
    fprintf(stdout, "%f ", gpPNLHI[i]);
  }
  fprintf(stdout, "\n\n");
}

//--------------------------------------------------------------------------
/*! \brief Method for writing the connectivity between PNs and LHIs to a file
 */
//--------------------------------------------------------------------------

void classol::write_pnlhisyns(FILE *f //!< File handle for a file to write PN to LHI conductivity values to
			      )
{
  fwrite(gpPNLHI, model.neuronN[0] * model.neuronN[2] * sizeof(float), 1, f);
  fprintf(stdout, "wrote pnlhi ... \n");
}


//--------------------------------------------------------------------------
/*! \brief Method for reading the connectivity between KCs and DNs (detector neurons) from a file
 */
//--------------------------------------------------------------------------

void classol::read_kcdnsyns(FILE *f //!< File handle for a file containing KC to DN (detector neuron) conductivity values 
			    )
{
  unsigned int retval =fread(gpKCDN, 1, model.neuronN[1]*model.neuronN[3]*sizeof(float),f);
  fprintf(stdout, "read kcdn ... \n");
  fprintf(stdout, "%u bytes, values start with: \n", retval);
  for(int i= 0; i < 100; i++) {
    fprintf(stdout, "%f ", gpKCDN[i]);
  }
  fprintf(stdout, "\n\n");
}


//--------------------------------------------------------------------------
/*! \brief Method to write the connectivity between KCs and DNs (detector neurons) to a file 
 */
//--------------------------------------------------------------------------

void classol::write_kcdnsyns(FILE *f //!< File handle for a file to write KC to DN (detectore neuron) conductivity values to
			     )
{
  fwrite(gpKCDN, model.neuronN[1]*model.neuronN[3]*sizeof(float),1,f);
  fprintf(stdout, "wrote kcdn ... \n");
}

void classol::read_sparsesyns_par(int synInd, Conductance C, FILE *f_ind,FILE *f_indInG,FILE *f_g //!< File handle for a file containing sparse conductivity values
			    )
{
  //allocateSparseArray(synInd,C.connN);

  unsigned int retval =fread(C.gp, 1, C.connN*sizeof(float),f_g);
  fprintf(stdout,"%d active synapses. \n",C.connN);
  retval = fread(C.gIndInG, 1, (model.neuronN[model.synapseSource[synInd]]+1)*sizeof(unsigned int),f_indInG);
  retval = fread(C.gInd, 1, C.connN*sizeof(int),f_ind);

  
  // general:
  fprintf(stdout,"Read conductance ... \n");
  fprintf(stdout, "Size is %d for synapse group %d. Values start with: \n",C.connN, synInd);
  for(int i= 0; i < 100; i++) {
    fprintf(stdout, "%f ", C.gp[i]);
  }
  fprintf(stdout,"\n\n");
  
  
  fprintf(stdout, "%d indices read. Index values start with: \n",C.connN);
  for(int i= 0; i < 100; i++) {
    fprintf(stdout, "%d ", C.gInd[i]);
  }  
  fprintf(stdout,"\n\n");
  
  
  fprintf(stdout, "%d g indices read. Index in g array values start with: \n", model.neuronN[model.synapseSource[synInd]]+1);
  for(int i= 0; i < 100; i++) {
    fprintf(stdout, "%d ", C.gIndInG[i]);
  }  
}

//--------------------------------------------------------------------------
/*! \brief Method for reading the input patterns from a file
 */
//--------------------------------------------------------------------------

void classol::read_input_patterns(FILE *f //!< File handle for a file containing input patterns
				  )
{
  // we use a predefined pattern number
  unsigned int retval = fread(p_pattern, 1, model.neuronN[0]*PATTERNNO*sizeof(float),f);
  fprintf(stdout, "read patterns ... \n");
  fprintf(stdout, "%u bytes, input pattern values start with: \n", retval);
  for(int i= 0; i < 100; i++) {
    fprintf(stdout, "%f ", p_pattern[i]);
  }
  fprintf(stdout, "\n\n");
}

//--------------------------------------------------------------------------
/*! \brief Method for calculating the baseline rates of the Poisson input neurons
 */
//--------------------------------------------------------------------------

void classol::generate_baserates()
{
  // we use a predefined pattern number
  for (int i= 0; i < model.neuronN[0]; i++) {
    baserates[i]= INPUTBASERATE;
  }
  fprintf(stdout, "generated basereates ... \n");
  fprintf(stdout, "baserate value: %d ", INPUTBASERATE);
  fprintf(stdout, "\n\n");  
}

//--------------------------------------------------------------------------
/*! \brief Method for simulating the model for a given period of time
 */
//--------------------------------------------------------------------------

void classol::run(float runtime, //!< Duration of time to run the model for 
		  unsigned int which //!< Flag determining whether to run on GPU or CPU only
		  )
{
  unsigned int pno;
  unsigned int offset= 0;
  int riT= (int) (runtime/DT);

  for (int i= 0; i < riT; i++) {
    if (iT%patSetTime == 0) {
      pno= (iT/patSetTime)%PATTERNNO;
      if (which == CPU) theRates= pattern;
      if (which == GPU)	theRates= d_pattern;
      offset= pno*model.neuronN[0];
    }
    if (iT%patSetTime == patFireTime) {
      if (which == CPU)
	theRates= baserates;
      if (which == GPU)
	theRates= d_baserates;
      offset= 0;
    }
    if (which == GPU)
       stepTimeGPU(theRates, offset, t);
    if (which == CPU)
       stepTimeCPU(theRates, offset, t);
    iT++;
    t= iT*DT;
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
    copyStateFromDevice();

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
  //  os << glbscnt0 << "  ";
  // for (int i= 0; i < glbscntPN; i++) {
  //   os << glbSpkPN[i] << " ";
  // }
  // os << " * ";
  // os << glbscntKC << "  ";
  // for (int i= 0; i < glbscntKC; i++) {
  //   os << glbSpkKC[i] << " ";
  // }
  // os << " * ";
  // os << glbscntLHI << "  ";
  // for (int i= 0; i < glbscntLHI; i++) {
  //   os << glbSpkLHI[i] << " ";
  // }
  // os << " * ";
  // os << glbscntDN << "  ";
  // for (int i= 0; i < glbscntDN; i++) {
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
  copySpikesFromDevice();
}

//--------------------------------------------------------------------------
/*! \brief Method for copying the number of spikes in all neuron populations that have occurred during the last time step
 
This method is a simple wrapper for the convenience function copySpikeNFromDevice() provided by GeNN.
*/
//--------------------------------------------------------------------------

void classol::getSpikeNumbersFromGPU() 
{
  copySpikeNFromDevice();
}

//--------------------------------------------------------------------------
/*! \brief Method for writing the spikes occurred in the last time step to a file
 */
//--------------------------------------------------------------------------

void classol::output_spikes(FILE *f, //!< File handle for a file to write spike times to
			    unsigned int which //!< Flag determining whether using GPU or CPU only
			    )
{

  //    fprintf(stdout, "%f %f %f %f %f\n", t, glbscntPN, glbscntKC, glbscntLHI,glbscntDN);
  for (int i= 0; i < glbscntPN; i++) {
    fprintf(f, "%f %d\n", t, glbSpkPN[i]);
  }
  for (int i= 0; i < glbscntKC; i++) {
    fprintf(f,  "%f %d\n", t, model.sumNeuronN[0]+glbSpkKC[i]);
  }
  for (int i= 0; i < glbscntLHI; i++) {
    fprintf(f, "%f %d\n", t, model.sumNeuronN[1]+glbSpkLHI[i]);
  }
  for (int i= 0; i < glbscntDN; i++) {
    fprintf(f, "%f %d\n", t, model.sumNeuronN[2]+glbSpkDN[i]);
  }
}

//--------------------------------------------------------------------------
/*! \brief Method for summing up spike numbers
 */
//--------------------------------------------------------------------------

void classol::sum_spikes()
{
  sumPN+= glbscntPN;
  sumKC+= glbscntKC;
  sumLHI+= glbscntLHI;
  sumDN+= glbscntDN;
}

//--------------------------------------------------------------------------
/*! \brief Method for copying the synaptic conductances of the learning synapses between KCs and DNs (detector neurons) back to the CPU memory
 */
//--------------------------------------------------------------------------

void classol::get_kcdnsyns()
{
  void *devPtr;
  cudaGetSymbolAddress(&devPtr, "d_gpKCDN");
  CHECK_CUDA_ERRORS(cudaMemcpy(gpKCDN, devPtr,
    model.neuronN[1]*model.neuronN[3]*sizeof(float), 
    cudaMemcpyDeviceToHost));
}



#endif	

