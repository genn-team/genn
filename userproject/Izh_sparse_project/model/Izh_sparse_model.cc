/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/
#include "Izh_sparse_model.h"

// GeNN includes
#include "modelSpec.h"

#include "sizes.h"

classIzh::classIzh()
{
  allocateMem();
  initialize();
  sumPExc = 0;
  sumPInh = 0;
}

void classIzh::importArray(scalar *dest, double *src, int sz) 
{
    for (int i= 0; i < sz; i++) {
	dest[i]= (scalar) src[i];
    }
}

void classIzh::exportArray(double *dest, scalar *src, int sz) 
{
    for (int i= 0; i < sz; i++) {
        dest[i]= (scalar) src[i];
    }
}


void classIzh::initializeAllVars(unsigned int which)
{
    // Initialise U variables based on auto-initialised
    for (int j=0; j< _NInh; j++){
        UPInh[j]=bPInh[j]*VPInh[j];
    }
    for (int j=0; j< _NExc; j++){
        UPExc[j]=bPExc[j]*VPExc[j];
    }
}

classIzh::~classIzh()
{
  freeMem();
}


//--------------------------------------------------------------------------
/*! \brief Read sparse connectivity from a file
 */
//--------------------------------------------------------------------------

void classIzh::read_sparsesyns_par(unsigned int numSrc,
                                   SparseProjection &C, //!< contains teh arrays to be initialized from file
                                   FILE *f_ind, //!< file pointer for the indices of post-synaptic neurons
                                   FILE *f_indInG, //!< file pointer for the summed post-synaptic neurons numbers
                                   FILE *f_g, //!< File handle for a file containing sparse conductivity values
                                   scalar *g //!< array to receive the conductance values
    )
{
    unsigned int retval=0; //to make the compiler happy
    double * gtemp = new double[C.connN]; //we need this now as files will always be generated as double but we may run the model with single precision
    //retval=fread(g, 1, C.connN*sizeof(double),f_g);
    fprintf(stdout,"Reading sparse projection values ... \n");
    retval=fread(gtemp, 1, C.connN*sizeof(double),f_g);

    importArray(g, gtemp, C.connN);
    if (retval!=C.connN*sizeof(double)) {
        fprintf(stderr, "ERROR: Number of elements read is different than it should be.");
    }
    fprintf(stdout,"%d active synapses in group. \n",C.connN);
    retval=fread(C.indInG, 1, (numSrc+1)*sizeof(unsigned int),f_indInG);
    if (retval!=(numSrc+1)*sizeof(unsigned int)) {
        fprintf(stderr, "ERROR: Number of elements read is different than it should be.");
    }
    retval=fread(C.ind, 1, C.connN*sizeof(int),f_ind);
    if (retval!=C.connN*sizeof(int)) {
        fprintf(stderr, "ERROR: Number of elements read is different than it should be.");
    }

    // general:
    fprintf(stdout,"Read sparse projection ... \n");
    fprintf(stdout, "Size is %d for synapse group. Values start with: \n",C.connN);
    for(int i= 0; i < 20; i++) {
        fprintf(stdout, "%f ", (scalar) g[i]);
    }
    fprintf(stdout,"\n\n");
    fprintf(stdout, "%d indices read. Index values start with: \n",C.connN);
    for(int i= 0; i < 20; i++) {
        fprintf(stdout, "%d ", C.ind[i]);
    }
    fprintf(stdout,"\n\n");
    fprintf(stdout, "%d g indices read. Index in g array values start with: \n", numSrc+1);
    for(int i= 0; i < 20; i++) {
        fprintf(stdout, "%d ", C.indInG[i]);
    }
    fprintf(stdout,"\n\n");
    delete [] gtemp;
}

void classIzh::run(double runtime, unsigned int which)
{
    int riT= (int) (runtime/DT+1e-6);
    if (which == GPU){
        for (int i= 0; i < riT; i++) {
#ifndef CPU_ONLY
            stepTimeGPU();
#endif
        }
    }

    if (which == CPU){
        for (int i= 0; i < riT; i++) {
            stepTimeCPU();
        }
    }

}

void classIzh::sum_spikes()
{
  sumPExc+= glbSpkCntPExc[0];
  sumPInh+= glbSpkCntPInh[0];
}

//--------------------------------------------------------------------------
// output functions

void classIzh::output_state(FILE *f, unsigned int which)
{
  if (which == GPU) 
#ifndef CPU_ONLY
    copyStateFromDevice();
#endif

  fprintf(f, "%f ", t);

   for (int i= 0; i < _NExc; i++) {
     fprintf(f, "%f ", VPExc[i]);
   }
   
   for (int i= 0; i < _NInh; i++) {
     fprintf(f, "%f ", VPInh[i]);
   }
   
  fprintf(f,"\n");
}

void classIzh::output_params(FILE *f, FILE *f2)
{
    for (int i= 0; i < _NExc; i++) {
        fprintf(f, "%f ", aPExc[i]);
        fprintf(f, "%f ", bPExc[i]);
        fprintf(f, "%f ", cPExc[i]);
        fprintf(f, "%f ", dPExc[i]);
        fprintf(f,"\n");
    }

    for (int i= 0; i < _NInh; i++) {
        fprintf(f2, "%f ", aPInh[i]);
        fprintf(f2, "%f ", bPInh[i]);
        fprintf(f2, "%f ", cPInh[i]);
        fprintf(f2, "%f ", dPInh[i]);
        fprintf(f2,"\n");

    }
}
//--------------------------------------------------------------------------
/*! \brief Method for copying all spikes of the last time step from the GPU
 
  This is a simple wrapper for the convenience function copySpikesFromDevice() which is provided by GeNN.
*/
//--------------------------------------------------------------------------

void classIzh::getSpikesFromGPU()
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

void classIzh::getSpikeNumbersFromGPU() 
{
#ifndef CPU_ONLY
    copySpikeNFromDevice();
#endif
}


void classIzh::output_spikes(FILE *f, unsigned int which)
{
    if (which == GPU) {
        //getSpikeNumbersFromGPU();
        getSpikesFromGPU();
    }

    for (int i= 0; i < glbSpkCntPExc[0]; i++) {
        fprintf(f,"%f %d\n", t, glbSpkPExc[i]);
    }

    for (int i= 0; i < glbSpkCntPInh[0]; i++) {
        fprintf(f, "%f %d\n", t, _NExc + glbSpkPInh[i]);
    }
}
