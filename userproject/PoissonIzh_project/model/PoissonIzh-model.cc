/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/
#include "PoissonIzh-model.h"

// GeNN includes
#include "modelSpec.h"

// generate_run generated code
#include "sizes.h"

classol::classol()
{
  allocateMem();
  initialize();
  sumPN= 0;
  sumIzh1= 0;
}

classol::~classol()
{
  freeMem();
}

void classol::init(unsigned int which)
{
    if (which == GPU) {
#ifndef CPU_ONLY
        copyStateToDevice();
#endif
    }
}
//--------------------------------------------------------------------------
/*! \brief Helper function to cast an array to the appropriate floating point type for the current model
 */
//--------------------------------------------------------------------------

void classol::importArray(scalar *dest, //!< pointer to destination
                          double *src, //!< pointer to the source
                          int sz)//!< number of elements of the array to be copied
{
    for (int i= 0; i < sz; i++) {
        dest[i]= (scalar) src[i];
    }
}

//--------------------------------------------------------------------------
/*! \brief Helper function to cast an array from the floating point type of the current model to double.
 */
//--------------------------------------------------------------------------

void classol::exportArray(double *dest, //!< pointer to destination
                          scalar *src,  //!< pointer to the source
                          int sz) //!< number of elements of the array to be copied
{
    for (int i= 0; i < sz; i++) {
        dest[i]= (scalar) src[i];
    }
}

void classol::read_PNIzh1syns(scalar *gp, FILE *f)
{

    int sz= _NPoisson * _NIzh;
    double *tmpg= new double[sz];
    unsigned int retval = fread(tmpg, 1, sz * sizeof(double),  f);
    importArray(gp, tmpg, sz);
    fprintf(stderr,"read PNIzh1 ... \n");
    fprintf(stderr, "%u bytes, values start with: \n", retval);
    for(int i= 0; i < 100; i++) {
        fprintf(stderr, "%f ", float(gp[i]));
    }
    fprintf(stderr,"\n\n");
    delete [] tmpg;
}

//--------------------------------------------------------------------------
/*! \brief Read sparse connectivity from a file
 */
//--------------------------------------------------------------------------

void classol::read_sparsesyns_par(unsigned int numPre, //!< number of presynaptic neurons
                                  SparseProjection &C, //!< contains the arrays to be initialized from file
                                  FILE *f_ind, //!< file pointer for the indices of post-synaptic neurons
                                  FILE *f_indInG, //!< file pointer for the summed post-synaptic neurons numbers
                                  FILE *f_g, //!< File handle for a file containing sparse conductivity values
                                  double * g)//!< array to receive the conductance values
{
    unsigned int retval = fread(g, C.connN * sizeof(scalar),1,f_g);
    fprintf(stderr,"%d active synapses. \n",C.connN);
    retval = fread(C.indInG, (numPre+1)*sizeof(unsigned int),1,f_indInG);
    retval = fread(C.ind, C.connN*sizeof(int),1,f_ind);


    // general:
    fprintf(stderr,"Read conductance ... \n");
    fprintf(stderr, "Size is %d for synapse group. Values start with: \n",C.connN);
    for(int i= 0; i < 100; i++) {
        fprintf(stderr, "%f ", float(g[i]));
    }
    fprintf(stderr,"\n\n");


    fprintf(stderr, "%d indices read. Index values start with: \n",C.connN);
    for(int i= 0; i < 100; i++) {
        fprintf(stderr, "%d ", C.ind[i]);
    }
    fprintf(stderr,"\n\n");


    fprintf(stderr, "%u bytes of %d g indices read. Index in g array values start with: \n", retval, numPre+1);
    for(int i= 0; i < 100; i++) {
        fprintf(stderr, "%d ", C.indInG[i]);
    }
}

void classol::run(float runtime, unsigned int which)
{
    unsigned int offsetPN= 0;
    int riT= (int) (runtime/DT+1e-6);

    for (int i= 0; i < riT; i++) {
        if (which == GPU) {
#ifndef CPU_ONLY
            stepTimeGPU();
#endif
        }
        else  {
            stepTimeCPU();
        }
    }
}

//--------------------------------------------------------------------------
// output functions

void classol::output_state(FILE *f, unsigned int which)
{
    if (which == GPU) {
#ifndef CPU_ONLY
        copyStateFromDevice();
#endif
    }
    fprintf(f, "%f ", t);

    for (int i= 0; i < _NIzh; i++) {
        fprintf(f, "%f ", VIzh1[i]);
    }

    fprintf(f,"\n");
}

#ifndef CPU_ONLY
void classol::getSpikesFromGPU()
{
    copySpikesFromDevice();
}

void classol::getSpikeNumbersFromGPU() 
{
    copySpikeNFromDevice();
}
#endif

void classol::output_spikes(FILE *f, unsigned int which)
{
    for (int i= 0; i < glbSpkCntPN[0]; i++) {
        fprintf(f, "%f %d\n", t, glbSpkPN[i]);
    }
    for (int i= 0; i < glbSpkCntIzh1[0]; i++) {
        fprintf(f,  "%f %d\n", t, _NPoisson+glbSpkIzh1[i]);
    }
}

void classol::sum_spikes()
{
    sumPN+= glbSpkCntPN[0];
    sumIzh1+= glbSpkCntIzh1[0];
}
