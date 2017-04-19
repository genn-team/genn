/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#ifndef _IZH_SPARSE_MODEL_CC_
#define _IZH_SPARSE_MODEL_CC_

#include "Izh_sparse_CODE/definitions.h"
#include "randomGen.h"
#include "gauss.h"
#include "Izh_sparse_model.h"

randomGauss RG;
randomGen R;

classIzh::classIzh()
{
  modelDefinition(model);
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

void classIzh::randomizeVar(scalar * Var, scalar strength, const NeuronGroup *neuronGrp)
{
  //kernel if gpu?
  for (int j=0; j< neuronGrp->getNumNeurons(); j++){
    Var[j]=Var[j]+strength*((scalar) R.n());
  }
}

void classIzh::randomizeVarSq(scalar * Var, scalar strength, const NeuronGroup *neuronGrp)
{
  //kernel if gpu?
  //randomGen R;
  scalar randNbr;
  for (int j=0; j< neuronGrp->getNumNeurons(); j++){
    randNbr=((scalar) R.n());
    Var[j]=Var[j]+strength*randNbr*randNbr;
  }
}


void classIzh::initializeAllVars(unsigned int which)
{
  auto *pInh = model.findNeuronGroup("PInh");
  auto *pExc = model.findNeuronGroup("PExc");
  randomizeVar(aPInh,0.08, pInh);
  randomizeVar(bPInh,-0.05,pInh);
  randomizeVarSq(cPExc,15.0,pExc);
  randomizeVarSq(dPExc,-6.0,pExc);

  for (int j=0; j< pInh->getNumNeurons(); j++){
    UPExc[j]=bPExc[j]*VPExc[j];
    UPInh[j]=bPInh[j]*VPInh[j];	
  }
  for (int j=pInh->getNumNeurons(); j< pExc->getNumNeurons(); j++){
    UPExc[j]=bPExc[j]*VPExc[j];
  }
#ifndef CPU_ONLY
  if (which == GPU) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_aPInh, aPInh, sizeof(scalar)*pInh->getNumNeurons(), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_bPInh, bPInh, sizeof(scalar)*pInh->getNumNeurons(), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_cPExc, cPExc, sizeof(scalar)*pExc->getNumNeurons(), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_dPExc, dPExc, sizeof(scalar)*pExc->getNumNeurons(), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_UPExc, UPExc, sizeof(scalar)*pExc->getNumNeurons(), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_UPInh, UPInh, sizeof(scalar)*pInh->getNumNeurons(), cudaMemcpyHostToDevice));
  }
#endif
}
	
void classIzh::init(unsigned int which)
{
  if (which == CPU) {
  }
  if (which == GPU) {
#ifndef CPU_ONLY
    copyStateToDevice();
#endif
  }
}


void classIzh::copy_device_mem_input()
{
#ifndef CPU_ONLY
  auto *pInh = model.findNeuronGroup("PInh");
  auto *pExc = model.findNeuronGroup("PExc");
  CHECK_CUDA_ERRORS(cudaMemcpy(d_I0PExc,I0PExc, pExc->getNumNeurons()*sizeof(scalar), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERRORS(cudaMemcpy(d_I0PInh,I0PInh, pInh->getNumNeurons()*sizeof(scalar), cudaMemcpyHostToDevice));
#endif
}

classIzh::~classIzh()
{
  freeMem();
}

// Functions related to explicit input
void classIzh::write_input_to_file(FILE *f)
{
  printf("input is const, no need to write");
  unsigned int outno;
  auto *pExc = model.findNeuronGroup("PExc");
  if (pExc->getNumNeurons() > 10)
  outno=10;
  else outno=pExc->getNumNeurons();

  fprintf(f, "%f ", t);
  for(int i=0;i<outno;i++) {
    fprintf(f, "%f ", I0PExc[i]);
  }
  fprintf(f,"\n");
}

void classIzh::create_input_values() //define your explicit input rule here
{
    auto *pExc = model.findNeuronGroup("PExc");
    for (int x= 0; x < pExc->getNumNeurons(); x++) {
	I0PExc[x]= meanInpExc*((scalar) RG.n());
    }

    auto *pInh = model.findNeuronGroup("PInh");
    for (int x= 0; x < pInh->getNumNeurons(); x++) {
	I0PInh[x]= meanInpInh*((scalar) RG.n());
    }
}

//--------------------------------------------------------------------------
/*! \brief Read sparse connectivity from a file
 */
//--------------------------------------------------------------------------

void classIzh::read_sparsesyns_par(const char *synGrpName, //!< index of the synapse population to be worked on
				   SparseProjection C, //!< contains teh arrays to be initialized from file
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

  auto *synGrp = model.findSynapseGroup(synGrpName);
  if (retval!=C.connN*sizeof(double)) fprintf(stderr, "ERROR: Number of elements read is different than it should be.");
  fprintf(stdout,"%d active synapses in group %s. \n",C.connN,synGrpName);
  retval=fread(C.indInG, 1, (synGrp->getSrcNeuronGroup()->getNumNeurons()+1)*sizeof(unsigned int),f_indInG);
  if (retval!=(synGrp->getSrcNeuronGroup()->getNumNeurons()+1)*sizeof(unsigned int)) fprintf(stderr, "ERROR: Number of elements read is different than it should be.");
  retval=fread(C.ind, 1, C.connN*sizeof(int),f_ind);
  if (retval!=C.connN*sizeof(int)) fprintf(stderr, "ERROR: Number of elements read is different than it should be.");

  // general:
  fprintf(stdout,"Read sparse projection ... \n");
  fprintf(stdout, "Size is %d for synapse group %s. Values start with: \n",C.connN, synGrpName);
  for(int i= 0; i < 20; i++) {
      fprintf(stdout, "%f ", (scalar) g[i]);
  }
  fprintf(stdout,"\n\n");
  fprintf(stdout, "%d indices read. Index values start with: \n",C.connN);
  for(int i= 0; i < 20; i++) {
    fprintf(stdout, "%d ", C.ind[i]);
  }  
  fprintf(stdout,"\n\n");  
  fprintf(stdout, "%d g indices read. Index in g array values start with: \n", synGrp->getSrcNeuronGroup()->getNumNeurons()+1);
  for(int i= 0; i < 20; i++) {
    fprintf(stdout, "%d ", C.indInG[i]);
  }  
	fprintf(stdout,"\n\n");
  delete [] gtemp;
}

//--------------------------------------------------------------------------
/*! \brief Generate random conductivity values for an all to all network
 */
//--------------------------------------------------------------------------

void classIzh::gen_alltoall_syns( scalar * g, //!< the resulting synaptic conductances
				  const char *synGrpName, //!< name of synapse group
				  scalar gscale //!< the maximal conductance of generated synapses
			    )
{
  //randomGen R;
  auto *synGrp = model.findSynapseGroup(synGrpName);
  unsigned int numPre = synGrp->getSrcNeuronGroup()->getNumNeurons();
  unsigned int numPost = synGrp->getTrgNeuronGroup()->getNumNeurons();
  for(int i= 0; i < numPre; i++) {
  	 for(int j= 0; j < numPost; j++){
	   g[i*numPost+j]=gscale*((scalar) R.n());
    }
  }
  fprintf(stdout,"\n\n");
}

void classIzh::setInput(unsigned int which)
{
	create_input_values();
	//if (which == GPU) copy_device_mem_input();
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

   auto *pExc = model.findNeuronGroup("PExc");
   for (int i= 0; i < pExc->getNumNeurons()-1; i++) {
     fprintf(f, "%f ", VPExc[i]);
   }
   
   auto *pInh = model.findNeuronGroup("PInh");
   for (int i= 0; i < pInh->getNumNeurons()-1; i++) {
     fprintf(f, "%f ", VPInh[i]);
   }
   
  fprintf(f,"\n");
}

void classIzh::output_params(FILE *f, FILE *f2)
{
    auto *pExc = model.findNeuronGroup("PExc");
    for (int i= 0; i < pExc->getNumNeurons()-1; i++) {
        fprintf(f, "%f ", aPExc[i]);
        fprintf(f, "%f ", bPExc[i]);
        fprintf(f, "%f ", cPExc[i]);
        fprintf(f, "%f ", dPExc[i]);
        fprintf(f, "%f ", I0PExc[i]);
        fprintf(f,"\n");
    }

    auto *pInh = model.findNeuronGroup("PInh");
    for (int i= 0; i < pInh->getNumNeurons()-1; i++) {
        fprintf(f2, "%f ", aPInh[i]);
        fprintf(f2, "%f ", bPInh[i]);
        fprintf(f2, "%f ", cPInh[i]);
        fprintf(f2, "%f ", dPInh[i]);
        fprintf(f2, "%f ", I0PInh[i]);
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

    auto *pInh = model.findNeuronGroup("PInh");
    unsigned int inhOffset = pInh->getIDRange().first;
    for (int i= 0; i < glbSpkCntPInh[0]; i++) {
	fprintf(f, "%f %d\n", t, inhOffset + glbSpkPInh[i]);
    }
}

#endif
