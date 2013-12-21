/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

#include <string>

//--------------------------------------------------------------------------
/*! \file generateCPU.cc 

  \brief Functions for generating code that will run the neuron and synapse simulations on the CPU. Part of the code generation section.

*/
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
/*!
  \brief Function that generates the code of the function the will simulate all neurons on the CPU.

*/
//--------------------------------------------------------------------------

void genNeuronFunction(NNmodel &model, //!< Model description 
		       string &path, //!< output stream for code
		       ostream &mos //!< output stream for messages
		       )
{
  string name, s, localID;
  unsigned int nt;
  ofstream os;

  name = path + toString("/") + model.name + toString("_CODE/neuronFnct.cc");
  os.open(name.c_str());
  // write header content
  writeHeader(os);
  os << endl;
  // compiler/include control (include once)
  os << "#ifndef _" << model.name << "_neuronFnct_cc" << endl;
  os << "#define _" << model.name << "_neuronFnct_cc" << endl;
  os << endl;

  // write doxygen comment
  os << "//-------------------------------------------------------------------------" << endl;
  os << "/*! \\file neuronFnct.cc" << endl << endl;
  os << "\\brief File generated from GeNN for the model " << model.name;
  os << " containing the the equivalent of neuron kernel function for the CPU-only version." << endl;
  os << "*/" << endl;
  os << "//-------------------------------------------------------------------------" << endl << endl;

  // CPU function for calculating neuron states
  // header
  os << "void calcNeuronsCPU(";
  for (int i = 0; i < model.neuronGrpN; i++) {
    if (model.neuronType[i] == POISSONNEURON) {
      // Note: Poisson neurons only used as input neurons; they do not 
      // receive any inputs
      os << "unsigned int *rates" << model.neuronName[i] << ", // poisson ";
      os << "\"rates\" of grp " << model.neuronName[i] << endl;
      os << "unsigned int offset" << model.neuronName[i] << ", // poisson ";
      os << "\"rates\" offset of grp " << model.neuronName[i] << endl;
    }
    if (model.receivesInputCurrent[i] >= 2) {
      os << "float *inputI" << model.neuronName[i] << ", // input current of grp " << model.neuronName[i] << endl;
    }
  }
  os << "float t)" << endl;
  os << "{" << endl;
  for (int i = 0; i < model.neuronGrpN; i++) {
    nt = model.neuronType[i];





    if (model.neuronDelaySlots[i] == 1) {
      os << "  glbscnt" << model.neuronName[i] << " = 0;" << endl;
    }
    else {
      os << "  spkQuePtr" << model.neuronName[i] << " = (spkQuePtr" << model.neuronName[i] << " + 1) % ";
      os << model.neuronDelaySlots[i] << ";" << endl;
      os << "  glbscnt" << model.neuronName[i] << "[spkQuePtr" << model.neuronName[i] << "] = 0;" << endl;
    }




    os << "  for (int n = 0; n < " <<  model.neuronN[i] << "; n++) {" << endl;





    for (int k = 0; k < nModels[nt].varNames.size(); k++) {
      os << "    " << nModels[nt].varTypes[k] << " l" << nModels[nt].varNames[k] << " = ";
      os << nModels[nt].varNames[k] << model.neuronName[i] << "[";
      if ((nModels[nt].varNames[k] == "V") && (model.neuronDelaySlots[i] != 1)) {
	os << "(((spkQuePtr" << model.neuronName[i] << " + " << (model.neuronDelaySlots[i] - 1) << ") % ";
	os << model.neuronDelaySlots[i] << ") * " << model.neuronN[i] << ") + ";
      }
      os << "n];" << endl;
    }





    os << "    float Isyn = 0;" << endl;
    if (model.inSyn[i].size() > 0) {
      os << "    Isyn = ";
      for (int j = 0; j < model.inSyn[i].size(); j++) {
	os << "inSyn" << model.neuronName[i] << j << "[n] * (";
	os << SAVEP(model.synapsePara[model.inSyn[i][j]][0]) << " - lV)";
	if (j < model.inSyn[i].size() - 1) {
	  os << " + ";
	}
	else {
	  os << ";" << endl;
	}
      }
    }
    if (model.receivesInputCurrent[i] == 1) { //receives constant input
      if (model.synapseGrpN == 0) {
   	os << "    Isyn = " << model.globalInp[i] << ";" << endl;
      }
      else {
	os << "    Isyn += " << model.globalInp[i] << ";" << endl;
      }
    }    	
    if (model.receivesInputCurrent[i] >= 2) {
      if (model.synapseGrpN == 0) {
	os << "    Isyn = inputI" << model.neuronName[i] << "[n];" << endl;
      }
      else {
	os << "    Isyn += inputI" << model.neuronName[i] << "[n];" << endl;
      }
    }
    os << "    // calculate membrane potential" << endl;
    string code = nModels[nt].simCode;





    for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
      substitute(code, tS("$(") + nModels[nt].varNames[k] + tS(")"),
		 tS("l") + nModels[nt].varNames[k]);
    }




    if (nt == POISSONNEURON) {
      substitute(code, tS("lrate"), 
		 tS("rates") + model.neuronName[i] + tS("[n + offset") + model.neuronName[i] + tS("]"));
    }
    substitute(code, tS("$(Isyn)"), tS("Isyn"));
    for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
      substitute(code, tS("$(") + nModels[nt].pNames[k] + tS(")"), 
		 tS(model.neuronPara[i][k]));
    }
    for (int k = 0, l = nModels[nt].dpNames.size(); k < l; k++) {
      substitute(code, tS("$(") + nModels[nt].dpNames[k] + tS(")"), 
		 tS(model.dnp[i][k]));
    }
    os << code;
    if (nModels[nt].thresholdCode != tS("")) {
      code = nModels[nt].thresholdCode;
      for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	substitute(code, tS("$(") + nModels[nt].varNames[k] + tS(")"), 
		   nModels[nt].varNames[k] + model.neuronName[i] + tS("[n]"));
      }
      substitute(code, tS("$(Isyn)"), tS("Isyn"));
      for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
	substitute(code, tS("$(") + nModels[nt].pNames[k] + tS(")"), 
		   tS(model.neuronPara[i][k]));
      }
      for (int k = 0, l = nModels[nt].dpNames.size(); k < l; k++) {
	substitute(code, tS("$(") + nModels[nt].dpNames[k] + tS(")"), 
		   tS(model.dnp[i][k]));
      }
      os << code << endl;
      os << "    if (_cond) {" << endl;
    }
    else {
      os << "    if (lV >= " << model.nThresh[i] << ") {" << endl;
    }
    os << "      // register a spike type event" << endl;






    os << "      glbSpk" << model.neuronName[i] << "[";
    if (model.neuronDelaySlots[i] != 1) {
      os << "(spkQuePtr" << model.neuronName[i] << " * " << model.neuronN[i] << ") + ";
    }
    os << "glbscnt" << model.neuronName[i];
    if (model.neuronDelaySlots[i] != 1) {
      os << "[spkQuePtr" << model.neuronName[i] << "]";
    }
    os << "++] = n;" << endl;





    if (model.neuronNeedSt[i]) {
      os << "      sT" << model.neuronName[i] << "[n] = t;" << endl;
    }
    if (nModels[nt].resetCode != tS("")) {
      code = nModels[nt].resetCode;
      for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	substitute(code, tS("$(") + nModels[nt].varNames[k] + tS(")"), 
		   nModels[nt].varNames[k] + model.neuronName[i] + tS("[n]"));
      }
      substitute(code, tS("$(Isyn)"), tS("Isyn"));
      for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
	substitute(code, tS("$(") + nModels[nt].pNames[k] + tS(")"), 
		   tS(model.neuronPara[i][k]));
      }
      for (int k = 0, l = nModels[nt].dpNames.size(); k < l; k++) {
	substitute(code, tS("$(") + nModels[nt].dpNames[k] + tS(")"), 
		   tS(model.dnp[i][k]));
      }
      os << "      // spike reset code" << endl;
      os << "      " << code << endl;
    }
    os << "    }" << endl;






    for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
      os << "    " << nModels[nt].varNames[k] <<  model.neuronName[i] << "[";
      if ((nModels[nt].varNames[k] == "V") && (model.neuronDelaySlots[i] != 1)) {
	os << "(spkQuePtr" << model.neuronName[i] << " * " << model.neuronN[i] << ") + ";
      }
      os << "n] = l" << nModels[nt].varNames[k] << ";" << endl;
    }





    for (int j = 0; j < model.inSyn[i].size(); j++) {
      os << "    inSyn"  << model.neuronName[i] << j << "[n] *= ";
      unsigned int synID = model.inSyn[i][j];
      os << SAVEP(model.dsp[synID][0]) << ";" << endl;
    }
    os << "  }" << endl;
    os << endl;
  }
  os << "}" << endl << endl;
  os << "#endif" << endl;
  os.close();
} 


//--------------------------------------------------------------------------
/*!
  \brief Function that generates code that will simulate all synapses of the model on the CPU.

*/
//--------------------------------------------------------------------------

void genSynapseFunction(NNmodel &model, //!< Model description
			string &path, //!< Path for code generation
			ostream &mos //!< output stream for messages
			)
{
  string name, s, localID, theLG;
  ofstream os;

  name = path + toString("/") + model.name + toString("_CODE/synapseFnct.cc");
  os.open(name.c_str());
  // write header content
  writeHeader(os);
  os << endl;
  // compiler/include control (include once)
  os << "#ifndef _" << model.name << "_synapseFnct_cc" << endl;
  os << "#define _" << model.name << "_synapseFnct_cc" << endl;
  os << endl;

  // write doxygen comment
  os << "//-------------------------------------------------------------------------" << endl;
  os << "/*! \\file synapseFnct.cc" << endl << endl;
  os << "\\brief File generated from GeNN for the model " << model.name << " containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version." << endl;
  os << "*/" << endl;
  os << "//-------------------------------------------------------------------------" << endl << endl;

  // Function for calculating synapse input to neurons
  // Function header
  os << "void calcSynapsesCPU(float t)" << endl;
  os << "{" << endl;
  if (model.lrnGroups > 0) {
    os << "  float dt, dg;" << endl;
  }
  if (model.needSynapseDelay) {
    os << "  int delaySlot;" << endl;
  }
  os << endl;
  for (int i = 0; i < model.neuronGrpN; i++) {
    if (model.inSyn[i].size() > 0) { // there is input onto this neuron group
      for (int j = 0; j < model.inSyn[i].size(); j++) {
	unsigned int synID = model.inSyn[i][j];
	unsigned int src = model.synapseSource[synID];
	float Epre = model.synapsePara[synID][1];
	float Vslope;
	if (model.synapseType[synID] == NGRADSYNAPSE) {
	  Vslope = model.synapsePara[synID][3];
	}



	if (model.neuronDelaySlots[src] != 1) {
	  os << "  delaySlot = (spkQuePtr" << model.neuronName[src] << " + ";
	  os << (int) (model.neuronDelaySlots[src] - model.synapseDelay[synID] + 1);
	  os << ") % " << model.neuronDelaySlots[src] << ";" << endl;
	}



	os << "  for (int j = 0; j < glbscnt" << model.neuronName[src];
	if (model.neuronDelaySlots[src] != 1) {
	  os << "[delaySlot]";
	}
	os << "; j++) {" << endl;




	os << "    for (int n = 0; n < " << model.neuronN[i] << "; n++) {" << endl;
	if (model.synapseGType[synID] == INDIVIDUALID) {
	  os << "      unsigned int gid = (glbSpk" << model.neuronName[src] << "[";





	  if (model.neuronDelaySlots[src] != 1) {
	    os << "delaySlot * " << model.neuronN[src] << " + ";
	  }





	  os << "j] * " << model.neuronN[i] << " + n);" << endl;
	}
	if (model.neuronType[src] != POISSONNEURON) {
	  os << "      if ";
	  if (model.synapseGType[synID] == INDIVIDUALID) {
	    os << "((B(gp" << model.synapseName[synID] << "[gid >> " << logUIntSz << "], gid & ";
	    os << UIntSz - 1 << ")) && ";
	  } 




	  os << "(V" << model.neuronName[src] << "[";
	  if (model.neuronDelaySlots[src] != 1) {
	    os << "(delaySlot * " << model.neuronN[src] << ") + ";
	  }
	  os << "glbSpk" << model.neuronName[src] << "[";
	  if (model.neuronDelaySlots[src] != 1) {
	    os << "(delaySlot * " << model.neuronN[src] << ") + ";
	  }
	  os << "j]] > " << Epre << ")";





	  if (model.synapseGType[synID] == INDIVIDUALID) {
	    os << ")";
	  }
	  os << " {" << endl;
	}
	else {
	  if (model.synapseGType[synID] == INDIVIDUALID) {
	    os << "          if (B(gp" << model.synapseName[synID] << "[gid >> " << logUIntSz << "], gid & ";
	    os << UIntSz - 1 << ")) {" << endl;
	  }
	}
	if (model.synapseGType[synID] == INDIVIDUALG) {
	  theLG = toString("gp")+model.synapseName[synID]+toString("[glbSpk");
	  theLG += model.neuronName[src] + toString("[");





	  if (model.neuronDelaySlots[src] != 1) {
	    theLG += toString("delaySlot * ") + toString(model.neuronN[src]) + toString(" + ");
	  }





	  theLG += toString("j] * ") + toString(model.neuronN[i]);
	  theLG += toString(" + n]");
	}
	if ((model.synapseGType[synID] == GLOBALG) ||
	    (model.synapseGType[synID] == INDIVIDUALID)) {
	  theLG = toString(model.g0[synID]);
	}
	if ((model.synapseType[synID] == NSYNAPSE) || 
	    (model.synapseType[synID] == LEARN1SYNAPSE)) {
	  os << "        inSyn" << model.neuronName[i] << j << "[n] += " << theLG << ";" << endl;
	}
	if (model.synapseType[synID] == NGRADSYNAPSE) {
	  if (model.neuronType[src] == POISSONNEURON) {
	    os << "        inSyn" << model.neuronName[i] << j << "[n] += " << theLG << " * tanh((";
	    os << SAVEP(model.neuronPara[src][2]) << " - " << SAVEP(Epre);
	  }
	  else {
	    os << "        inSyn" << model.neuronName[i] << j << "[n] += " << theLG << " * tanh((V";





	    os << model.neuronName[src] << "[";
	    if (model.neuronDelaySlots[src] != 1) {
	      os << "(delaySlot * " << model.neuronN[src] << ") + ";
	    }
	    os << "glbSpk" << model.neuronName[src] << "[";
	    if (model.neuronDelaySlots[src] != 1) {
	      os << "(delaySlot * " << model.neuronN[src] << ") + ";
	    }
	    os << "j]] - " << SAVEP(Epre);





	  }
	  os << ") / " << SAVEP(Vslope) << ");" << endl;
	}
	// if needed, do some learning (this is for pre-synaptic spikes)
	if (model.synapseType[synID] == LEARN1SYNAPSE) {
	  // simply assume INDIVIDUALG for now
	  os << "          dt = sT" << model.neuronName[i] << "[n] - t - ";
	  os << SAVEP(model.synapsePara[synID][11]) << ";" << endl;
	  os << "          if (dt > " << model.dsp[synID][1] << ") {" << endl;
	  os << "            dg = -" << SAVEP(model.dsp[synID][5]) << ";" << endl;
	  os << "          }" << endl;
	  os << "          else if (dt > 0.0) {" << endl;
	  os << "            dg = " << SAVEP(model.dsp[synID][3]) << " * dt + ";
	  os << SAVEP(model.dsp[synID][6]) << ";" << endl;
	  os << "          }" << endl;
	  os << "          else if (dt > " << model.dsp[synID][2] << ") {" << endl;
	  os << "            dg = " << SAVEP(model.dsp[synID][4]) << " * dt + ";
	  os << SAVEP(model.dsp[synID][6]) << ";" << endl;
	  os << "          }" << endl;
	  os << "          else {" << endl;
	  os << "            dg = -" << SAVEP(model.dsp[synID][7]) << ";" << endl;
	  os << "          }" << endl;





	  os << "          grawp" << model.synapseName[synID] << "[glbSpk" << model.neuronName[src] << "[";
	  if (model.neuronDelaySlots[src] != 1) {
	    os << "delaySlot * " << model.neuronN[src] << " + ";
	  }
	  os << "j]*" << model.neuronN[i] << " + n] += dg;" << endl;





	  os << "          gp" << model.synapseName[synID] << "[glbSpk" << model.neuronName[src] << "[";
	  if (model.neuronDelaySlots[src] != 1) {
	    os << "delaySlot * " << model.neuronN[src] << " + ";
	  }






	  os << "j] * " << model.neuronN[i] << " + n] = ";
	  os << "gFunc" << model.synapseName[synID] << "(grawp" << model.synapseName[synID];
	  os << "[glbSpk" << model.neuronName[src] << "[";






	  if (model.neuronDelaySlots[src] != 1) {
	    os << "delaySlot * " << model.neuronN[src] << " + ";
	  }





	  os << "j] * " << model.neuronN[i] << " + n]);" << endl; 
	}
	if ((model.neuronType[src] != POISSONNEURON) ||
	    (model.synapseGType[synID] == INDIVIDUALID)) {
	  os << "      }" << endl;
	}
	os << "    }" << endl;
	os << "  }" << endl;
      }
    }
  }
  os << "}" << endl << endl;

  if (model.lrnGroups > 0) {
    // function for learning synapses, post-synaptic spikes
    // function header
    os << "void learnSynapsesPostHost(float t)" << endl;
    os << "{" << endl;
    os << "  float dt, dg;" << endl;
    os << endl;

    for (int i = 0; i < model.lrnGroups; i++) {





      unsigned int k = model.lrnSynGrp[i];
      unsigned int src = model.synapseSource[k];
      unsigned int nN = model.neuronN[src];
      unsigned int trg = model.synapseTarget[k];
      float Epre = model.synapsePara[k][1];





      os << "  for (int j = 0; j < glbscnt" << model.neuronName[trg];
      if (model.neuronDelaySlots[trg] != 1) {
	os << "[spkQuePtr" << model.neuronName[trg] << "]" << endl;
      }
      os << "; j++) {" << endl;





      os << "    for (int n = 0; n < " << model.neuronN[src] << "; n++) {" << endl;
      os << "      if (V" << model.neuronName[trg] << "[";
      if (model.neuronDelaySlots[trg] != 1) {
	os << "(spkQuePtr" << model.neuronName[trg] << " * " << model.neuronN[trg] << ") + ";
      }
      os << "glbSpk" << model.neuronName[trg] << "[";
      if (model.neuronDelaySlots[trg] != 1) {
	os << "(spkQuePtr" << model.neuronName[trg] << " * " << model.neuronN[trg] << ") + ";
      }
      os << "j]] > " << Epre << ") {" << endl;





      os << "        dt = t - sT" << model.neuronName[src] << "[n]";
      if (model.neuronDelaySlots[src] != 1) {
	os << " + " << (DT * model.synapseDelay[k]);
      }
      os << " - " << SAVEP(model.synapsePara[k][11]) << ";" << endl;





      os << "        if (dt > " << model.dsp[k][1] << ") {" << endl;
      os << "          dg = -" << SAVEP(model.dsp[k][5]) << ";" << endl;
      os << "        }" << endl;
      os << "        else if (dt > 0.0) {" << endl;
      os << "          dg = " << SAVEP(model.dsp[k][3]) << " * dt + " << SAVEP(model.dsp[k][6]) << ";" << endl;
      os << "        }" << endl;
      os << "        else if (dt > " << model.dsp[k][2] << ") {" << endl;
      os << "          dg = " << SAVEP(model.dsp[k][4]) << " * dt + " << SAVEP(model.dsp[k][6]) << ";" << endl;
      os << "        }" << endl;
      os << "        else {" << endl;
      os << "          dg = -" << SAVEP(model.dsp[k][7]) << ";" << endl;
      os << "        }" << endl;
      os << "        grawp" << model.synapseName[k] << "[n * ";
      os << model.neuronN[trg] << " + glbSpk" << model.neuronName[trg] << "[";


      if (model.neuronDelaySlots[trg] != 1) {
	os << "(spkQuePtr" << model.neuronName[trg] << " * " << model.neuronN[trg] << ") + ";
      }



      os << "j]] += dg;" << endl;
      os << "        gp" << model.synapseName[k] << "[n * ";
      os << model.neuronN[trg] << " + glbSpk" << model.neuronName[trg] << "[";



      if (model.neuronDelaySlots[trg] != 1) {
	os << "(spkQuePtr" << model.neuronName[trg] << " * " << model.neuronN[trg] << ") + ";
      }



      os << "j]] = gFunc" << model.synapseName[k] << "(grawp" << model.synapseName[k] << "[n * ";
      os << model.neuronN[trg] << " + glbSpk" << model.neuronName[trg] << "[";



      if (model.neuronDelaySlots[trg] != 1) {
	os << "(spkQuePtr" << model.neuronName[trg] << " * " << model.neuronN[trg] << ") + ";
      }



      os << "j]]);" << endl; 
      os << "      }" << endl;
      os << "    }" << endl;
      os << "  }" << endl;
    }
    os << "}" << endl;
  }
  os << endl;
  os << "#endif" << endl;
  os.close();
}
