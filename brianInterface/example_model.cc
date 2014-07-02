//--------------------------------------------------------------------------
/*! \sa{modelDefinition(NNmodel &model)}
 */
//--------------------------------------------------------------------------

void modelDefinition(NNmodel &model) 
{
  neruonModel n;

  // define a model type
  n.varNames.clear();
  n.varTypes.clear();
  n.varNames.push_back(tS("V"));
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("m"));
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("h"));
  n.varTypesz.push_back(tS("float"));
  n.varNames.push_back(tS("n"));
  n.varTypes.push_back(tS("float"));
  n.pNames.clear();
  n.pNames.push_back(tS("gNa"));
  n.pNames.push_back(tS("ENa"));
  n.pNames.push_back(tS("gK"));
  n.pNames.push_back(tS("EK"));
  n.pNames.push_back(tS("gl"));
  n.pNames.push_back(tS("El"));
  n.pNames.push_back(tS("C"));
  n.dpNames.clear();
  n.simCode= tS("   float Imem;\n\
    unsigned int mt;\n\
    float mdt= DT/25.0f;\n\
    for (mt=0; mt < 25; mt++) {\n\
      Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n\
              $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n\
              $(gl)*($(V)-($(El)))-Isyn);\n\
      float _a= 0.32f*(-52.0f-$(V)) / (exp((-52.0f-$(V))/4.0f)-1.0f);\n\
      float _b= 0.28f*($(V)+25.0f)/(exp(($(V)+25.0f)/5.0f)-1.0f);\n\
      $(m)+= (_a*(1.0f-$(m))-_b*$(m))*mdt;\n\
      _a= 0.128*expf((-48.0f-$(V))/18.0f);\n\
      _b= 4.0f / (expf((-25.0f-$(V))/5.0f)+1.0f);\n\
      $(h)+= (_a*(1.0f-$(h))-_b*$(h))*mdt;\n\
      _a= .032f*(-50.0f-$(V)) / (expf((-50.0f-$(V))/5.0f)-1.0f); \n\
      _b= 0.5f*expf((-55.0f-$(V))/40.0f);\n\
      $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n\
      $(V)+= Imem/$(C)*mdt;\n\
    }\n");
  nModels.push_back(n);

  // define a "network" of neuron models
  model.setName("IzhEx");
  model.addNeuronPopulation("Izh1", _NC1, IZHIKEVICH, exIzh_p, exIzh_ini);
}
