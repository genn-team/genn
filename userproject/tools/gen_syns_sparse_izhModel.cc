//--------------------------------------------------------------------------
/*! \file gen_syns_sparse_izhModel.cc

\brief This file is part of a tool chain for running the Izhikevich network model.

*/ 
//--------------------------------------------------------------------------
//gdb -tui --args ./gen_syns_sparse_izhModel 1000 1000 0.5 -1 izh
//g++ -Wall -Winline -g -I../../lib/include/numlib -o gen_syns_sparse_izhModel gen_syns_sparse_izhModel.cc


#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace std;

#include "randomGen.h"
#include "randomGen.cc"

int printVector(vector<unsigned int>&);
int printVector(vector<double>&);

randomGen R;
randomGen Rind;

double gsyn;
//  double *gAlltoAll;
double *garray; 
//  double *g; 
unsigned int *ind; 
  
  //exc-exc
  //double *gAlltoAll_ee;
  double *garray_ee; 
  //double *g_ee = new double[nConn*nExc]; //same here for writing to file
  std::vector<double> g_ee;
  std::vector<unsigned int> indInG_ee;
  std::vector<unsigned int> ind_ee;
  //int maxInColI_ee;
  
  //exc-inh
 // double *gAlltoAll_ei;
  double *garray_ei;
  std::vector<double> g_ei;
  std::vector<unsigned int> indInG_ei;
  std::vector<unsigned int> ind_ei;
  //int maxInColI_ei;
 
  //inh-exc
  //double *gAlltoAll_ie;
  double *garray_ie;
  std::vector<double> g_ie;
  std::vector<unsigned int> indInG_ie;
  std::vector<unsigned int> ind_ie;
  //int maxInColI_ie;
 
  //inh-inh
  //double *gAlltoAll_ii;
  double *garray_ii;
  std::vector<double> g_ii;
  std::vector<unsigned int> indInG_ii;
  std::vector<unsigned int> ind_ii;
  //int maxInColI_ii;

int main(int argc, char *argv[])
{
  if (argc != 6)
  {
    cerr << "usage: gen_syns_sparse_izhModel <nNeurons> <nConnPerNeuron> <meanSExc> <meanSInh>";
    cerr << " <outfile>" << endl;
    exit(1);
  }
  
  unsigned int nN= atoi(argv[1]);
  unsigned int nExc= (int)(4*nN/5);
  unsigned int nConn= atoi(argv[2]);
  double meangsynExc= atof(argv[3]);
  double meangsynInh= atof(argv[4]);


  //alltogether
  char filename[100];

  strcpy(filename,argv[5]);

  //ee
    
  char filename_ee[100];
  char filename_index_ee[100];
  char filename_indInG_ee[100]; 
  //char filename_nonopt_ee[100];
  char filename_info_ee[100];  
  
  strcpy(filename_ee,filename);
  strcat(filename_ee,"_ee");
    
  strcpy(filename_index_ee,filename);
  strcat(filename_index_ee,"_ind_ee");

  strcpy(filename_indInG_ee,filename);
  strcat(filename_indInG_ee,"_indInG_ee");

  /*strcpy(filename_nonopt_ee,filename);
  strcat(filename_nonopt_ee,"_nonopt_ee");
*/
  strcpy(filename_info_ee,filename);
  strcat(filename_info_ee,"_info_ee");
  
  ofstream os_ee(filename_ee, ios::binary);
  ofstream os_index_ee(filename_index_ee, ios::binary);
  ofstream os_indInG_ee(filename_indInG_ee, ios::binary);
  //ofstream os_nonopt_ee(filename_nonopt_ee, ios::binary);
  ofstream os_info_ee(filename_info_ee, ios::binary);
  
  //ei
  char filename_ei[100];
  char filename_index_ei[100];
  char filename_indInG_ei[100]; 
  //char filename_nonopt_ei[100];
  char filename_info_ei[100];  
  
  strcpy(filename_ei,filename);
  strcat(filename_ei,"_ei");
  
  strcpy(filename_index_ei,filename);
  strcat(filename_index_ei,"_ind_ei");

  strcpy(filename_indInG_ei,filename);
  strcat(filename_indInG_ei,"_indInG_ei");

  /*strcpy(filename_nonopt_ei,filename);
  strcat(filename_nonopt_ei,"_nonopt_ei");*/

  strcpy(filename_info_ei,filename);
  strcat(filename_info_ei,"_info_ei");
  
  ofstream os_ei(filename_ei, ios::binary);
  ofstream os_index_ei(filename_index_ei, ios::binary);
  ofstream os_indInG_ei(filename_indInG_ei, ios::binary);
  //ofstream os_nonopt_ei(filename_nonopt_ei, ios::binary);
  ofstream os_info_ei(filename_info_ei, ios::binary);
  
  //ie
  char filename_ie[100];
  char filename_index_ie[100];
  char filename_indInG_ie[100]; 
  //char filename_nonopt_ie[100];
  char filename_info_ie[100];  
  
  strcpy(filename_ie,filename);
  strcat(filename_ie,"_ie");
  
  strcpy(filename_index_ie,filename);
  strcat(filename_index_ie,"_ind_ie");

  strcpy(filename_indInG_ie,filename);
  strcat(filename_indInG_ie,"_indInG_ie");

  /*strcpy(filename_nonopt_ie,filename);
  strcat(filename_nonopt_ie,"_nonopt_ie");*/

  strcpy(filename_info_ie,filename);
  strcat(filename_info_ie,"_info_ie");
  
  ofstream os_ie(filename_ie, ios::binary);
  ofstream os_index_ie(filename_index_ie, ios::binary);
  ofstream os_indInG_ie(filename_indInG_ie, ios::binary);
  //ofstream os_nonopt_ie(filename_nonopt_ie, ios::binary);
  ofstream os_info_ie(filename_info_ie, ios::binary);
    
  //ii
  char filename_ii[100];
  char filename_index_ii[100];
  char filename_indInG_ii[100]; 
  //char filename_nonopt_ii[100];
  char filename_info_ii[100];  
  
      
  strcpy(filename_ii,filename);
  strcat(filename_ii,"_ii");
  
  strcpy(filename_index_ii,filename);
  strcat(filename_index_ii,"_ind_ii");

  strcpy(filename_indInG_ii,filename);
  strcat(filename_indInG_ii,"_indInG_ii");

  /*strcpy(filename_nonopt_ii,filename);
  strcat(filename_nonopt_ii,"_nonopt_ii");*/

  strcpy(filename_info_ii,filename);
  strcat(filename_info_ii,"_info_ii");
  
  ofstream os_ii(filename_ii, ios::binary);
  ofstream os_index_ii(filename_index_ii, ios::binary);
  ofstream os_indInG_ii(filename_indInG_ii, ios::binary);
  //ofstream os_nonopt_ii(filename_nonopt_ii, ios::binary);
  ofstream os_info_ii(filename_info_ii, ios::binary);  
  
  //gAlltoAll = new double[nN*nN];
  garray = new double[nConn]; 
  //g = new double[nConn*nN]; 
  ind = new unsigned int[nConn*nN]; 
  
  //exc-exc
  //gAlltoAll_ee = new double[nExc*nExc];
  garray_ee = new double[nConn]; 
  std::vector<double> g_ee;
  std::vector<unsigned int> indInG_ee;
  std::vector<unsigned int> ind_ee;
  
  //exc-inh
  //gAlltoAll_ei = new double[nExc*nInh];
  garray_ei = new double[nConn];
  std::vector<double> g_ei;
  std::vector<unsigned int> indInG_ei;
  std::vector<unsigned int> ind_ei;
 
  //inh-exc
 // gAlltoAll_ie = new double[nInh*nExc];
  garray_ie = new double[nConn];
  std::vector<double> g_ie;
  std::vector<unsigned int> indInG_ie;
  std::vector<unsigned int> ind_ie;

  //inh-inh
 // gAlltoAll_ii = new double[nInh*nInh];
  garray_ii = new double[nConn];
  std::vector<double> g_ii;
  std::vector<unsigned int> indInG_ii;
  std::vector<unsigned int> ind_ii;
  
  cerr << "# call was: ";
  for (int i = 0; i < argc; i++) cerr << argv[i] << " ";
  cerr << endl;

  indInG_ee.push_back(0);
  indInG_ei.push_back(0); 
  indInG_ie.push_back(0); 
  indInG_ii.push_back(0);  
 
  
  unsigned int sum_ee=0;
  unsigned int sum_ei=0;
  unsigned int sum_ie=0;
  unsigned int sum_ii=0;

  //number of pre-to-post is controlled but post-to-pre should be controlled by counting the number of connections for each postsynaptic neuron
  unsigned int *precount = new unsigned int[nN]; 

  for (unsigned int i= 0; i < nN; i++) {
  	precount[i]=0;
  }


  for (unsigned int i= 0; i < nN; i++) {
 		//reservoir sampling to choose nConn random connections for each neuron
 		for (unsigned int j=0 ; j< nConn; j++){
    	gsyn=R.n();
    	
    	if (i<nExc) {
        	gsyn*=meangsynExc;
        }
        else{
        	gsyn*=meangsynInh;
        }
    	garray[j]=gsyn;
   	ind[i*nConn+j] = j;
	   // gAlltoAll[i*nN+j] = gsyn;*/
	    precount[j]++;
    }
    for (unsigned int j=nConn ; j< nN; j++){
    	ind[j]=j;
    }
    for (unsigned int j=nConn; j< nN; j++){
			unsigned int rn = (unsigned int)(R.n()*(j+1));
      if (rn<nConn){
        /*if (precount[j]>nConn){
          cout << "postsynaptic neuron " << j << " has more than " << nConn << " connections..." << endl;
        }*/
        gsyn=R.n();
        if (i<nExc) {
        	gsyn*=meangsynExc;
        }
        else{
        	gsyn*=meangsynInh;
        }
        //cerr << i << ": create a connection for " << j << " to replace " << rn << endl; 
        //if (gAlltoAll[i*nN+ind[i*nConn+rn]]==0) 
        precount[ind[i*nConn+rn]]--;
     //   gAlltoAll[i*nN+ind[i*nConn+rn]]=0;
        precount[j]++;
        ind[i*nConn+rn]=j;
	      garray[rn]=gsyn;
	   //   gAlltoAll[i*nN+j]=gsyn;
	   //   gAlltoAll[i*nN+rn]=0;
      }
     /* else{
      	gAlltoAll[i*nN+rn]=gsyn;
      }*/
    }   
    
    //connectivity is set for the presynaptic neuron. Now push it to subgroups of populations

   	//cout << "nexc: " << nExc<< ", nInh: " << nInh << endl;
 /*   for (int p=0;p<nConn*nN;p++){
    if (p%nConn==0) cout << " for line "<< p <<"\n";
  	cout << ind[p] << " ";
  }*/
    for (unsigned int j=0 ; j< nConn; j++){
      if ((i<nExc)&&(ind[i*nConn+j]<nExc)){ //exc-exc
        g_ee.push_back(garray[j]);
        //gAlltoAll_ee[i*nExc+ind[i*nConn+j]]=garray[j];
        ind_ee.push_back(ind[i*nConn+j]);
        sum_ee++;
      }
      
      if ((i<nExc)&&(ind[i*nConn+j]>=nExc)){ //exc-inh
        g_ei.push_back(garray[j]);
        //gAlltoAll_ei[i*nInh+(ind[i*nConn+j]-nExc)]=garray[j];
        ind_ei.push_back(ind[i*nConn+j]-nExc);
        sum_ei++;
      }
      
      if ((i>=nExc)&&(ind[i*nConn+j]<nExc)){ //inh-exc
        g_ie.push_back(garray[j]);
        //gAlltoAll_ie[(i-nExc)*nExc+(ind[i*nConn+j])]=garray[j];
        ind_ie.push_back(ind[i*nConn+j]);
        sum_ie++;
      }
      
      if ((i>=nExc)&&(ind[i*nConn+j]>=nExc)){ //inh-inh
        g_ii.push_back(garray[j]);
        //gAlltoAll_ii[(i-nExc)*nInh+(ind[i*nConn+j]-nExc)]=garray[j];
        ind_ii.push_back(ind[i*nConn+j]-nExc);
        sum_ii++;
      }
    }
    
    if (i<nExc){
    	indInG_ee.push_back(sum_ee);
    	indInG_ei.push_back(sum_ei);
    }
    else{ 
   		indInG_ie.push_back(sum_ie); 
			indInG_ii.push_back(sum_ii);
		}
		
//    memcpy(g+i*nConn,garray,nConn*sizeof(double));   
	     //gOld[i*nMB+j]= 0.0f;
  }
 

  //os.write((char *)g, nN*nConn*sizeof(double));
  //os_nonopt.write((char *)gOld, nN*nN*sizeof(double));
  //os.close();
  
  //os_index.write((char *)ind, nN*nConn*sizeof(unsigned int));
  //os_indInG.write((char *)indInG, nN*sizeof(unsigned int));
  //os_nonopt.write((char *)gAlltoAll, nN*nN*sizeof(double));
  
 //os_index.close();
  //os_indInG.close();
  //os_nonopt.close();
 
  /*cout << "\nprinting g:\n" << endl; 
  for (unsigned int j=0;j<nConn*nN;j++){
    if (j%nConn==0) cout << "\n";
  	cout << g[j] << " ";
  }*/
  
  /*double *gAlltoAll_ee = new double[nExc*nExc];
  double *garray_ee = new double[nConn]; 
  std::vector<double> g_ee;
  std::vector<unsigned int> indInG_ee;
  std::vector<unsigned int> ind_ee;*/
 
 
  //ee
  size_t sz = g_ee.size();
  cout << "ee vect.size: " << sz << endl;
  os_info_ee.write((char *)&sz,sizeof(size_t));
  os_ee.write(reinterpret_cast<const char*>(&g_ee[0]), sz * sizeof(g_ee[0]));
  sz = ind_ee.size();
  cout << "ee ind size: " << sz << endl;
  os_index_ee.write(reinterpret_cast<const char*>(&ind_ee[0]), sz * sizeof(ind_ee[0]));
  sz = indInG_ee.size();
  cout << "ee count size: " << sz << endl;
  os_indInG_ee.write(reinterpret_cast<const char*>(&indInG_ee[0]), sz * sizeof(indInG_ee[0]));
  //os_nonopt_ee.write((char *)gAlltoAll_ee, nExc*nExc*sizeof(double));
  
  os_ee.close();
  os_index_ee.close();
  os_indInG_ee.close();
  //os_nonopt_ee.close();
  os_info_ee.close();
  
  //ei
  sz = g_ei.size();
  cout << "ei vect.size: " << sz << endl;
  os_info_ei.write((char *)&sz,sizeof(size_t));
  os_ei.write(reinterpret_cast<const char*>(&g_ei[0]), sz * sizeof(g_ei[0]));
  sz = ind_ei.size();
  cout << "ei ind size: " << sz << endl;
  os_index_ei.write(reinterpret_cast<const char*>(&ind_ei[0]), sz * sizeof(ind_ei[0]));
  sz = indInG_ei.size();
  cout << "ei count size: " << sz << endl;
  os_indInG_ei.write(reinterpret_cast<const char*>(&indInG_ei[0]), sz * sizeof(indInG_ei[0]));
  //os_nonopt_ei.write((char *)gAlltoAll_ei, nExc*nInh*sizeof(double));
  
  os_ei.close();
  os_index_ei.close();
  os_indInG_ei.close();
  //os_nonopt_ei.close();
  os_info_ei.close();
  
  //ie
  sz = g_ie.size();
  cout << "ie vect.size: " << sz << endl;
  os_info_ie.write((char *)&sz,sizeof(size_t));
  os_ie.write(reinterpret_cast<const char*>(&g_ie[0]), sz * sizeof(g_ie[0]));
  sz = ind_ie.size();
  cout << "ie ind size: " << sz << endl;
  os_index_ie.write(reinterpret_cast<const char*>(&ind_ie[0]), sz * sizeof(ind_ie[0]));
  sz = indInG_ie.size();
  cout << "ie count size: " << sz << endl;
  os_indInG_ie.write(reinterpret_cast<const char*>(&indInG_ie[0]), sz * sizeof(indInG_ie[0]));
  //os_nonopt_ie.write((char *)gAlltoAll_ie, nInh*nExc*sizeof(double));
  
  os_ie.close();
  os_index_ie.close();
  os_indInG_ie.close();
  //os_nonopt_ie.close();
  os_info_ie.close();
  
  //ii
  sz = g_ii.size();
  cout << "ii vect.size: " << sz << endl;
  os_info_ii.write((char *)&sz,sizeof(size_t));
  os_ii.write(reinterpret_cast<const char*>(&g_ii[0]), sz * sizeof(g_ii[0]));
  sz = ind_ii.size();
  cout << "ii ind size: " << sz << endl;
  os_index_ii.write(reinterpret_cast<const char*>(&ind_ii[0]), sz * sizeof(ind_ii[0]));
  sz = indInG_ii.size();
  cout << "ii count size: " << sz << endl;
  os_indInG_ii.write(reinterpret_cast<const char*>(&indInG_ii[0]), sz * sizeof(indInG_ii[0]));
  //os_nonopt_ii.write((char *)gAlltoAll_ii, nInh*nInh*sizeof(double));
  
  os_ii.close();
  os_index_ii.close();
  os_indInG_ii.close();
  //os_nonopt_ii.close();
  os_info_ii.close();
  
 // delete[] g;
 // delete[] gAlltoAll;
  delete[] garray;
  delete[] ind;
  
  /*delete[] gAlltoAll_ee;
  delete[] gAlltoAll_ei;
  delete[] gAlltoAll_ie;
  delete[] gAlltoAll_ii;*/
  
  delete[] garray_ee;
  delete[] garray_ei;
  delete[] garray_ie;
  delete[] garray_ii;

  //delete[] indInG;
  /*
  cout << "\nprinting g_ee" << endl;
  printVector(g_ee);
  cout << "\nprinting g_ei" << endl;
  printVector(g_ei);
  cout << "\nprinting g_ie" << endl;
  printVector(g_ie);
  cout << "\nprinting g_ii" << endl;
  printVector(g_ii);

	cout << endl;

  cout << "\nprinting indInG_ee" << endl;
  printVector(indInG_ee);
  cout << "\nprinting indInG_ei" << endl;
  printVector(indInG_ei);
  cout << "\nprinting indInG_ie" << endl;
  printVector(indInG_ie);
  cout << "\nprinting indInG_ii" << endl;
  printVector(indInG_ii);
  
	cout << endl;

  cout << "\nprinting ind_ee" << endl;
  printVector(ind_ee);
  cout << "\nprinting ind_ei" << endl;
  printVector(ind_ei);
  cout << "\nprinting ind_ie" << endl;
  printVector(ind_ie);
  cout << "\nprinting ind_ii" << endl;
  printVector(ind_ii);
  
  cout << "printing precount:" << endl;
  for (unsigned int j=0;j<nN;j++){
    cout <<  precount[j] << " ";
  }
  cout << endl;
  */
  return 0;
}

int printVector(vector<unsigned int>& v){
	for (unsigned int i=0;i<v.size();i++){
		cout << v[i] << " " ;
	}
	cout << endl;
	return 0;
}

int printVector(vector<double>& v){
  for (unsigned int i=0;i<v.size();i++){
		cout << v[i] << " ";
	}
	cout << endl;
	return 0;
}
