#!/bin/bash
#call this as:
#$ bash testprojects.sh "what is new in this run" 2>&1 |tee outputtestscript
#then:
#$ grep -i warning outputtestscript
#$ grep -i error outputtestscript
#$ grep -i seg outputtestscript
#
custommsg=$1 #Reminder about what is being tried, to be written in the .time file
set -e #exit if error or segfault
BmDir=$GENN_PATH/userproject/refProjFiles

printf "This script may take a while to finish all tests if you use it with default parameters. \n\
You may consider shortening the duration of simulation in classol_sim.h for the MBody examples.\n\n"

firstrun=false;

echo "Making tools..."
cd tools
make clean && make
cd ..

echo "Checking if benchmarking directory exists..."
if [ -d "$BmDir" ]; then
  echo "benchmarking directory exists. Using input data from" $BmDir 
  printf "\n"
else
  mkdir -p $BmDir
  echo "Benchmarking directory does not exist. Creating a new one at" $BmDir " and running the test only once (first run for reference)."
  printf "\n"
  firstrun=true;
fi 

echo "firstrun " $firstrun

printf "\nTEST 1: TEST BY CREATING NEW INPUT PATTERNS"
printf "\n\n*********************** Testing MBody1 ****************************\n"
cd MBody1_project
make clean && make
printf "\n\n####################### MBody1 GPU ######################\n"
if [ "$firstrun" = false ]; then
  echo ${custommsg} >> testing_output/testing.time
  printf "With new setup... \n"  >> testing_output/testing.time
fi
./generate_run 1 100 1000 20 100 0.0025 testing MBody1 0 FLOAT 0
cp testing_output/testing.out.st testing_output/testing.out.st.GPU
printf "\n\n####################### MBody1 CPU ######################\n"
./generate_run 0 100 1000 20 100 0.0025 testing MBody1 0 FLOAT 0
cp testing_output/testing.out.st testing_output/testing.out.st.CPU 

printf "\n\n*********************** Testing MBody_userdef ****************************\n"
cd ../MBody_userdef_project
make clean && make
if [ "$firstrun" = false ]; then
  echo ${custommsg} >> testing_output/testing.time
  printf "With new setup... \n"  >> testing_output/testing.time
fi
printf "\n\n####################### MBody_userdef GPU ######################\n"
./generate_run 1 100 1000 20 100 0.0025 testing MBody_userdef 0 FLOAT 0
cp testing_output/testing.out.st testing_output/testing.out.st.GPU
printf "\n\n####################### MBody_userdef CPU ######################\n"
./generate_run 0 100 1000 20 100 0.0025 testing MBody_userdef 0 FLOAT 0
cp testing_output/testing.out.st testing_output/testing.out.st.CPU
printf "\n\n*********************** Testing Izh_sparse 10K neurons****************************\n"
cd ../Izh_sparse_project
make clean && make
if [ "$firstrun" = false ]; then
  echo ${custommsg} >> testing_output/testing.time
  printf "With new setup... \n"  >> testing_output/testing.time
fi
printf "\n\n####################### Izh_sparse 10K GPU ######################\n"
./generate_run 1 10000 1000 1 testing Izh_sparse 0 FLOAT 0
#cp testing_output/testing.out.st testing_output/testing.out.st.GPU
printf "\n\n####################### Izh_sparse 10K CPU ######################\n"
./generate_run 0 10000 1000 1 testing Izh_sparse 0 FLOAT 0
#cp testing_output/testing.out.st testing_output/testing.out.st.CPU
if [ "$firstrun" = true ]; then
  cp -R inputfiles inputfiles10K
fi  

#need to recompile if we want to rerun with different number of neurons. To be revisited...
#printf "\n\n*********************** Testing Izh_sparse 1K neurons****************************\n"
#printf "\n\n####################### Izh_sparse 1K GPU ######################\n"
#./generate_run 1 1000 1000 1 outdir Izh_sparse 0 0
#if [ "$firstrun" = true ]; then
#  cp -R inputfiles inputfiles1K
#fi 
#cp testing/testing.out.st testing/testing.out.st.GPU
#printf "\n\n####################### Izh_sparse 1K CPU ######################\n"
#./generate_run 0 1000 1000 1 outdir Izh_sparse 0 0
#cp testing/testing.out.st testing/testing.out.st.CPU

cd ..  
#if this is the first time, copy reference input files into the benchmarking directory
  
if [ "$firstrun" = true ]; then
  printf "Benchmarking is run for the first time. Creating reference input files with the results of these runs. \nIf any error occurs please delete the benchmarking directory and run the script after the values are corrected.\nCopying reference files...\n"
  cp -R MBody1_project/testing_output $BmDir/MBody1
  cp -R MBody1_project/testing_output $BmDir/MBody_userdef #use the same input for MBody_userdef and MBody1
  cp -R Izh_sparse_project/testing_output $BmDir/Izh_sparse
  cp -R Izh_sparse_project/inputfiles10K $BmDir/Izh_sparse/inputfiles10K
  #cp -R PoissonIzh_project/testing $BmDir/PoissonIzh
  #cp -R OneComp_project/testing $BmDir/PoissonIzh_project
  #cp -R PoissonIzh_project/testing $BmDir/PoissonIzh_project
  #cp -R SynDelay_project/testing $BmDir/SynDelay
  echo "First run complete!"
  exit 0;
else
  cp MBody1_project/testing_output/testing.time $BmDir/MBody1/testing.time
  cp MBody_userdef_project/testing_output/testing.time $BmDir/MBody_userdef/testing.time
  cp Izh_sparse_project/testing_output/testing.time $BmDir/Izh_sparse/testing.time
  #if you add new tests, don't forget to copy the output to your ref files 
fi

printf "\nTEST 2: TEST BY USING REFERENCE INPUT PATTERNS\n"  
#be careful with network sizes
#if you add new tests, don't forget to copy the output to your ref files 
cd MBody1_project
cp -R $BmDir/MBody1/* testing_output/
printf "With reference setup... \n"  >> testing_output/testing.time
printf "\n\n####################### MBody1 GPU TEST 2 ######################\n"
model/classol_sim testing 1
printf "\n\n####################### MBody1 CPU TEST 2 ######################\n"
model/classol_sim testing 0
cd ../MBody_userdef_project
cp -R $BmDir/MBody_userdef/* testing_output/
printf "With reference setup (same as MBody1 as well)... \n"  >> testing_output/testing.time
printf "\n\n####################### MBody_userdef GPU TEST 2 ######################\n"
model/classol_sim testing 1
printf "\n\n####################### MBody_userdef CPU TEST 2 ######################\n"
model/classol_sim testing 0
cd ../Izh_sparse_project
cp -R $BmDir/Izh_sparse/* testing_output/
printf "With reference setup (input is still random, so the results are not expected to be identical)... \n"  >> testing_output/testing.time
cp -R $BmDir/Izh_sparse/inputfiles10K/* inputfiles/
model/Izh_sim_sparse testing 1
model/Izh_sim_sparse testing 0
cd ..
#cp -R $BmDir/PoissonIzh PoissonIzh_project/testing
#cp -R $BmDir/OneComp OneComp_project/testing
#cp -R $BmDir/SynDelay SynDelay_project/testing

cp MBody1_project/testing_output/testing.time $BmDir/MBody1/testing.time
cp MBody_userdef_project/testing_output/testing.time $BmDir/MBody_userdef/testing.time
cp Izh_sparse_project/testing_output/testing.time $BmDir/Izh_sparse/testing.time

printf "\nMBody1 time tail\n"
tail -n 23 MBody1_project/testing_output/testing.time
printf "\nMBody_userdef time tail\n"
tail -n 18 MBody_userdef_project/testing_output/testing.time
printf "\nIzh_sparse time tail\n"
tail -n 18 Izh_sparse_project/testing_output/testing.time
  
echo "Test complete! Checking if weekly copy of the output is necessary..."

monthandyear=`date +'%W_%m%y'`

echo ${monthandyear}
FILE="outputtestscript_"${monthandyear}
if [ -f $FILE ];
then
  echo $FILE" exists."
else
  cp outputtestscript $FILE
  echo "Created a copy of the output message at "${FILE}
fi
