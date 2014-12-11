import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os.path
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle

numArgumentsProvided =  len(sys.argv)
if  numArgumentsProvided < 8:
    print 'usage: python ' + sys.argv[0] + ' <srcDir> <srcFilename> <totalNeurons> <endtime> majorGridY minorGridY <displayActivationBars_1_0> [boxstart] [boxend]'
    sys.exit() 

srcDir = sys.argv[1]
srcFilename = sys.argv[2]
srcPath = srcDir + '/' + srcFilename
print 'Looking for file ' , srcPath
#x,y = np.loadtxt(fname=srcPath,delimiter=',', usecols=(0,1),unpack=True)

# or alternatively
spikes = np.loadtxt(fname=srcPath,delimiter=',')
x= spikes[:,0]
y = spikes[:,1]

displayActivationBars = int(sys.argv[7])==1

fig  = plt.figure(srcFilename,figsize=(23,13))

if displayActivationBars:
   #draw a raster plot to the left (with activation bars to the right)
   plt.axes([0.03,0.03,0.5,0.94],zorder=1)
else :
   #draw a full width raster plot (with equivalent heatmap below)
   plt.axes([0.03,0.5,0.94,0.46])

axRaster = plt.gca() #get hold of current Axes
axRaster.grid(1)
#axRaster.set_axisbelow(True) #draw grid behind plot

totalNeurons = int(sys.argv[3])
print 'Total neurons:', totalNeurons
plt.ylim(0,totalNeurons)

endTime = float(sys.argv[4])
plt.xlim(0,endTime)

majorGridY = int(sys.argv[5])
axRaster.yaxis.set_major_locator(MultipleLocator(majorGridY))

clusterSize = int(sys.argv[6])
minorGridY = clusterSize
axRaster.yaxis.set_minor_locator(MultipleLocator(minorGridY)) #draw horiz lines to demarcate clusters

#highlight region where samples align 
axRaster.axvspan(30000, 30500, facecolor='g', alpha=0.2)

axRaster.xaxis.set_major_locator(MultipleLocator(10000))
axRaster.xaxis.set_minor_locator(MultipleLocator(5000))

axRaster.grid(which='major', axis='x', linewidth=0.5, linestyle='-', color='0.5')
axRaster.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.85')
axRaster.grid(which='major', axis='y', linewidth=1.5, linestyle='-', color='0.5')
axRaster.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
#axRaster.set_xticklabels([])
#axRaster.set_yticklabels([])


if displayActivationBars:
   alph=1.0 #for short, close up raster we can used black dots
else:
   alph=0.1 # for long wide raster we use transparent to stop everything just going black
  
plt.scatter(x,y,c='black',s=1,marker='.',alpha=alph)
plt.title(srcFilename)


if displayActivationBars:
   #Draw a bargraph alongside raster plot showing activation levels in AN, RN and PN

   #make historgram from the spike counts in each cluster
   numClusters = totalNeurons/clusterSize
   print 'numClusters ',numClusters
   numBins = numClusters 
   clusterSpikeCounts, bin_edges = np.histogram(y,bins=numBins)
   print 'clusterSpikeCounts:', clusterSpikeCounts
   print 'bin_edges:', bin_edges
   barWidth = 0.8
   offset = (1 - barWidth) /2 
   vrLeft=0.532
   vrBottom=0.03
   vrWidth=0.25

   indices = np.arange(numBins)
   axRnVrActivation = plt.axes([0.532,0.03,0.25,0.94],zorder=0)
   axRnVrActivation.xaxis.set_visible(False) 
   axRnVrActivation.yaxis.set_minor_locator(MultipleLocator(1))
   axRnVrActivation.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
   plt.barh(indices + offset, clusterSpikeCounts, barWidth, color='black')


else:
   #Draw a full width heatmap version below the raster plot
   axHeatMap = plt.axes([0.03,0.02,0.94,0.46])
   #plt.ylim(0,totalNeurons)

   endTimeSec = endTime/1000
   sampleFreqHz = 2 
   resolution = 1 #bins per data sample presented
   timeBins = resolution * endTimeSec * sampleFreqHz
   print 'timeBins:', timeBins

   clusterBins  = totalNeurons / clusterSize ;
   print 'clusterBins:', clusterBins
   heatmap, xedges, yedges = np.histogram2d(x,y, bins=(timeBins*3,clusterBins), normed=False)

   #stretchY  = 1.5 #use more pixels in vertical direction to make plot higher. 1.5 is good for 40sec plot, 43VRs (6360 neurons)
   stretchY  = float(9540.0/totalNeurons) * float(endTime/40000) #adjust size if longer plot is supplied (e.g. 300sec) or more VRs used
   extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]*stretchY] 

   axHeatMap.imshow(heatmap.T, origin='lower', extent=extent, interpolation='none', cmap=cm.jet ) # jet, gist_gray_r, hot, rainbow
   #axHeatMap.imshow(heatmap.T, origin='lower', extent=extent, cmap=cm.jet )

   #demarcate RN, PN, AN regions
   axHeatMap.xaxis.set_major_locator(MultipleLocator(10000))
   axHeatMap.grid(which='major', axis='x', linewidth=0.5, linestyle='-', color='0.5')
   axHeatMap.xaxis.set_minor_locator(MultipleLocator(5000))
   axHeatMap.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
   axHeatMap.yaxis.set_major_locator(MultipleLocator(majorGridY*stretchY)) #for some reason stretching messes the yaxis
   axHeatMap.grid(which='major', axis='y', linewidth=0.5, linestyle='-', color='0.5')

   #demarcate the class evaluation zone on the heatmap
   if  numArgumentsProvided >= 10:
      startEvalTime = float(sys.argv[8])
      endEvalTime = float(sys.argv[9])
      #axHeatMap.axvline(linewidth=1,x=startEvalTime,color='r')
      #axHeatMap.axvline(linewidth=1,x=endEvalTime,color='r')
      startRegionAN = majorGridY*stretchY*2
      startRegionPN = majorGridY*stretchY
      endRegionAN =  totalNeurons*stretchY - 15
      #draw a box around AN indicating the evalution 
      axHeatMap.add_patch(Rectangle((startEvalTime, startRegionPN), endEvalTime-startEvalTime,endRegionAN-startRegionPN, edgecolor='r', fill=False, linewidth=1))


   #cax = axes([0.85, 0.1, 0.075, 0.8])
   #colorbar(cax=cax)

plt.show()

