function plotSynHist(name, tme)
% plot a histogram of synaptic strength
% name: name of the run
% tme: the time of the synaptic strength snapshot
  
  path(path, '../../matlab');
  offset= 1;
  dataname= [ '../' name '_output/' name '.' num2str(tme) '.syn' ]
  f= fopen(dataname);
  d= fread(f,'float');
  size(d)
  figure; 
  hist(d,100);
  
  