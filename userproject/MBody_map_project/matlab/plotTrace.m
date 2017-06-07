function plotTrace(name, which, tRange)
% plot a voltage trace
% name is the name of the run
% which is a vector of neuron IDs (starting with 0)
% tRange is the time range in which to plot
  dataname= [ '../' name '_output/' name '.out.Vm' ]
  d= load(dataname);
  size(d)
  trng= find((d(:,1) > tRange(1)) & (d(:,1) < tRange(2)));
  figure; hold on;
  tno= length(which);
  for i= 1:tno
    id= which(i);
    plot(d(trng,1), d(trng,id+2)+(i-1)*150);
  end