function plotSt(name,rng,rngy,color)
% name of the spike time file
% time range to plot [ tmin, tmax ]
% rng of neuron IDs to plot [ IDmin IDmax ]

  data= load(name);
  hold on;
  tline= zeros(2,2);
  thestart= find(data(:,1) > rng(1));
  theend= find(data(:,1) < rng(2));
  for i=thestart(1):theend(length(theend))
    if ((data(i,2) >= rngy(1)) && (data(i,2) <= rngy(2)))
      tline(1,1)=data(i,1);
      tline(1,2)=data(i,2);
      tline(2,1)=data(i,1);
      tline(2,2)= data(i,2)-0.8;
      line(tline(:,1),tline(:,2),'linewidth',2,'color',color);
    end
  end
%  ylim(rngy);
  
