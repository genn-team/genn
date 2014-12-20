%function d= plotDetailed(name)
%  thename= [ '../' name '_output/' name '.out.dat' ];
%  d= load(thename);
  nKCColl= 1;
  nLBColl= 10;
  % first nKCColl* 4 variable V, m, h, n of KCs
  % then nLBColl* 5 vars inSynDN, VDN, mDN, hDN, nDN
  % then nKCColl*nLBColl*2 vars gKCDN, gRawKCDN
  t= d(:,1);
  start= 2
  stop= start+(nKCColl-1)*4
  VKC= d(:,start:4:stop);
  start= stop+4+1 
  stop= start+(nLBColl-1)*5
  VDN= d(:,start:5:stop);
  start= stop+4;
  gKCDN= d(:,start:2:end);
  start= start+1;
  gRawKCDN= d(:,start:2:end);
  figure;
  hold on;
  for i=1:nKCColl
    plot(t, VKC(:,i)+100*(i-1));
    line([ t(1) t(end)], 100*(i-1)*[ 1 1 ],'color','k')
  end
  
  figure;
  hold on;
  for i=1:nLBColl
    plot(t, VDN(:,i)+100*(i-1));
    line([ t(1) t(end)], 100*(i-1)*[ 1 1 ],'color','k')
  end
  
  figure;
  hold on;
  for i=1:nKCColl*nLBColl
    plot(t, gKCDN(:,i)+0.01*(i-1));
  end
  
  figure;
  hold on;
  for i=1:nKCColl*nLBColl
    plot(t, gRawKCDN(:,i)+0.01*(i-1));
  end
  
  % now a figure for one specific synapse
  figure;
  subplot(2,1,1);
  hold on;
  plot(t, VKC(:,1)+100);
  plot(t, VDN(:,1));
  subplot(2,1,2);
  plot(t, gKCDN(:,1));
  