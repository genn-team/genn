function plotRaster(name, nPN, nKC, nLHI, nDN, tRange)
  path(path, '../../matlab');
  disp(path)
  offset= 0;
  dataname= [ '../' name '_output/' name '.out.st' ]
  plotStc(dataname, tRange, [ offset offset+nPN-1 ], [ 1 0 0 ]);
  offset= offset+nPN;
  plotStc(dataname, tRange, [ offset offset+nKC-1 ], [ 0 0.6 0 ]);
  offset= offset+nKC;
  plotStc(dataname, tRange, [ offset offset+nLHI-1 ], [ 0 0 1 ]);
  offset= offset+nLHI;
  plotStc(dataname, tRange, [ offset offset+nDN-1 ], [ 0 0 0 ]);
  offset= offset+nDN;
  ylim([ -1 offset ]);
  