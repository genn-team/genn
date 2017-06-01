f=fopen('../test1_output/test1.inpat');
d= fread(f,'float');
d2= reshape(d,100,100);
for i=1:100
  axes('position',[ 0.02 i/100 0.96 0.01 ]);
  bar(d2(:,i));
end

