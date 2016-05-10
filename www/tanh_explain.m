x=-10:0.1:10;

figure
plot(x,tanh(x)+1,'r','linewidth',2);
set(gca,'visible','off');
print -depsc 'tanh.eps'