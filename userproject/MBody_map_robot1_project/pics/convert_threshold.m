function convert_threshold(basename, min, max, qunt, outname)

ad= [];
for i=min:max
    name= [ basename num2str(i) '.png' ];
    d= imread(name);
    figure; 
    colormap('gray');
    imagesc(d);    
    threshold= quantile(reshape(d,1,1024),qunt);
    up= find(d <= threshold);
    down= find(d > threshold);
    d= im2double(d);
    d(up)= 1.0;
    d(down)= 0.0;
    figure;
    imagesc(d);
    ad= [ ad; d ];
end
	
save( outname, 'ad', '-ascii');
