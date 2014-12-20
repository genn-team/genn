d{1}= load('../profile_FAST_CPU/testing.timingprofile');
d{2}= load('../profile_ALTERNATIVE_CPU/testing.timingprofile');
d{3}= load('../profile_SAFE_CPU/testing.timingprofile');
d{4}= load('../profile_FAST_GPU/testing.timingprofile');
d{5}= load('../profile_ALTERNATIVE_GPU/testing.timingprofile');
d{6}= load('../profile_SAFE_GPU/testing.timingprofile');

clr= {'r' 'g' 'b' 'k' 'm' 'c'}
for j=1:3
  figure;
  hold on;
  for i=1:6
    plot(d{i}(:,j),clr{i});
  end
end

