[map1,map2] = meshgrid(1:60);
targets = [15,45;10,15;45,20];
res = zeros(60);
for i = 1:3
    tmp = (map1 - targets(i,1)).^2 + (map2 - targets(i,2)).^2;
    dist = tmp .^ 0.5;
    res = res + exp(-tmp * 0.01);
end

mesh(res-1)
xlabel('X-axis')
ylabel('Y-axis')
zlabel('Reward')
title('Exponential Function')
grid on