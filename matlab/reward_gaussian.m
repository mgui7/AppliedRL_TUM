[map1,map2] = meshgrid(1:60);
targets = [15,45;10,15;45,20];
res = zeros(60*3);

for i = 1:3
    x = targets(i,1);
    y = targets(i,2);
    res(x:x+120,y:y+120) = res(x:x+120,y:y+120) + fspecial('gaussian', [121 121], 9);
end

mesh(res(61:120,61:120))
xlabel('X-axis')
ylabel('Y-axis')
zlabel('Reward')
title('Gaussian Distribution')
grid on