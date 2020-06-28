% %% 
% clc;clear all;close all;
% % The first try.
% % The world will move.
% 
% org_point_shift = 20;
% radius = 50;
% word_grid = org_point_shift-radius:org_point_shift+radius;
% [X,Y] = meshgrid(word_grid);
% 
% high_reward = 20;
% reward = -sqrt((X-org_point_shift).^2+(Y-org_point_shift).^2)+high_reward;
% surf(X,Y,reward)
% 
% %% 
% % 
% clc;clear all;close all;
% % The world doesn't move.
% 
% 
% radius = 150;
% word_grid = -radius:radius;
% [X,Y] = meshgrid(word_grid);
% high_reward = 300;
% 
% point1_x = 135;
% reward1 = -sqrt((X-point1_x).^2+(Y-point1_x).^2)+high_reward;
% 
% point2_x = -135;
% reward2 = -sqrt((X-point2_x).^2+(Y-point2_x).^2)+high_reward;
% reward = reward1+reward2;
% mask = reward >20;
% surf(X,Y,reward.*mask)

%% 
% 
% Create a grid world
scale = 10;
width = 4;

radius = 300/scale;
[X,Y] = meshgrid(-radius:radius);
high_reward = 5;

% Point 1
point1_x = -150/scale;
point1_y = 150/scale;
reward1 = pdist([[X,Y];[point1_x,point1_y]], 'cityblock')/width + high_reward;

reward1(reward1<0)=0;

% Point 2
point2_x = -150/scale;
point2_y = -150/scale;
reward2 = pdist([[X,Y];[point2_x,point2_y]], 'cityblock')/width + high_reward;
reward2(reward2<0)=0;

% Point 3
point3_x = 150/scale;
point3_y = -150/scale;
reward3 = pdist([[X,Y];[point3_x,point3_y]], 'cityblock')/width + high_reward;
reward3(reward3<0)=0;

% Point 4
point4_x = 150/scale;
point4_y = 150/scale;
reward4 = pdist([[X,Y];[point4_x,point4_y]], 'cityblock')/width + high_reward;
reward4(reward4<0)=0;

% rewards are made a summary.
reward = reward1+reward2+reward3+reward4;

% plot the 3D world
surf(X+300/scale,Y+300/scale,reward)

