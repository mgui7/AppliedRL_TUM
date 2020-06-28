Screen_width = 600;
Screen_length = 600;
robot_size = 40;
T = Screen_length - robot_size;
A = 300;


y = linspace(1,Screen_length,600);
x = linspace(300,300,600);
x1 = A*sin((2*pi/T)*(y-robot_size/2))+Screen_width/2;
x2 = A*sin((2*pi/T)*(y-robot_size/2))/2+Screen_width/2;

x_start_end = [300,300];
y_start_end = [20,580];


scatter(x_start_end,y_start_end);

title("Robot Route");
xlabel("x");
ylabel("y");

hold on
plot(x,y,x1,y,x2,y);
hold on
hold off