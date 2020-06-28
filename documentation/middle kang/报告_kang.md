强化学习算法中，有两个重要的因素，环境和agent。通过agent不断的在环境中实践和试错，持续优化策略policy，最终能够在环境中生存，并获得最高的reward。

由于特殊原因，在本学期无法使用真实的机器人和现实环境进行测学学习，因此在这个实验中，我们使用python，这种最流行的计算机语言进行试验。在强化学习领域，最重要的库应该是gym了。根据官网的介绍，Gym is a toolkit for developing and comparing reinforcement learning algorithms. 本次项目将使用gym已经搭建好的结构和提供的额外方法，来辅助实现算法。

## 第一个例子和最基础的结构

~~~python
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
~~~

这是来自官网的最简单的实例，这个环境展示的是一个pole和一个cart相互连接，pole在cart上做垂直圆周运动。

~~~python
env = gym.make('CartPole-v0')
~~~

第一步是为了获得一个环境，这个环境已经在gym中注册过了。但在本次实验中，环境的class将会和训练代码处于同一个path中，因此这一步可以在实际实验中省略。

~~~python
env.reset()
~~~

当环境变量被赋值的时候，init 和reset 函数会初始化整个世界和agent.

init函数，主要负责变量的初始化，例如状态初始化，界面初始化等等，但是，它不会对任何变量进行赋值。初始化具体值，将会被reset函数执行。例如，reset函数将会给agent的状态进行赋值，这个赋值可以是常量，也可以是一个随机变量。为了能够更好的训练agent，通常选择随机初始化。

~~~python
for _ in range(1000):
~~~

这一步开始对环境进行迭代，每一次迭代，环境和agent将会刷新。

~~~python
env.render()
~~~

使用render函数，将具体的动作与状态可视化出来。

~~~python
env.step(env.action_space.sample())
~~~

最重要的函数，输入是action，输出是state。在这个函数内部，agent会读取输入的action，并根据上一步的state，推算出下一步的state。其中的具体过程可以参考物理世界的模型，即重力，速度，加速度等等。清晰的结构并且选择合适的模型与参数，能够为RL算法的训练提供很好的帮助。

因为rl算法还没有被加入，因此这里的输入action使用随机动作来代替。

~~~python
env.close()
~~~

关闭整个环境。当agent超过了规定的state以后，整个过程结束，之后会结算reward，同时，render窗口会关闭，界面会重新刷新。

## 针对该项目的结构

Thought the standard example from gym，we recontructed and rebuild it according to our simulated robot's project.

~~~python
if __name__ == "__main__":
    env = MainEnv()
    for _ in range(MAX_EPISODES):
        s = env.reset()
        while True:
            action = rl.choose_action(s)
            for _ in range(SAM_STEP): 
                s_prime, reward, done = env.step(action)
                if DO_PLOT:env.render()

			s_prime = discretization(s_prime)

            rl.learn(s, action, reward, s_prime)

            s = s_prime
            
            if done: 
                env.close()
                break
~~~

首先，通过

~~~python
env = MainEnv()
~~~

创建新的类，然后进入for loop。一次for loop 循环中，机器人将会运行一个完整的过程。

~~~python
s = env.reset()
~~~

初始化状态，并获得最初的state。

~~~python
while True
~~~

进入while循环，如果本次流程不结束，while将不会终止。

~~~python
action = rl.choose_action(s)
~~~

在这一步中，rl算法会根据当前的当前的state，提供一个action。

~~~python
for _ in range(SAM_STEP): 
	s_prime, reward, done = env.step(action)
	if DO_PLOT:env.render()
~~~

每一次调用step函数，模拟agent都会移动一个单位时间，同时更新state，reward和done（判断本次流程是否结束的variable）。而因为我们使用的是discrete RL方法，因此不能输出每一时刻的状态（否则就是连续的），因此这里for循环被加入，并且根据全局变量SAM_STEP来模拟对机器人的一次观测（observation or sample）。

~~~python
s_prime = discretization(s_prime)
~~~

由于agent输出的状态是连续的并且state值是double的，我们需要使用discretization函数，将连续的状态转化为特定的离散状态。

~~~python
rl.learn(s, action, reward, s_prime)
~~~

The pre-states, latest states, reward and action are training date to train and update the q-table.

最后，

~~~python
s = s_prime
if done: 
    env.close()
    break
~~~

更新老的状态，并且判断流程是否结束，如果结束，done为True，render will close and the program starts to do the next episode.

## About render() function

render() function 主要负责绘制世界和机器人与机器手臂的运动。

render的显示逻辑如下图所示：

<img src="D:\OneDrive\文档\TUM\2020SS\Applied Reinforce learning\render_basic.png" alt="render_basic" style="zoom:40%;" />

如果create new viewer 则会运行如下内容：

~~~python
from gym.envs.classic_control import rendering
self.viewer = rendering.Viewer(SCREEN_W, SCREEN_H)
l, r, t, b       = -ROBOT_W / 2, ROBOT_W / 2, ROBOT_H / 2, -ROBOT_H / 2
robot            = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
self.robot_trans = rendering.Transform()
robot.add_attr(self.robot_trans)
self.viewer.add_geom(robot)
~~~

首先import 由gym提供的rendering库，通过这个库中的函数，我们可以快速绘制矩形，圆形等基本形状，并可实现基本的运动，但它同时存在一些缺点，优缺点的比较将在之后的段落讨论。

上面的例子是一个最基本的绘图结构。

~~~python
self.viewer = rendering.Viewer(SCREEN_W, SCREEN_H)
~~~

首先创建viewer类，屏幕的宽度和高度将作为类的初始值，之后所有的create elements和update location将通过这个类进行。

~~~python
l, r, t, b       = -ROBOT_W / 2, ROBOT_W / 2, ROBOT_H / 2, -ROBOT_H / 2
robot            = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
~~~

通过rendering函数已经提供的函数 FilledPolygon()，绘制机器人。另外要注意的是，机器人的重心坐标为原点（0，0），之后机器人的运动和位置，将会一直使用它。

~~~python
self.robot_trans = rendering.Transform()
robot.add_attr(self.robot_trans)
~~~

允许robot进行位移，下一个步骤的机器人的位移和位置的更新，将通过robot_trans实现。

~~~python
self.viewer.add_geom(robot)
~~~

将创建好的机器人加入viewer中，完成机器人画面的初始化。



update viewer 则需要下列代码，以机器人的x坐标为example：

~~~python
x = self.state[0]
self.robot_trans.set_translation(x)
return self.viewer.render(return_rgb_array=mode == 'rgb_array')
~~~

使用step函数更新完self.state后，首先提取出对应的机器人状态坐标，通过给self.robot_trans赋值，绘制最新的机器人位置。



## 对比 rendering module 和 pyglet

rendering module是gym提供的module，而pyglet则是更加底层的module，rendering需要调用pyglet。通过implement rendering 方法之后，我们找到了它的有点和缺点。

rendering module的好处是能够参考官方给的示例，结构清晰简单，只需要直接module 中的FilledPolygon()和Transform()即可绘制机器人，以及控制机器人的运动。同时，调用self.viewer.render就可以简单的刷新屏幕。

它的缺点也同时很明显，无法更改背景，以及插入图片等元素。为了能够事实显示机器人的位移，手臂的角度，以及reward，能够刷新的文字信息也需要被显示，单纯通过调用rendering module很难做到。

通过使用pyglet效果则更好，能够将环境中的元素用图片来表示，并且支持文字的刷新。缺点则是需要自己写一个复杂却十分完整的viewer类，通过清空画面和draw new frame 来实现显示功能。

作为前期和中期的快速搭建和测试，使用rendering函数就能够可视化大部分必要的信息，但如果作为最终结果，还需要将rendering更换成pyglet类，并在窗口添加相关的文字信息。例如机器人的实时位置，已经获得的reward等。

## 绘制元素

上面的章节讨论了，使用相应的模块和代码来输出agent和环境更新画面，本节将介绍需要绘制的元素。

在屏幕上一共有两种不同类型的元素，静态型和动态型。

机器人的起点和终点以及运动的轨迹，属于静态类型，这就意味着，它们不会发生任何变化并且能够一直显示。

其它元素属于静态型。为了绘制这些元素，首先我们需要找到它们相对自身的anchor，从而能够确定每一个element的运动。机器人作为底座，anchor在其几何中心。两个机械臂的anchor一样，坐标如下图所示。确定好了anchor的位置以后，就能够将它们放置在屏幕坐标系中，计算运动的公式。

![](D:\OneDrive\文档\TUM\2020SS\Applied Reinforce learning\draw elements.jpg)

机器人和手臂的位置坐标使用如下公式表示：

这里是公式