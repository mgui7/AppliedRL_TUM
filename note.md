GUI
structure

How to discrete


RL部分实现

介绍三个reward
1. 2 规定好舒徐的
2. 3 没有规定顺序
3. sparse 没有规定顺序

最后报告分成两部分：
1. 规定顺序，使用reward2
2. 不规定顺序，使用reward3 和 sparse


比较reward：
固定action，比较reward

比较action：
固定reward，比较action


什么叫做收敛?
cycle的情况


- 对比训练时间（收敛速度）

sparse 或 reward3
选择前后 和 单纯向前，对比收敛速度。


收敛：
开始，随意的系统
迭代
获得一个比较稳定的算法



完成步数/未完成步数：
收集结束状态，如果没有规定时间结束-》收集多少点
如果结束了，那么记录结束步数。


评价其他组





reward2 直观, action 27 个