# group-05

The project code of course "Applied Reinforcement Learning" in Technical University of Munich Summer Semester 2020.

## Team members
- B.Sc. Yikai Kang (03728450)
- B.Sc. Ming Gui (03687866)
- B.Sc. Bowen Ma (03721259)

## Goal of Project

In this Project, Reinforcement Learning is implemented to solve the task that robot base and robotic arms can get all scores in movement.

## How to use
1. Open main.py
2. Run.

## Introduction of files and folders
### Folders
- /maps: Path of robot is saved in maps.
- /resources: Some image items which are read by Viewer class.
### Files
- env.py: Original py file. All classes are saved in it. If you see more details of program, you can explore env.py.
- main.py: Clean py file to quick testing. It will import classes and functions from env.py, therefore this file is very clean and you can test your algorithm directly.
- reward_calculate.py: We provided different reward modes to test the performance of RL algorithm. You can switch them in env.py => REWARD_MODE.

## Introduction of States
You may see 5 items in state list variable.
For example, you can see the following numbers: 
[570.0, 200.00000000000165, 45, 45, (1, 1, 1)]
1. x position of robot base
2. y position 
3. Angle of arm1
4. Angle of arm2
5. Status of scores: 1 means it still exists, 0 means it have been collected. 

## Introduction of Actions
Three different types of actions. The action variable is also a list.
It looks like:
[1,0,1]
1. Robot base direction:  1 or 0 or -1
2. Arm1 direction: 1 or 0 or -1
3. Arm2 direction: 1 or 0 or -1
1 means move(rotate) forwards, 0 means don't move and -1 means move(rotate) backwards.

