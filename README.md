# ML-for-AGV-dispatching-module
The simulation module for AGV dispatching with ML approach (SVM and DQN). 

This module is for AGV dispatching simulation with ML approach, SVM and DQN are included.
Edit run.py and change the string variable "Task" for differet purpose (see the describe bellow).
-Task = "TestD": Run AGV dispatching with single dispatching rules. (Default)
-Task = "CollectV": Create SVM training samples for Vehicle Initiated dispatching rule.
-Task = "CollectW": Create SVM training samples for Workstation Initiated dispatching rule.
-Task = "TestSVM": Train SVM dispatching model with training samples created and test the result.
-Task = "TrainDQN": Train the DQN dispatching network with the simulation environment and export the parameters of trained network.
-Task = "TestDQN": Test result of DQN dispatching model with the trained parameters file.

The excecute "run.py" to run the simulation of ML AGV dispatching.

## AGV dispatch MDP problem
![image](picture or gif url)

## State Setting

## Reward Function

## Research Abstract
The path planning problem and the dispatching problem about AGVs system will be discussed in the research. In AGVs path planning problem, a path planning method “A Star with Future Congestion” is proposed by considering the congestion cost of the nodes to be planned. This research also proposed a dead-lock resolution algorithm to deal with the conflict and dead-lock problem in the operation of multi AGVs system.
In the AGVs dispatching issue, an AGV dynamic dispatching in Flexible Manufacturing System (FMS) by machine learning technique is presented in this paper. The objective is to minimize mean tardiness of orders in FMS. The machine learning-based AGV dispatching approach - support vector machine (SVM) AGV dispatcher is proposed. The idea of the dispatcher is to make dispatching decision base on the system attributes. The simulation runs will be carrying out for generating the training data for SVM. The system attributes that might affect the performance of machine learning dispatcher will also be discussed.
Finally, another machine learning-based approach of AGV dispatching is proposed. The Deep Q Network (DQN) dispatcher, which is able to dynamically adjust the dispatching policy depend on the reward function, will be discussed. The definition of states, actions, and rewards for AGV dispatching problem are main issues for this paper.
In the AGVs dispatching experientment, SVM dispatcher has better effiency when the manufacturing situation is closed to the simulation model for training datas. The experiement result also shows that DQN dispatcher has better adaptivity to environment when the system status changes dramatically from time to time.
