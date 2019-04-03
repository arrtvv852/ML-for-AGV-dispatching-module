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
![image](https://github.com/arrtvv852/ML-for-AGV-dispatching-module/blob/master/DQNfigure.PNG)

## State Setting
![image](https://github.com/arrtvv852/ML-for-AGV-dispatching-module/blob/master/State.PNG)
## Reward Function
![image](https://github.com/arrtvv852/ML-for-AGV-dispatching-module/blob/master/Reward.PNG)
## Illustration
![image](https://github.com/arrtvv852/ML-for-AGV-dispatching-module/blob/master/Illustration.PNG)
## Published Research
https://github.com/arrtvv852/ML-for-AGV-dispatching-module/blob/master/A%20Support%20Vector%20Machine%20Approach%20for%20AGV%20Dispatching.pdf
