# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:51:17 2018

@author: CIMlab徐孟維
"""
import simpy
from ShopFloor import Center
import Learn as Le
import numpy as np
import pandas as pd
import pickle
import time

Task = "None" #TestD, TestQ, TestDQN, TestSVM, CollectV, CollectW, TrainQ, TrainDQN
Task = "TestD"
'''
np.random.seed(124)
env = simpy. Environment()
Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=2, AGV_disRuleW=2, Ledispatch = "None")
env.run(until = 10000)
'''
if Task == "TestD":
    mt = []
    th = []
    method = []

    P = []
    startT = time.clock()
    for v in range(6):
        for w in range(3):
            m = []
            t = []
            for i in range(30):
                np.random.seed(124+i)
                env = simpy. Environment()
                Controller = Center(env, x=16, y=16, routRule=1, AGV_num=6, WS_num=12, AGV_disRuleV=v, AGV_disRuleW=w, Ledispatch = "None")
                env.run(until = 10000)
                m.append(Controller.MeanTardiness)
                t.append(Controller.Throughput)
            print("V: {}, W: {}, Throughput: {}".format(v, w, np.mean(t)))
            print("V: {}, W: {}, MeanTardiness: {}".format(v, w, np.mean(m)))
            th.append(np.mean(t))
            mt.append(np.mean(m))
            method.append([v, w])
            P.append(m)
    t0 = pd.DataFrame(method)
    t0.columns = ["VehicleRule", "WorkstationRule"]
    t1 = pd.DataFrame(th)
    t1.columns = ["Throughput"]
    t2 = pd.DataFrame(mt)
    t2.columns = ["MeanTardiness"]
    R = pd.concat([t0, t1, t2], axis = 1)
    R.to_csv("Result/R.csv")
    Per = pd.DataFrame(P)
    Per.to_csv("Result/performance.csv")
    print("CPUtime:", time.clock()-startT)
    

elif Task == "TrainQ":
    le = []
    le.append(Le.QLearning([7, 7, 7, 7], 6, Alpha = 0.1, Gamma = 0.9, incre = 0.95))
    le.append(Le.QLearning([7, 7, 7, 7], 3, Alpha = 0.1, Gamma = 0.9, incre = 0.95))
    
    expt = []
    expm = []
    expaV = []
    expaW = []
    
    for i in range(500):
        np.random.seed(124)
        env = simpy. Environment()
        Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=6, AGV_disRuleW=3, Ledispatch = le)
        env.run(until = 10001)
        
        if le[0].Epsilon + le[0].incre <= 0.95:
            le[0].Epsilon += le[0].incre
        if le[1].Epsilon + le[1].incre <= 0.95:
            le[1].Epsilon += le[1].incre
            
        expaV.append(list(le[0].Actions))
        for j in range(6):
            le[0].Actions[j] = 0
        expaW.append(list(le[1].Actions))
        for j in range(3):
            le[1].Actions[j] = 0
        expt.append(Controller.Throughput)
        expm.append(Controller.MeanTardiness)
        
        #print("Throughput: {}".format(Controller.Throughput))
        print("{}.MeanTardiness: {}".format(i, Controller.MeanTardiness))
    
    t1 = pd.DataFrame(expaV)
    t1.columns = ["STT", "MOQS", "MFCFS", "DS", "EDD", "DS"]
    t2 = pd.DataFrame(expaW)
    t2.columns = ["NV", "LIV", "LU"]
    t3 = pd.DataFrame(expt)
    t3.columns = ["Throughput"]
    t4 = pd.DataFrame(expm)
    t4.columns = ["MeanTardiness"]
    LER = pd.concat([t1, t2, t3, t4], axis = 1)
    
    with open('VID_Qtable', 'wb') as fp:
        pickle.dump(le[0].Qtable, fp)
    with open('WID_Qtable', 'wb') as fp:
        pickle.dump(le[1].Qtable, fp)
    
    LER.to_csv("Result/LER.csv")

elif Task == "TestQ":
    le = []
    le.append(Le.QLearning([7, 7, 7, 7], 6, Alpha = 0.1, Gamma = 0.9, incre = 0.95))
    le.append(Le.QLearning([7, 7, 7, 7], 3, Alpha = 0.1, Gamma = 0.9, incre = 0.95))
    
    with open('VID_Qtable', 'rb') as fp:
        le[0].Qtable = pickle.loads(fp)
    with open('WID_Qtable', 'rb') as fp:
        le[1].Qtable = pickle.loads(fp)

    le[0].Epsilon = 1
    le[1].Epsilon = 1
    performance = []
    for i in range(30):
        np.random.seed(124+i)
        env = simpy. Environment()
        Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=6, AGV_disRuleW=3, Ledispatch = le)
        env.run(until = 10000)
        performance.append(Controller.MeanTardiness)
        print("{}. MeanTardiness: {}".format(i, Controller.MeanTardiness))
        
    P = pd.DataFrame(performance)
    P.columns = ["Mean Tardiness"]
    P.to_csv("Result/Performance.csv")
    
elif Task == "TuneDQN":
    D = [7, 0]
    
    PerA = []
    A_mean = []
    for A in [0.01, 0.03, 0.05, 0.07, 0.09]:
        le = []
        le.append(Le.DQN(n_actions = 6, n_features = 22, LR = A, R_disc = 0.9, greedy = 0.9\
                         , greedy_incre = 0.9, replace_iter = 25\
                         , memory_size = 3000, batch_size = 300, Type = "V"))
        le.append(Le.DQN(n_actions = 3, n_features = 22, LR = A, R_disc = 0.9, greedy = 0.9\
                         , greedy_incre = 0.9, replace_iter = 25\
                         , memory_size = 300, batch_size = 30, Type = "W"))
        
        for i in range(300):
            
            np.random.seed(124)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=7, AGV_disRuleW=0, Ledispatch = le)
            env.run(until = 10001)
        
            le[0].Learning()
            
        le[0].epsilon = 1
        performance = []
        for i in range(30):
            np.random.seed(124+i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=7, AGV_disRuleW=0, Ledispatch = le)
            env.run(until = 10000)
            performance.append(Controller.MeanTardiness) 
            
        PerA.append(performance)
        A_mean.append(np.mean(performance))
        print("A = ", A, ": ", np.mean(performance))
        
    bestA = [0.01, 0.03, 0.05, 0.07, 0.09][np.argmin(A_mean)]
    P = pd.DataFrame(PerA)
    P.index = [0.01, 0.03, 0.05, 0.07, 0.09]
    P.to_csv("Result/LearningRate.csv")
    
    PerG = []
    G_mean = []
    for G in [0.1, 0.3, 0.5, 0.7, 0.9]:
        le = []
        le.append(Le.DQN(n_actions = 6, n_features = 22, LR = bestA, R_disc = G, greedy = 0.9\
                         , greedy_incre = 0.9, replace_iter = 25\
                         , memory_size = 3000, batch_size = 300, Type = "V"))
        le.append(Le.DQN(n_actions = 3, n_features = 22, LR = bestA, R_disc = G, greedy = 0.9\
                         , greedy_incre = 0.9, replace_iter = 25\
                         , memory_size = 300, batch_size = 30, Type = "W"))
        
        for i in range(300):
            np.random.seed(124)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=7, AGV_disRuleW=0, Ledispatch = le)
            env.run(until = 10001)
        
            le[0].Learning()
            
        le[0].epsilon = 1
        performance = []
        for i in range(30):
            np.random.seed(124+i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=7, AGV_disRuleW=0, Ledispatch = le)
            env.run(until = 10000)
            performance.append(Controller.MeanTardiness) 
            
        PerG.append(performance)
        G_mean.append(np.mean(performance))
        print("G = ", G, ": ", np.mean(performance))
        
    bestG = [0.1, 0.3, 0.5, 0.7, 0.9][np.argmin(G_mean)]
    P = pd.DataFrame(PerG)
    P.index = [0.1, 0.3, 0.5, 0.7, 0.9]
    P.to_csv("Result/Gamma.csv")
    
    PerT = []
    T_mean = []
    for T in [500, 1000, 2000]:
        le = []
        le.append(Le.DQN(n_actions = 6, n_features = 22, LR = bestA, R_disc = bestG, greedy = 0.9\
                         , greedy_incre = 0.9, replace_iter = 25\
                         , memory_size = 3000, batch_size = 300, Type = "V"))
        le.append(Le.DQN(n_actions = 3, n_features = 22, LR = bestA, R_disc = bestG, greedy = 0.9\
                         , greedy_incre = 0.9, replace_iter = 25\
                         , memory_size = 300, batch_size = 30, Type = "W"))
        
        for i in range(300):
            np.random.seed(124)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=7, AGV_disRuleW=0, Ledispatch = le)
            Controller.Period = T
            Controller.Warmup = 2000
            env.run(until = 10001)
        
            le[0].Learning()
            
        le[0].epsilon = 1
        performance = []
        for i in range(30):
            np.random.seed(124+i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=7, AGV_disRuleW=0, Ledispatch = le)
            Controller.Period = T
            env.run(until = 10000)
            performance.append(Controller.MeanTardiness) 
            
        PerT.append(performance)
        T_mean.append(np.mean(performance))
        print("T = ", T, ": ", np.mean(performance))
        
    bestT = [500, 1000, 2000][np.argmin(T_mean)]
    P = pd.DataFrame(PerT)
    P.index = [500, 1000, 2000]
    P.to_csv("Result/DeltaT.csv")
    
    PerE = []
    E_mean = []
    for E in [0, 0.01, 0.05, 0.1, 0.2]:
        le = []
        le.append(Le.DQN(n_actions = 6, n_features = 22, LR = bestA, R_disc = bestG, greedy = 1-E\
                         , greedy_incre = 1-E, replace_iter = 25\
                         , memory_size = 3000, batch_size = 300, Type = "V"))
        le.append(Le.DQN(n_actions = 3, n_features = 22, LR = bestA, R_disc = bestG, greedy = 1-E\
                         , greedy_incre = 1-E, replace_iter = 25\
                         , memory_size = 300, batch_size = 30, Type = "W"))
        
        for i in range(300):
            np.random.seed(124)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=7, AGV_disRuleW=0, Ledispatch = le)
            Controller.Period = bestT
            env.run(until = 10001)
        
            le[0].Learning()
            
        le[0].epsilon = 1
        performance = []
        for i in range(30):
            np.random.seed(124+i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=7, AGV_disRuleW=0, Ledispatch = le)
            Controller.Period = bestT
            env.run(until = 10000)
            performance.append(Controller.MeanTardiness) 
            
        PerE.append(performance)
        E_mean.append(np.mean(performance))
        print("E = ", E, ": ", np.mean(performance))
        
    bestE = [0, 0.01, 0.05, 0.1, 0.2][np.argmin(E_mean)]
    P = pd.DataFrame(PerE)
    P.index = [0, 0.01, 0.05, 0.1, 0.2]
    P.to_csv("Result/Epsilon.csv")
            
            
        
        

elif Task == "TrainDQN":
    le = []
    le.append(Le.DQN(n_actions = 6, n_features = 22, LR = 0.01, R_disc = 0.9, greedy = 0.95\
                     , greedy_incre = 0.95, replace_iter = 25\
                     , memory_size = 3000, batch_size = 300, Type = "V"))
    le.append(Le.DQN(n_actions = 3, n_features = 22, LR = 0.01, R_disc = 0.9, greedy = 0.95\
                     , greedy_incre = 0.95, replace_iter = 25\
                     , memory_size = 300, batch_size = 30, Type = "W"))
    
    expt = []
    expm = []
    expaV = []
    expaW = []
    
    for i in range(300):
        np.random.seed(123)
        env = simpy. Environment()
        Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=0, AGV_disRuleW=4, Ledispatch = le)
        env.run(until = 10001)
        
        le[1].Learning()
        print("{}.MeanTardiness: {}".format(i, Controller.MeanTardiness))
        le[1].performance_his.append(Controller.MeanTardiness)
        
        np.random.seed(123)
        env = simpy. Environment()
        Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=7, AGV_disRuleW=0, Ledispatch = le)
        env.run(until = 10001)
        
            
        expaV.append(list(le[0].Actions))
        for j in range(6):
            le[0].Actions[j] = 0
        expaW.append(list(le[1].Actions))
        for j in range(3):
            le[1].Actions[j] = 0
        expt.append(Controller.Throughput)
        expm.append(Controller.MeanTardiness)
    
        le[0].Learning()
     
        #print("Throughput: {}".format(Controller.Throughput))
        print("{}.MeanTardiness: {}".format(i, Controller.MeanTardiness))
        le[0].performance_his.append(Controller.MeanTardiness)
    
    le[0].Save_net()
    le[0].plot_performance()
    
    t1 = pd.DataFrame(expaV)
    t1.columns = ["STT", "MOQS", "MFCFS", "DS", "EDD", "DS"]
    #t2 = pd.DataFrame(expaW)
    #t2.columns = ["NV", "LIV", "LU"]
    t3 = pd.DataFrame(expt)
    t3.columns = ["Throughput"]
    t4 = pd.DataFrame(expm)
    t4.columns = ["MeanTardiness"]
    LER = pd.concat([t1, t3, t4], axis = 1)

elif Task == "TestDQN":
    le = []
    le.append(Le.DQN(n_actions = 6, n_features = 22, LR = 0.01, R_disc = 0.9, greedy = 0.95\
                     , greedy_incre = 0.005, replace_iter = 25\
                     , memory_size = 3000, batch_size = 300, Type = "V"))
    le.append(Le.DQN(n_actions = 3, n_features = 22, LR = 0.01, R_disc = 0.9, greedy = 0.95\
                     , greedy_incre = 0.005, replace_iter = 25\
                     , memory_size = 300, batch_size = 30, Type = "W"))
    
    le[0].Load_net()
    le[1].Load_net()
    
    le[0].epsilon = 1
    le[1].epsilon = 1
    performance = []
    for i in range(10):
        np.random.seed(124+i)
        env = simpy. Environment()
        Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=7, AGV_disRuleW=0, Ledispatch = le)
        env.run(until = 10000)
        performance.append(Controller.MeanTardiness)
        print("{}. MeanTardiness: {}".format(i, Controller.MeanTardiness))
        
    P = pd.DataFrame(performance)
    P.columns = ["Mean Tardiness"]
    P.to_csv("Result/Performance.csv")
    
elif Task == "CollectV":
    le = []
    le.append(Le.SVM())
    le.append(Le.SVM())
    
    for i in range(500):
        print("Collecting Sample({})".format(i+1))
        for j in range(6):
            np.random.seed(124+i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=j, AGV_disRuleW=0, Ledispatch = le, Task = ["Collect", "None"])
            Controller.Period = 2000
            Controller.Warmup = 1000
            env.run(until = 3001)
        R = []
        s = le[0].rawData[0][0]
        for k in le[0].rawData:
            R.append(k[1])
        le[0].Create_sample(s, R)
        le[0].rawData = []
    
        for j in range(6):
            np.random.seed(1000-i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=j, AGV_disRuleW=0, Ledispatch = le, Task = ["Collect", "None"])
            Controller.Period = 2000
            Controller.Warmup = 3000
            env.run(until = 5001)
        R = []
        s = le[0].rawData[0][0]
        for k in le[0].rawData:
            R.append(k[1])
        le[0].Create_sample(s, R)
        le[0].rawData = []
        
        for j in range(6):
            np.random.seed(2*i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=j, AGV_disRuleW=0, Ledispatch = le, Task = ["Collect", "None"])
            Controller.Period = 2000
            Controller.Warmup = 5000
            env.run(until = 7001)
        R = []
        s = le[0].rawData[0][0]
        for k in le[0].rawData:
            R.append(k[1])
        le[0].Create_sample(s, R)
        le[0].rawData = []
        
        for j in range(6):
            np.random.seed(5*i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=j, AGV_disRuleW=0, Ledispatch = le, Task = ["Collect", "None"])
            Controller.Period = 2000
            Controller.Warmup = 7000
            env.run(until = 9001)
        R = []
        s = le[0].rawData[0][0]
        for k in le[0].rawData:
            R.append(k[1])
        le[0].Create_sample(s, R)
        le[0].rawData = []
    
    le[0].Save_samples()

elif Task == "CollectW":
    le = []
    le.append(Le.SVM())
    le.append(Le.SVM())
    
    for i in range(500):
        print("Collecting Sample({})".format(i+1))
        for j in range(3):
            np.random.seed(124+i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=0, AGV_disRuleW=j, Ledispatch = le, Task = ["None", "Collect"])
            Controller.Period = 2000
            Controller.Warmup = 1000
            env.run(until = 3001)
        R = []
        s = le[1].rawData[0][0]
        for k in le[1].rawData:
            R.append(k[1])
        le[1].Create_sample(s, R)
        le[1].rawData = []
    
        for j in range(3):
            np.random.seed(1000-i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=0, AGV_disRuleW=j, Ledispatch = le, Task = ["None", "Collect"])
            Controller.Period = 2000
            Controller.Warmup = 3000
            env.run(until = 5001)
        R = []
        s = le[1].rawData[0][0]
        for k in le[1].rawData:
            R.append(k[1])
        le[1].Create_sample(s, R)
        le[1].rawData = []
        
        for j in range(3):
            np.random.seed(2*i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=0, AGV_disRuleW=j, Ledispatch = le, Task = ["None", "Collect"])
            Controller.Period = 2000
            Controller.Warmup = 5000
            env.run(until = 7001)
        R = []
        s = le[1].rawData[0][0]
        for k in le[1].rawData:
            R.append(k[1])
        le[1].Create_sample(s, R)
        le[1].rawData = []
        
        for j in range(3):
            np.random.seed(5*i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=0, AGV_disRuleW=j, Ledispatch = le, Task = ["None", "Collect"])
            Controller.Period = 2000
            Controller.Warmup = 7000
            env.run(until = 9001)
        R = []
        s = le[1].rawData[0][0]
        for k in le[1].rawData:
            R.append(k[1])
        le[1].Create_sample(s, R)
        le[1].rawData = []
    
    le[1].Save_samples()

elif Task == "TuneSVM":
    D500 = pd.read_csv("training_setD500.csv").iloc[: ,1:]
    D1000 = pd.read_csv("training_setD1000.csv").iloc[: ,1:]
    D2000 = pd.read_csv("training_setD2000.csv").iloc[: ,1:]
    '''
    PerC = []
    C_mean = []
    for C in range(-5, 6):
        le = []
        le.append(Le.SVM("rbf", 2**C, 1, 10))
        le.append(Le.SVM("rbf", 2**C, 1, 10))
        
        le[0].Load_samples(D2000)
        
        le[0].Learn_all()
            
        performance = []
        for i in range(30):
            np.random.seed(124+i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=8, AGV_disRuleW=0, Ledispatch = le)
            Controller.Period = 500
            env.run(until = 10000)
            performance.append(Controller.MeanTardiness) 
            
        PerC.append(performance)
        C_mean.append(np.mean(performance))
        print("C = ", C, ": ", np.mean(performance))
        
    bestC = range(-5, 6)[np.argmin(C_mean)]
    P = pd.DataFrame(PerC)
    P.index = range(-5, 6)
    P.to_csv("Result/C.csv")
    
    PerR = []
    R_mean = []
    for R in range(-5, 6):
        le = []
        le.append(Le.SVM("rbf", 2**bestC, 2**R, 10))
        le.append(Le.SVM("rbf", 2**bestC, 2**R, 10))
        
        le[0].Load_samples(D2000)
        
        le[0].Learn_all()
            
        performance = []
        for i in range(30):
            np.random.seed(124+i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=8, AGV_disRuleW=0, Ledispatch = le)
            Controller.Period = 500
            env.run(until = 10000)
            performance.append(Controller.MeanTardiness) 
            
        PerR.append(performance)
        R_mean.append(np.mean(performance))
        print("R = ", R, ": ", np.mean(performance))
        
    bestR = range(-5, 6)[np.argmin(R_mean)]
    P = pd.DataFrame(PerR)
    P.index = range(-5, 6)
    P.to_csv("Result/R.csv")
    '''
    PerT = []
    T_mean = []
    for T in [D1000, D2000]:
        le = []
        le.append(Le.SVM("rbf", 2**5, 2**0, 10))
        le.append(Le.SVM("rbf", 2**1, 2**0, 10))
        
        
        le[0].Load_samples(T)
        
        le[0].Learn_all()
            
        performance = []
        for i in range(30):
            np.random.seed(124+i)
            env = simpy. Environment()
            Controller = Center(env, x=16, y=16, routRule=1, AGV_num=7, WS_num=12, AGV_disRuleV=8, AGV_disRuleW=0, Ledispatch = le)
            Controller.Period = 500
            env.run(until = 10000)
            performance.append(Controller.MeanTardiness) 
            
        PerT.append(performance)
        T_mean.append(np.mean(performance))
        print("T = : ", np.mean(performance))
        
    bestT = [1000, 2000][np.argmin(T_mean)]
    P = pd.DataFrame(PerT)
    P.index = [1000, 2000]
    P.to_csv("Result/T.csv")
    
    
 
elif Task == "TestSVM":
    le = []
    le.append(Le.SVM("rbf", 2**2, 2**-2, 10))
    le.append(Le.SVM("rbf", 2**0, 2**2, 10))
    
    data1 = pd.read_csv("training_setV1.csv").iloc[: ,1:]
    #data2 = pd.read_csv("training_set2.csv").iloc[: ,1:]
    #data3 = pd.read_csv("training_set3.csv").iloc[: ,1:]
    #data4 = pd.read_csv("training_set4.csv").iloc[: ,1:]
    dataA = pd.read_csv("training_setW1.csv").iloc[: ,1:]
    #dataB = pd.read_csv("training_setB.csv").iloc[: ,1:]
    #dataC = pd.read_csv("training_setC.csv").iloc[: ,1:]
    #dataD = pd.read_csv("training_setD.csv").iloc[: ,1:]
    
    le[0].Load_samples(data1)
    #le[0].Load_samples(data2)
    #le[0].Load_samples(data3)
    #le[0].Load_samples(data4)
    le[1].Load_samples(dataA)
    #le[1].Load_samples(dataB)
    #le[1].Load_samples(dataC)
    #le[1].Load_samples(dataD)
    
    le[0].Learn_all()
    le[1].Learn_all()
    
    
    performance = []
    for i in range(30):
        np.random.seed(124+i)
        env = simpy. Environment()
        Controller = Center(env, x=16, y=16, routRule=1, AGV_num=1, WS_num=12, AGV_disRuleV=8, AGV_disRuleW=5, Ledispatch = le)
        Controller.Period = 500
        Controller.Warmup = 2000
        env.run(until = 10000)
        performance.append(Controller.MeanTardiness)
        print("TH: {}".format(Controller.Throughput))
        print("{}. Tardiness: {}".format(i, Controller.MeanTardiness))
        
    P = pd.DataFrame(performance)
    P.columns = ["Mean Tardiness"]
    P.to_csv("Result/VW.csv")

