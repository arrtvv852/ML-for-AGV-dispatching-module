"""
Created on Sat Nov 18 23:29:07 2017

@author: CIMlab徐孟維
"""
import simpy
from Routing import ShortestPath as sp
import tkinter as tk
import Dispatcher as dp
import numpy as np
import Learn as Le
import copy
import time

#unit = 50
#Hight = 7
#Width = 7
#Origin = [1, 1]


class Job(object):
    
    current_seq = 0
    

    def __init__(self, env, Controller, Jtype):
        self.env = env
        self.Controller = Controller
        self.Type = Jtype
        self.FlowTime = self.env.now
        self.Tardiness = 0
        self.create(Jtype)
        #
        #
        self.position = Controller.stations[0]
        
        self.Queue = self.env.event()
        
        self.Controller.jobs.append(self)
        self.Controller.job_num += 1
        self.Controller.Rtravel += (len(self.seq)-1)
        self.Controller.sysJob.append(self)
        self.position.out_num += 1
        self.position.out_queue.append(self)
        
        self.env.process(self.flow())

    def create(self, JT):
        if JT == 1:
            #self.DueDate = 1062 + self.env.now
            self.DueDate = 1416*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 3, 4, 6, 5, 1, 8, 9, 10, 2, 7, 11]
            self.PT = [0, 44, 5, 58, 97, 9, 84, 77, 96, 58, 89, 0]
            self.PT = (np.array([0, 44, 5, 58, 97, 9, 84, 77, 96, 58, 89, 0])/2).tolist()
        elif JT == 2:
            #self.DueDate = 1064 + self.env.now
            self.DueDate = 1418*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 5, 8, 2, 9, 1, 4, 3, 6, 10, 7, 11]
            self.PT = [0, 15, 31, 87, 57, 77, 85, 81, 39, 73, 21, 0]
            self.PT = (np.array([0, 15, 31, 87, 57, 77, 85, 81, 39, 73, 21, 0])/2).tolist()
        elif JT == 3:
            #self.DueDate = 947 + self.env.now
            self.DueDate = 1262*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 10, 7, 6, 4, 2, 1, 9, 3, 8, 6, 11]
            self.PT = [0, 82, 22, 10, 70, 49, 40, 34, 48, 80, 71, 0]
            self.PT = (np.array([0, 82, 22, 10, 70, 49, 40, 34, 48, 80, 71, 0])/2).tolist()
        elif JT == 4:
            #self.DueDate = 869 + self.env.now
            self.DueDate = 1158*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 2, 3, 8, 6, 9, 5, 4, 7, 10, 1, 11]
            self.PT = [0, 91, 17, 62, 75, 47, 11, 7, 72, 35, 55, 0]
            self.PT = (np.array([0, 91, 17, 62, 75, 47, 11, 7, 72, 35, 55, 0])/2).tolist()
        elif JT == 5:
            #self.DueDate = 1058 + self.env.now
            self.DueDate = 1410*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 7, 2, 4, 1, 3, 9, 5, 8, 10, 6, 11]
            self.PT = [0, 71, 90, 75, 64, 94, 15, 12, 67, 20, 50, 0]
            self.PT = (np.array([0, 71, 90, 75, 64, 94, 15, 12, 67, 20, 50, 0])/2).tolist()
        elif JT == 6:
            #self.DueDate = 1097 + self.env.now
            self.DueDate = 1462*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 8, 6, 9, 3, 5, 7, 4, 2, 10, 1, 11]
            self.PT = [0, 70, 93, 77, 29, 58, 93, 68, 57, 7, 52, 0]
            self.PT = (np.array([0, 70, 93, 77, 29, 58, 93, 68, 57, 7, 52, 0])/2).tolist()
        elif JT == 7:
            #self.DueDate = 965 + self.env.now
            self.DueDate = 1286*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 7, 2, 5, 6, 3, 4, 8, 9, 10, 1, 11]
            self.PT = [0, 87, 63, 26, 6, 82, 27, 56, 48, 36, 95, 0]
            self.PT = (np.array([0, 87, 63, 26, 6, 82, 27, 56, 48, 36, 95, 0])/2).tolist()
        elif JT == 8:
            #self.DueDate = 887 + self.env.now
            self.DueDate = 1182*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 1, 6, 9, 10, 4, 7, 5, 8, 3, 2, 11]
            self.PT = [0, 36, 15, 41, 78, 76, 84, 30, 76, 36, 8, 0]
            self.PT = (np.array([0, 36, 15, 41, 78, 76, 84, 30, 76, 36, 8, 0])/2).tolist()
        elif JT == 9:
            #self.DueDate = 1011 + self.env.now
            self.DueDate = 1348*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 6, 3, 4, 7, 5, 8, 9, 10, 2, 1, 0]
            self.PT = [0, 88, 81, 13, 82, 54, 13, 29, 40, 78, 75, 0]
            self.PT = (np.array([0, 88, 81, 13, 82, 54, 13, 29, 40, 78, 75, 0])/2).tolist()
        elif JT == 10:
            #self.DueDate = 849 + self.env.now
            self.DueDate = 1132*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 10, 5, 7, 8, 1, 3, 9, 6, 4, 2, 11]
            self.PT = [0, 88, 54, 64, 32, 52, 6, 54, 82, 6, 26, 0]
            self.PT = (np.array([0, 88, 54, 64, 32, 52, 6, 54, 82, 6, 26, 0])/2).tolist()
    def create2(self, JT):
        if JT == 1:
            #self.DueDate = 536 + self.env.now
            #self.DueDate = 402 + self.env.now
            #self.DueDate = 268 + self.env.now
            self.DueDate = 536*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 2, 9, 1, 3, 6, 7, 4, 8, 10, 5, 11]
            self.PT = [0, 13, 16, 17, 7, 14, 19, 13, 9, 12, 20, 0]
        elif JT == 2:
            #self.DueDate = 452 + self.env.now
            #self.DueDate = 339 + self.env.now
            #self.DueDate = 226 + self.env.now
            self.DueDate = 452*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 7, 5, 10, 9, 8, 4, 6, 2, 1, 3, 11]
            self.PT = [0, 12, 19, 5, 6, 15, 15, 12, 8, 6, 12, 0]
        elif JT == 3:
            #self.DueDate = 464 + self.env.now
            #self.DueDate = 348 + self.env.now
            #self.DueDate = 232 + self.env.now
            #self.DueDate = 947 + self.env.now
            self.DueDate = 464*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 8, 7, 6, 9, 4, 1, 3, 5, 10, 2, 11]
            self.PT = [0, 8, 8, 10, 20, 12, 10, 15, 15, 5, 17, 0]
        elif JT == 4:
            #self.DueDate = 500 + self.env.now
            #self.DueDate = 375 + self.env.now
            #self.DueDate = 250 + self.env.now
            #self.DueDate = 869 + self.env.now
            self.DueDate = 500*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 3, 10, 4, 2, 8, 6, 5, 9, 7, 1, 11]
            self.PT = [0, 8, 20, 14, 19, 17, 18, 9, 20, 6, 8, 0]
        elif JT == 5:
            #self.DueDate = 472 + self.env.now
            #self.DueDate = 354 + self.env.now
            #self.DueDate = 236 + self.env.now
            #self.DueDate = 1058 + self.env.now
            self.DueDate = 472*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 5, 1, 3, 8, 9, 7, 2, 10, 4, 6, 11]
            self.PT = [0, 20, 13, 12, 7, 9, 10, 8, 16, 6, 16, 0]
        elif JT == 6:
            #self.DueDate = 558 + self.env.now
            #self.DueDate = 419 + self.env.now
            #self.DueDate = 279 + self.env.now
            #self.DueDate = 1097 + self.env.now
            self.DueDate = 558*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 8, 3, 10, 4, 9, 2, 1, 6, 7, 5, 11]
            self.PT = [0, 6, 14, 19, 14, 20, 12, 16, 17, 14, 6, 0]
        elif JT == 7:
            #self.DueDate = 460 + self.env.now
            #self.DueDate = 345 + self.env.now
            #self.DueDate = 230 + self.env.now
            #self.DueDate = 965 + self.env.now
            self.DueDate = 460*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 2, 3, 8, 10, 7, 1, 5, 6, 4, 9, 11]
            self.PT = [0, 16, 6, 7, 17, 13, 17, 17, 17, 14, 15, 0]
        elif JT == 8:
            #self.DueDate = 504 + self.env.now
            #self.DueDate = 378 + self.env.now
            #self.DueDate = 252 + self.env.now
            #self.DueDate = 887 + self.env.now
            self.DueDate = 504*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 4, 6, 3, 1, 10, 5, 7, 2, 8, 9, 11]
            self.PT = [0, 6, 13, 14, 9, 13, 13, 19, 15, 18, 16, 0]
        elif JT == 9:
            #self.DueDate = 486 + self.env.now
            #self.DueDate = 365 + self.env.now
            #self.DueDate = 243 + self.env.now
            #self.DueDate = 1011 + self.env.now
            self.DueDate = 486*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 7, 2, 3, 9, 10, 5, 6, 1, 4, 8, 11]
            self.PT = [0, 8, 13, 13, 7, 9, 10, 16, 11, 16, 18, 0]
        elif JT == 10:
            #self.DueDate = 488 + self.env.now
            #self.DueDate = 366 + self.env.now
            #self.DueDate = 244 + self.env.now
            #self.DueDate = 849 + self.env.now
            self.DueDate = 488*(0.5+np.random.random()) + self.env.now
            self.seq = [0, 3, 4, 1, 5, 10, 9, 2, 6, 8, 7, 11]
            self.PT = [0, 13, 18, 19, 6, 5, 19, 19, 6, 9, 18, 0]
            
    def flow(self):
        yield self.env.timeout(0)
        dp.WID(self.Controller, self, self.Controller.AGV_disRuleW, self.Controller.routRule, self.Controller.Parameter)
        self.out_req = self.Controller.stations[0].out_buffer.request()
        while self.current_seq < 11:
            yield self.Queue
            self.current_seq += 1
            self.Queue = self.env.event()
            yield self.env.timeout(0)
            station = self.Controller.stations[self.seq[self.current_seq]]
            self.position = station
            req = station.resource.request()
            
            station.in_queue.append(self)
            station.in_num += 1
            
            yield req
            station.processing = self
            pt = self.PT[self.current_seq]
            station.in_queue.remove(self)
            station.in_num -= 1
            yield self.env.timeout(pt)
            station.processing = 0
            
            self.out_req = station.out_buffer.request()
            yield self.out_req
            
            station.out_queue.append(self)
            station.out_num += 1
            station.resource.release(req)
            if self.current_seq != 11:
                self.Controller.jobs.append(self)
                self.Controller.job_num += 1
                dp.WID(self.Controller, self, self.Controller.AGV_disRuleW, self.Controller.routRule, self.Controller.Parameter)
                
        self.Tardiness = max(0, self.env.now - self.DueDate)
        self.Tardiness = round(float(self.Tardiness))
        self.FlowTime = self.env.now - self.FlowTime
            

            
        self.Controller.Tardiness.append(self.Tardiness)
        #self.Controller.MeanTardiness = np.mean(self.Controller.Tardiness)
        self.Controller.FlowTime.append(self.FlowTime)
        self.Controller.Throughput += 1
        self.Controller.sysJob.remove(self)
        
        self.Controller.status.tardiness.append(self.Tardiness)
        self.Controller.status.throughput += 1
        #self.Controller.status.meantardiness = np.mean(self.Controller.status.tardiness)
            
     

            
            
            
class Station(object):
    def __init__(self, env, Controller, id):
        self.env = env
        self.Controller = Controller
        self.ID = id
        self.in_queue = []
        self.in_num = 0
        self.out_queue = []
        self.out_num = 0
        self.out_cap = 5
        
        self.processing = []
        self.cum_PT = 0
        self.location(id)
        
        if self.ID != 0 and self.ID != 11:
            self.resource = simpy.Resource(self.env, 1)
            self.out_buffer = simpy.Resource(self.env, self.out_cap)
        else:
            self.resource = simpy.Resource(self.env, 1000)
            self.out_buffer = simpy.Resource(self.env, 1000)
            
        self.Controller.stations.append(self)
        
    def location(self, ID):
        if ID == 0:
            self.x = 0
            self.y = 2
        elif ID == 1:
            self.x = 9
            self.y = 8
        elif ID == 2:
            self.x = 5
            self.y = 2
        elif ID == 3:
            self.x = 8
            self.y = 2
        elif ID == 4:
            self.x = 13
            self.y = 4
        elif ID == 5:
            self.x = 13
            self.y = 7
        elif ID == 6:
            self.x = 13
            self.y = 10
        elif ID == 7:
            self.x = 3
            self.y = 13
        elif ID == 8:
            self.x = 6
            self.y = 13
        elif ID == 9:
            self.x = 9
            self.y = 13
        elif ID == 10:
            self.x = 4
            self.y = 8
        elif ID == 11:
            self.x = 0
            self.y = 1


class Vehicle(object):
    def __init__(self, env, Controller, mesh, id, x, y):
        self.ID = id
        self.x = x
        self.y = y
        
        
        self.path = 0
        self.status = []
        
        self.env = env
        self.Controller = Controller
        self.create_show()
        self.mesh = mesh
        self.redet = self.env.event()
        self.Redet = False
        
        self.env.process(self.nevigation())
        
        self.Waiting = self.env.event()
        self.wait = self.env.process(self.redetect())
        
        #
        self.content = 0
        self.nextpath = 0
        self.Goal = 0
        self.Parking = False
        
        self.Charging = False
        self.Electricity = np.random.uniform(0.5, 1)
        

        self.Controller.AGVs.append(self)
        self.Controller.IV.append(self)
        self.Controller.IV_num += 1
        
        #Stats
        self.startIdle = True
        self.IdleTime = 0
        self.Idling = False
        
        
        
    def nevigation(self):
        self.req_ori = self.mesh[self.x][self.y].request()
        yield self.req_ori
        #print("Vehicle ", self.ID, " now at ", [self.x, self.y], " (Time: ", self.env.now, ")")
        
        
        dp.VID(self.Controller, self, self.Controller.AGV_disRuleV, self.Controller.routRule, self.Controller.Parameter)
        while True:
            if self.status != []:
                if self.status[0] == 1:
                    nextpos = [self.x, self.y+1]
                elif self.status[0] == 2:
                    nextpos = [self.x, self.y-1]
                elif self.status[0] == 3:
                    nextpos = [self.x-1, self.y]
                elif self.status[0] == 4:
                    nextpos = [self.x+1, self.y]
                if self.detect(self.mesh[nextpos[0]][nextpos[1]].users):
                    yield self.env.timeout(0)
                    if self.status == []:
                        nextpos = [self.x, self.y]
                    elif self.status[0] == 1:
                        nextpos = [self.x, self.y+1]
                    elif self.status[0] == 2:
                        nextpos = [self.x, self.y-1]
                    elif self.status[0] == 3:
                        nextpos = [self.x-1, self.y]
                    elif self.status[0] == 4:
                        nextpos = [self.x+1, self.y]
                        



                        
                
                self.Redet = False
                
                
                self.req_tar = self.mesh[nextpos[0]][nextpos[1]].request()
                
                
                self.redet.succeed()
                
                #self.req_ori.append(req_tar)
                yield self.req_tar | self.Waiting
                self.Waiting = self.env.event()
                self.wait.interrupt()
                if self.Redet:
                #if False:
                    #self.Redet = False
                    
                    if self.req_tar in self.mesh[nextpos[0]][nextpos[1]].users:
                        self.mesh[nextpos[0]][nextpos[1]].release(self.req_tar)
                    if self.req_tar in self.mesh[nextpos[0]][nextpos[1]].queue:
                        self.mesh[nextpos[0]][nextpos[1]].queue.remove(self.req_tar)
                        #self.req_tar.pop(0)
                else:
                    self.Redet = False
                    
                    for i in range(self.Controller.split):
                    
                        yield self.env.timeout(1/self.Controller.split)
                        if self.status != []:
                            self.moving(self.status[0])
                            
                            
                    #yield self.env.timeout(1)
                    

                    self.mesh[self.x][self.y].release(self.req_ori)
                    

                    self.x = nextpos[0]
                    self.y = nextpos[1]
                    self.req_ori = self.req_tar
                    if self.status != []:
                        self.status.pop(0)
                    #print("Vehicle ", self.ID, " now at ", [self.x, self.y], " (Time: ", self.env.now, ")")
            
            #AGV load job
            elif self.nextpath != 0:
                yield self.env.timeout(1)
                self.content = self.Goal
                self.content.position.out_buffer.release(self.content.out_req)
                self.Goal = 0
                self.content.position.out_num -= 1
                self.content.position.out_queue.remove(self.content)
                self.content.position = self
                self.path = self.nextpath
                self.status = self.path.direct
                self.nextpath = 0
            
            #AGV Unload job
            elif self.content != 0:
                yield self.env.timeout(1)
                self.content.Queue.succeed()
                
                self.Controller.IV_num += 1
                self.Controller.IV.append(self)
                self.Controller.Rtravel -= 1
                self.content = 0
                if self.Electricity > 1/20:
                    dp.VID(self.Controller, self, self.Controller.AGV_disRuleV, self.Controller.routRule, self.Controller.Parameter)
                else:
                    self.Charging = True
                    self.Controller.IV_num -= 1
                    self.Controller.IV.remove(self)
                    self.path = sp([self.x, self.y], [self.Controller.Cstation.x\
                                        , self.Controller.Cstation.y]\
                                        , self.Controller.routRule, self.Controller.x-1\
                                        , self.Controller.y-1, self.Controller.BLOCK, self, self.Controller.Parameter)
                    self.status = self.path.direct
                    self.startIdle = self.Controller.Time
                    self.Idling = self.Controller.Time
                
            #Start Charging
            elif self.Charging:
                yield self.env.timeout(300)
                self.Electricity = 1
                self.Charging = False
                self.Controller.IV_num += 1
                self.Controller.IV.append(self)
            #Start Parking
            elif self.Parking:
                self.Parking = False
                self.Controller.IV_num += 1
                self.Controller.IV.append(self)
                self.content = 0
                #dp.VID(self.Controller, self, self.Controller.AGV_disRule, self.Controller.routRule, self.Controller.Parameter)
                self.startIdle = self.Controller.Time
                self.Idling = self.Controller.Time
            
            else:
                #dp.VID(self.Controller, self, self.Controller.AGV_disRuleV, self.Controller.routRule, self.Controller.Parameter)
                #yield self.env.timeout(1)
                if self.Electricity > 1/20:
                    yield self.env.timeout(1)
                else:
                    self.Charging = True
                    self.Controller.IV_num -= 1
                    self.Controller.IV.remove(self)
                    self.path = sp([self.x, self.y], [self.Controller.Cstation.x\
                                        , self.Controller.Cstation.y]\
                                        , self.Controller.routRule, self.Controller.x-1\
                                        , self.Controller.y-1, self.Controller.BLOCK, self, self.Controller.Parameter)
                    self.status = self.path.direct
                    self.startIdle = self.Controller.Time
                    self.Idling = self.Controller.Time
                
                
                

                
    def detect(self, req):
        sensor = False
        if req != []:
            for i in range(self.Controller.AGV_num):
                if self.Controller.AGVs[i].req_ori == req[0]:
                     V1 = self
                     V2 = self.Controller.AGVs[i]
                     self.deadlock_resolve(V1, V2)
                    
                '''
                for j in self.Controller.AGVs[i].req_ori:
                    if req[0] == j:
                        V1 = self
                        V2 = self.Controller.AGVs[i]
                        self.deadlock_resolve(V1, V2)
                '''
            sensor = True
            
        return sensor
    
        
    def redetect(self):
        while True:
            try:
                yield self.redet
                self.redet = self.env.event()
                yield self.env.timeout(2)
                self.Redet = True
                self.Waiting.succeed()
            except simpy.Interrupt:
                self.redet = self.env.event()
        
    def deadlock_resolve(self, V1, V2):
        toward = V1.status[0]
        if toward == 1:
            anti = 2
            if V1.x-1>=0 and [V1.x-1, V1.y] not in self.Controller.BLOCK:
                swap = 3
                swap2 = 4
            elif V1.x+1<V1.Controller.x and [V1.x+1, V1.y] not in self.Controller.BLOCK:
                swap = 4
                swap2 = 3
        elif toward == 2:
            anti = 1
            if V1.x+1<V1.Controller.x and [V1.x+1, V1.y] not in self.Controller.BLOCK:
                swap = 4
                swap2 = 3
            elif V1.x-1>=0 and [V1.x-1, V1.y] not in self.Controller.BLOCK:
                swap = 3
                swap2 = 4
        elif toward == 3:
            anti = 4
            if V1.y+1<V1.Controller.y and [V1.x, V1.y+1] not in self.Controller.BLOCK:
                swap = 1
                swap2 = 2
            elif V1.y-1>=0 and [V1.x,V1.y-1] not in self.Controller.BLOCK:
                swap = 2
                swap2 = 1
        elif toward == 4:
            anti = 3
            if V1.y-1>=0 and [V1.x,V1.y-1] not in self.Controller.BLOCK:
                swap = 2
                swap2 = 1
            elif V1.y+1<V1.Controller.y and [V1.x,V1.y+1] not in self.Controller.BLOCK:
                swap = 1
                swap2 = 2
            
        checkpoint = [[V1.x, V1.y+1], [V1.x, V1.y-1], [V1.x-1, V1.y], [V1.x+1, V1.y]]
        block = []
        for i in checkpoint:
            if i[0] < self.Controller.x and i[1] < self.Controller.y and i[0]>=0 and i[1]>=0:
                if self.mesh[i[0]][i[1]].users != []:
                    block.append(i)
                    
        
                    
        if V2.status != []:
            if V2.status[0] == anti or self.Redet:
                ori1 = sp([V1.x, V1.y], V1.path.path[len(V1.path.path)-1], self.Controller.routRule, self.Controller.x-1, self.Controller.y-1, []+self.Controller.BLOCK, V1, self.Controller.Parameter).distance
                ori2 = sp([V2.x, V2.y], V2.path.path[len(V2.path.path)-1], self.Controller.routRule, self.Controller.x-1, self.Controller.y-1, []+self.Controller.BLOCK, V2, self.Controller.Parameter).distance
                if V1.path.path[len(V1.path.path)-1] in block or len(block)>=3:
                    modi1 = 9999
                else:
                    modi1 = sp([V1.x, V1.y], V1.path.path[len(V1.path.path)-1], self.Controller.routRule, self.Controller.x-1, self.Controller.y-1, block+self.Controller.BLOCK, V1, self.Controller.Parameter).distance
                
                checkpoint = [[V2.x, V2.y+1], [V2.x, V2.y-1], [V2.x-1, V1.y], [V2.x+1, V2.y]]
                block2 = []
                for i in checkpoint:
                    if i[0] < self.Controller.x and i[1] < self.Controller.y and i[0] >= 0 and i[1] >= 0:
                        if self.mesh[i[0]][i[1]].users != []:
                            block2.append(i)
                
                if V2.path.path[len(V2.path.path)-1] in block2 or len(block2)>=3:
                    modi2 = 9999
                else:
                    modi2 = sp([V2.x, V2.y], V2.path.path[len(V2.path.path)-1], self.Controller.routRule, self.Controller.x-1, self.Controller.y-1, block2+self.Controller.BLOCK, V2, self.Controller.Parameter).distance
                
                
                
                diff1 = modi1-ori1
                diff2 = modi2-ori2
                
                
                
                if diff1 <= diff2 and V1.path.path[len(V1.path.path)-1] not in block and len(block)<4:
                #if True:
                    tmp = sp([V1.x, V1.y], V1.path.path[len(V1.path.path)-1], self.Controller.routRule, self.Controller.x-1, self.Controller.y-1, block+self.Controller.BLOCK, V1, self.Controller.Parameter)
                    if tmp.path !=[]:
                        V1.path = tmp
                        V1.status = V1.path.direct
                
                if self.Redet:
                    if V1.path.path != [] and V1.path.path[len(V1.path.path)-1] == [V2.x, V2.y]:
                    #if V1.path.path != [] and V1.path.path[len(V1.path.path)-1] == [V2.x, V2.y] and V2.path.path[len(V2.path.path)-1] == [V1.x, V1.y]:
                        V1.status.insert(0, swap2)
                        V1.status.insert(0, swap)
        else:
            if V1.path.path[len(V1.path.path)-1] not in block and len(block)<3:
                V1.path = sp([V1.x, V1.y], V1.path.path[len(V1.path.path)-1], self.Controller.routRule, self.Controller.x-1, self.Controller.y-1, block+self.Controller.BLOCK, V1, self.Controller.Parameter)
                V1.status = V1.path.direct
            elif V1.path.path[len(V1.path.path)-1] in block:
                V2.path = sp([V2.x, V2.y], [V2.Controller.Cstation.x\
                                        , V2.Controller.Cstation.y]\
                                        , V2.Controller.routRule, V2.Controller.x-1, V2.Controller.y-1, V2.Controller.BLOCK, V2, V2.Controller.Parameter)
                if V2 in V2.Controller.IV:
                    V2.status = V2.path.direct
                    V2.Parking = True
                
                    V2.Controller.IV.remove(V2)
                    V2.Controller.IV_num -= 1
                
    def create_show(self):
        #AGVs
        p = [(self.Controller.Origin[0]+self.x)*self.Controller.unit, (self.Controller.Origin[1]+self.Controller.Hight-self.y)*self.Controller.unit]
        p1 = [(self.Controller.Origin[0]-0.4+self.x)*self.Controller.unit, (self.Controller.Origin[1]+self.Controller.Hight-0.4-self.y)*self.Controller.unit]
        p2 = [(self.Controller.Origin[0]+0.4+self.x)*self.Controller.unit, (self.Controller.Origin[1]+self.Controller.Hight+0.4-self.y)*self.Controller.unit]

        self.Show = self.Controller.canvas.create_oval(p1[0], p1[1], p2[0], p2[1], fill = "purple")
        self.show = self.Controller.canvas.create_text(p[0], p[1], text = "A" + str(self.ID), font = ("arial", 10), fill = "yellow")
        
    def moving(self, action):
        base_action = [0, 0]
        if action == 1:
            base_action = [0, -1]
        elif action == 2:
            base_action = [0, 1]
        elif action == 3:
            base_action = [-1, 0]
        elif action == 4:
            base_action = [1, 0]
        dist = self.Controller.unit/self.Controller.split
        self.Controller.canvas.move(self.Show, base_action[0]*dist
                             , base_action[1]*dist)
        self.Controller.canvas.move(self.show, base_action[0]*dist
                             , base_action[1]*dist)
        self.Controller.window.update()
                

        
        
class Charging(object):
    def __init__(self, env, Controller, x, y):
        self.env = env
        self.Controller = Controller
        self.x = x
        self.y = y
        #self.resource = simpy.Resource(self.env, 10)
        self.Controller.mesh[self.x][self.y]  = simpy.Resource(self.env, 10)
        
        #image
        
        p = [(self.Controller.Origin[0]+self.x)*self.Controller.unit, (self.Controller.Origin[1]+self.Controller.Hight-self.y)*self.Controller.unit]
        p1 = [(self.Controller.Origin[0]-0.7+self.x)*self.Controller.unit, (self.Controller.Origin[1]+self.Controller.Hight-0.7-self.y)*self.Controller.unit]
        p2 = [(self.Controller.Origin[0]+0.7+self.x)*self.Controller.unit, (self.Controller.Origin[1]+self.Controller.Hight+0.7-self.y)*self.Controller.unit]
        self.Show = self.Controller.canvas.create_oval(p1[0], p1[1], p2[0], p2[1], fill = "light blue")
        self.show = self.Controller.canvas.create_text(p[0], p[1], text = "C", font = ("arial", 10), fill = "purple")
        
        
            
                
                
                
                
class Center(object):
    def __init__(self, env, x, y, routRule, AGV_num, WS_num, AGV_disRuleV, AGV_disRuleW, Ledispatch = "None", Task = ["None", "None"]):
        self.Warmup = 2000
        self.Period = 2000
        
        self.env = env
        self.Time = 0
        self.AGVs = []
        self.x = x
        self.y = y
        self.AGV_num = AGV_num
        self.routRule = routRule
        self.Parameter = [0.6, 0.3, 0.1, 3, 7]
        self.Task = Task
        
        self.FTmatrix = [[0, 15, 5, 8, 15, 18, 21, 20, 17, 14, 10, 1],
                         [15, 0, 10, 9, 8, 5, 6, 5, 8, 11, 5, 16],
                         [5, 10, 0, 3, 10, 13, 16, 15, 14, 17, 9, 6],
                         [8, 9, 3, 0, 7, 10, 13, 14, 13, 16, 10, 9],
                         [15, 8, 10, 7, 0, 3, 6, 13, 16, 19, 13, 16],
                         [18, 5, 13, 10, 3, 0, 3, 10, 13, 16, 11, 19],
                         [21, 6, 16, 13, 6, 3, 0, 7, 10, 13, 11, 22],
                         [20, 5, 15, 14, 13, 10, 7, 0, 3, 6, 10, 21],
                         [17, 8, 14, 13, 16, 13, 10, 3, 0, 3, 7, 18],
                         [14, 11, 17, 16, 19, 16, 13, 6, 3, 0, 6, 15],
                         [10, 5, 9, 10, 13, 11, 11, 10, 7, 6, 0, 11],
                         [1, 16, 6, 9, 16, 19, 22, 21, 18, 15, 11, 0]]
        
        PD = [[9, 8], [5, 2], [8, 2], [13, 4], [13, 7], [13, 10], [3, 13], [6, 13], [9, 13], [4, 8]]
        self.BLOCK = []
        for i in range(len(PD)):
            block = PD[i]
            if i == 0 or i == 1 or i == 2 or i == 9:
                block[1] -= 1
                self.BLOCK.append(block)
                self.BLOCK.append([block[0] - 1, block[1] - 1])
                self.BLOCK.append([block[0] - 1, block[1]])
                self.BLOCK.append([block[0], block[1] - 1])
                self.BLOCK.append([block[0] + 1, block[1] - 1])
                self.BLOCK.append([block[0] + 1, block[1]])
            elif i == 6 or i == 7 or i == 8:
                block[1] += 2
                self.BLOCK.append(block)
                self.BLOCK.append([block[0] - 1, block[1] - 1])
                self.BLOCK.append([block[0] - 1, block[1]])
                self.BLOCK.append([block[0], block[1] - 1])
                self.BLOCK.append([block[0] + 1, block[1] - 1])
                self.BLOCK.append([block[0] + 1, block[1]])
            else:
                block[0] += 1
                self.BLOCK.append(block)
                self.BLOCK.append([block[0], block[1] - 1])
                self.BLOCK.append([block[0], block[1] + 1])
                self.BLOCK.append([block[0] + 1, block[1] - 1])
                self.BLOCK.append([block[0] + 1, block[1]])
                self.BLOCK.append([block[0] + 1, block[1] + 1])
                
        
        
        self.V1 = 0 #The Vehicle found conflict
        self.V2 = 0 #The Vehicle V1 encounter
        
        
        self.mesh = self.create_path(self.env, x, y, 1, self.BLOCK)
        self.speed = 15
        #self.env.process(self.Conflict_Controll())
        self.env.process(self.TimeCount())
        self.env.process(self.Job_arrive())
        
        ###
        self.Hight = self.y-1
        self.Width = self.x-1
        self.unit = 40
        self.Origin = [3, 2]
        self.split = 5
        self.BuildFloor()
        
        self.jobs = []
        self.job_num = 0
        self.Rtravel = 0
        self.IV = []
        self.IV_num = 0
        self.stations = []
        self.WS_num = WS_num
        
        
        self.Tardiness = []
        self.MeanTardiness = 0
        self.FlowTime = []
        self.sysJob = []
        self.Makespan = 0
        self.Throughput = 0
        
        self.Create_Stations()
        self.Create_AGVs()
        #self.Create_Jobs()
        
        self.status = Le.Status(self)
        self.AGV_disRuleV = 0
        self.AGV_disRuleW = 0
        
        if AGV_disRuleV >= 6 or self.Task[0] == "Collect":
            self.set_LE(Ledispatch[0], "None", "V")
        self.tempV = AGV_disRuleV
            
        if AGV_disRuleW >= 3 or self.Task[1] == "Collect":
            self.set_LE("None", Ledispatch[1], "W")
        self.tempW = AGV_disRuleW
            
        
        self.Cstation = Charging(self.env, self, x=0, y=10)
        
    def set_LE(self, LEV, LEW, Type):
        if Type == "V":
            self.LEV = LEV
        elif Type == "W":
            self.LEW = LEW
        
        
                        
    def create_path(self, env, x, y, Capacity, block):
        mesh = []
        for i in range(y):
            temp = []
            for j in range(x):
                if [i, j] in block:
                    a = simpy.Resource(env, 1)
                    temp.append(a)
                elif [i, j] in [[9, 8], [5, 2], [8, 2], [13, 4], [13, 7], [13, 10], [3, 13], [6, 13], [9, 13], [4, 8], [0, 1], [0, 2]]:
                    a = simpy.Resource(env, 1)
                    temp.append(a)
                else:
                    temp.append(simpy.Resource(env, Capacity))
            mesh.append(temp)
        return mesh
        

    def Job_arrive(self):
        
        while True:
            inter_arrival = np.random.exponential(34)
                
            yield self.env.timeout(inter_arrival)
            t = np.random.choice(range(10))
            Job(self.env, self, t+1)
            '''
        while True:
            inter_arrival = np.random.exponential(105)
                
            yield self.env.timeout(inter_arrival)
            t = np.random.choice(range(10))
            Job(self.env, self, t+1)
            
        while True:
            inter_arrival = np.random.exponential(36)
                
            yield self.env.timeout(inter_arrival)
            t = np.random.choice(range(10))
            a = Job(self.env, self, t+1)
            a.create2(t+1)
            
        while True:
            inter_arrival = np.random.exponential(54)
                
            yield self.env.timeout(inter_arrival)
            t = np.random.choice(range(10))
            a = Job(self.env, self, t+1)
            a.create2(t+1)
            '''
    
    def TimeCount(self):
        yield self.env.timeout(0)
        self.Time = 0
        while True > 0:
            yield self.env.timeout(1)
            self.Time += 1
            
            self.canvas.delete(self.time)
                
            self.TIME = self.canvas.create_text(self.unit\
                                            , 0.5*self.unit, text = "Time:"\
                                            , font = ("arial", 20)\
                                            , fill = "Blue")
            self.time = self.canvas.create_text(2.5*self.unit\
                                            , 0.5*self.unit, text = str(round(self.Time))\
                                            , font = ("arial", 20)\
                                            , fill = "Blue")
            
            for i in range(self.split):
                    
                yield self.env.timeout(1/self.split)
                self.Time += 1/self.split
                if self.speed < 50:
                    time.sleep(1/(self.split*self.speed))
                    
                    
                    
            if round(self.Time) == self.Warmup:
                if hasattr(self, "LEV") and self.Task[0] != "Collect":
                    if self.tempV == 6:
                        s = self.status.Get("v")
                    else:
                        s = self.status.Get("V")
                    self.status.currentstate[0] = s
                    a = self.LEV.Choose_act(s)
                    self.status.currentact[0] = a
                    self.AGV_disRuleV = a
                elif self.Task[0] == "Collect":
                    self.status.currentstate[0] = self.status.Get("V")
                    self.AGV_disRuleV = self.tempV
                else:
                    self.AGV_disRuleV = self.tempV
                if hasattr(self, "LEW") and self.Task[1] != "Collect":
                    if self.tempW == 3:
                        s = self.status.Get("w")
                    else:
                        s = self.status.Get("W")
                    self.status.currentstate[1] = s
                    a = self.LEW.Choose_act(s)
                    self.status.currentact[1] = a
                    self.AGV_disRuleW = a
                elif self.Task[1] == "Collect":
                    self.status.currentstate[1] = self.status.Get("W")
                    self.AGV_disRuleW = self.tempW
                else:
                    self.AGV_disRuleW = self.tempW
                    
                self.Tardiness = []
                self.status.tardiness = []
            

            if round(self.Time - self.Warmup) % self.Period == 0 and round(self.Time) > self.Warmup:
                #Vehicle Initiated LE
                
                if hasattr(self, "LEV") or self.Task[0] == "Collect":
                    s = self.status.currentstate[0]
                    s_ = self.status.Get("V")
                    r = -self.status.meantardiness
                    if self.Task[0] == "Collect":
                        self.LEV.Memorize(s, r)
                    elif self.tempV == 8:
                        self.AGV_disRuleV = self.LEV.Choose_act(s_)
                    else:
                        self.status.currentstate[0] = s_
                        a = self.status.currentact[0]
                        self.AGV_disRuleV = self.LEV.Choose_act(s_)
                        if self.tempV == 6:
                            s_ = self.status.Get("v")
                            self.LEV.Learning(s, a, s_, r)
                        else:
                            self.LEV.store_transition(s, a, s_, r)
                        self.status.currentact[0] = self.AGV_disRuleV
                
                if hasattr(self, "LEW") or self.Task[1] == "Collect":
                    s = self.status.currentstate[1]
                    s_ = self.status.Get("W")
                    r = -self.status.meantardiness
                    if self.Task[1] == "Collect":
                        self.LEW.Memorize(s, r)
                    elif self.tempW == 5:
                        self.AGV_disRuleW = self.LEW.Choose_act(s_)
                    else:
                        self.status.currentstate[1] = s_
                        a = self.status.currentact[1]
                        self.AGV_disRuleW = self.LEW.Choose_act(s_)
                        if self.tempW == 3:
                            s_ = self.status.Get("w")
                            self.LEW.Learning(s, a, s_, r)
                        else:
                            self.LEW.store_transition(s, a, s_, r)
                        self.status.currentact[1] = self.AGV_disRuleW
                    
                    self.status.tardiness = []
                    self.status.throughput = 0
                    self.status.meantardiness = 0
                    
            for i in self.AGVs:
                i.Electricity -= 1/7200
            
            if round(self.Time - self.Warmup) % 500 == 0:
                tard = copy.deepcopy(self.Tardiness)
                tard1 = copy.deepcopy(self.status.tardiness)
                for i in self.sysJob:
                    tard.append(max(0, self.env.now - i.DueDate))
                    tard1.append(max(0, self.env.now - i.DueDate))
                self.MeanTardiness = np.mean(tard)
                self.status.meantardiness = np.mean(tard1)
                
                
                
    def Create_Stations(self):
        for i in range(self.WS_num):
            Station(self.env, self, i)
            
    def Create_AGVs(self):
        for i in range(self.AGV_num):
            Vehicle(self.env, self, self.mesh, i+1, x = 0, y = 10)
            
    def Create_Jobs(self):
        for i in range(10):
            Job(self.env, self, i+1)
            
    def BuildFloor(self):
        
        self.window = tk.Tk()
        
        self.window.title("Flexible Manufacturing System")
        self.window.geometry("{1}x{1}".format((self.Hight+4)*self.unit
                      , (self.Width+5)*self.unit))
        self.canvas = tk.Canvas(bg = "white", height = (self.Hight+5)\
                *self.unit, width = (self.Width+5)*self.unit)

        #Grid Layout
        for c in range(0, (self.Width*self.unit+1), self.unit):
            x0, y0, x1, y1 = self.Origin[0]*self.unit+c, self.Origin[1]*self.unit\
                    , self.Origin[0]*self.unit+c, (self.Hight+self.Origin[1])\
                    *self.unit
            self.canvas.create_line(x0, y0, x1, y1)

        for r in range(0, (self.Hight*self.unit+1), self.unit):
            x0, y0, x1, y1 = self\
                    .Origin[0]*self.unit, self.Origin[1]*self.unit+r\
                    , (self.Width+self.Origin[0])*self.unit\
                    , self.Origin[1]*self.unit+r
            self.canvas.create_line(x0, y0, x1, y1)

        #Loading Point
        L = [[0, 2]]
        for i in range(len(L)):
            p1 = [(self.Origin[0]+L[i][0])*self.unit-5, (self.Origin[1]+self.Hight-L[i][1])\
                  *self.unit-5]
            p2 = [(self.Origin[0]+L[i][0])*self.unit+5, (self.Origin[1]+self.Hight-L[i][1])\
                  *self.unit+5]
            self.canvas.create_oval(p1[0], p1[1], p2[0], p2[1], fill = "blue")
        #Unloading Point
        U = [[0, 1]]
        for i in range(len(U)):
            p1 = [(self.Origin[0]+U[i][0])*self.unit-5, (self.Origin[1]+self.Hight-U[i][1])\
                  *self.unit-5]
            p2 = [(self.Origin[0]+U[i][0])*self.unit+5, (self.Origin[1]+self.Hight-U[i][1])\
                  *self.unit+5]
            self.canvas.create_oval(p1[0], p1[1], p2[0], p2[1], fill = "green")
        #P/D Point
        PD = [[9, 8], [5, 2], [8, 2], [13, 4], [13, 7], [13, 10], [3, 13], [6, 13], [9, 13], [4, 8]]
        for i in range(len(PD)):
            p1 = [(self.Origin[0]+PD[i][0])*self.unit-5, (self.Origin[1]\
                  +self.Hight-PD[i][1])*self.unit-5]
            p2 = [(self.Origin[0]+PD[i][0])*self.unit+5, (self.Origin[1]\
                  +self.Hight-PD[i][1])*self.unit+5]
            self.canvas.create_oval(p1[0], p1[1], p2[0], p2[1], fill = "black")
            

        #AS/RS
        p = [1.5*self.unit, (self.Origin[1]+self.Hight*3/4+1)*self.unit]
        p1 = [2.5*self.unit, (self.Origin[1]+self.Hight*3/4+4)*self.unit]
        p2 = [0.5*self.unit, (self.Origin[1]+self.Hight*3/4-2)*self.unit]
        self.canvas.create_rectangle(p1[0], p1[1], p2[0], p2[1]\
                                     , fill = "yellow")
        self.canvas.create_text(p[0], p[1], text = "AS/RS"
                                , font = ("arial", 12), fill = "black")

        #Workstations
        for i in range(len(PD)):
            if i == 0 or i == 1 or i == 2 or i == 9:
                PD[i][1] -= 1.5
            elif i == 3 or i == 4 or i == 5:
                PD[i][0] += 1.5
            else:
                PD[i][1] += 1.5
        for i in range(len(PD)):
            p = [(self.Origin[0]+PD[i][0])*self.unit, (self.Origin[1]\
                 +self.Hight-PD[i][1])*self.unit]
            if i == 3 or i == 4 or i == 5:
                p1 = [(self.Origin[0]+PD[i][0]-1)*self.unit, (self.Origin[1]\
                      +self.Hight-PD[i][1]-1.5)\
                      *self.unit]
                p2 = [(self.Origin[0]+PD[i][0]+1)*self.unit, (self.Origin[1]\
                      +self.Hight-PD[i][1]+1.5)\
                      *self.unit]
            else:
                p1 = [(self.Origin[0]+PD[i][0]-1.5)*self.unit, (self.Origin[1]\
                      +self.Hight-PD[i][1]-1)\
                      *self.unit]
                p2 = [(self.Origin[0]+PD[i][0]+1.5)*self.unit, (self.Origin[1]\
                      +self.Hight-PD[i][1]+1)\
                      *self.unit]
            self.canvas.create_rectangle(p1[0], p1[1], p2[0], p2[1]\
                                         , fill = "orange")
            self.canvas.create_text(p[0], p[1], text = "W"+str(i+1)\
                                    , font = ("arial", 15), fill = "black")
            
        self.time = self.canvas.create_text(2.5*self.unit\
                                            , 0.5*self.unit, text = ""\
                                            , font = ("arial", 20)\
                                            , fill = "Blue")

        
        self.canvas.pack()

            
'''
env = simpy.rt.RealtimeEnvironment(factor = 0.5, strict=False)
Controller = Center(env, Hight+1, Width+1, 1)
Controller.fms = 0
Controller.create_task([1, 7], [1, 1])
Controller.create_task([6, 2], [0, 2])
Controller.create_task([1, 1], [1, 7])
        
env.run()
'''
