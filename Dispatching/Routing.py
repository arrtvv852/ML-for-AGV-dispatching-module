# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:55:40 2017

@author: CIMlab徐孟維
"""
'''
Alpha = 0.06
Beta = 0.03
Gamma = 0.01
Dist1 = 3
Dist2 = 7
'''

class ShortestPath:
    def __init__(self, current, goal, rule, x, y, block, Vehicle, Parameter):
        self.rule = rule
        self.current = current
        self.Vehicle = Vehicle
        #Parameters:
        self.Alpha = Parameter[0]
        self.Beta = Parameter[1]
        self.Gamma = Parameter[2]
        self.Dist1 = Parameter[3]
        self.Dist2 = Parameter[4]
        #A*
        self.path, self.direct, self.distance = self.Astar(current, goal, x, y, block)
        
    
    class Node(object):
        def __init__(self, actual, estimate, path, direct, position):
            self.actual = actual
            self.estimate = estimate
            self.path = path
            self.direct = direct
            self.position = position
            
    def Heuristic_dist(self, current, goal):
        if self.rule == 0:
            #Dijkstra's Distance
            return 0
        elif self.rule == 1:
            #Manhattan Distance
            dist = abs(goal[0]-current[0])+abs(goal[1]-current[1])
            return dist
        elif self.rule == 2:
            w = 0
            for i in range(self.Vehicle.Controller.AGV_num):
                pos = [self.Vehicle.Controller.AGVs[i].x, self.Vehicle.Controller.AGVs[i].y]
                remote = abs(pos[0]-current[0])+abs(pos[1]-current[1])
                if self.current == pos:
                    w = w
                elif remote < self.Dist1:
                    w += self.Alpha
                elif remote < self.Dist2:
                    w += self.Beta
                else:
                    w += self.Gamma
            dist = abs(goal[0]-current[0])+abs(goal[1]-current[1]) + w
            return dist
        elif self.rule == 3:
            w = 0
            for i in range(self.Vehicle.Controller.AGV_num):
                if self.Vehicle != self.Vehicle.Controller.AGVs[i]:
                    for j in self.Vehicle.Controller.AGVs[i].path.path:
                        remote = abs(j[0]-current[0])+abs(j[1]-current[1])
                        if remote < self.Dist1:
                            w += self.Alpha
                        elif remote < self.Dist2:
                            w += self.Beta
                        else:
                            w += self.Gamma
            dist = abs(goal[0]-current[0])+abs(goal[1]-current[1]) + w
            return dist
    
    def Astar(self, start, goal, x, y, block):
        #Initial
        node = self.Node(0, self.Heuristic_dist(start, goal), [], [], start)
        current = [node]
        histPath = []
        if start == goal:
            return [], [], 0
        count = 0
        #Search
        while goal not in current[0].path:
            
            count += 1
            if count > 1000:
                break
                
            
            parent = current[0]
            actual = parent.actual+1
            #Stay
            position = parent.position
            estimate = self.Heuristic_dist(position, goal)
            path = list(parent.path)
            direct = list(parent.direct)
            #path.append(position)
            #direct.append(0)
            node = self.Node(actual, estimate, path, direct, position)
            #current.append(node)
            #Up
            if parent.position[1]+1 <= y and [parent.position[0], parent.position[1]+1] not in histPath and [parent.position[0], parent.position[1]+1] not in block:
                parent = current[0]
                position = [parent.position[0], parent.position[1]+1]
                estimate = self.Heuristic_dist(position, goal)
                path = list(parent.path)
                direct = list(parent.direct)
                path.append(position)
                direct.append(1)
                node = self.Node(actual, estimate, path, direct, position)
                current.append(node)
                if position not in histPath:
                    histPath.append(position)
            #Down
            if parent.position[1]-1 >= 0 and [parent.position[0]\
                               , parent.position[1]-1] not in histPath and [parent.position[0], parent.position[1]-1] not in block:
                parent = current[0]
                position = [parent.position[0], parent.position[1]-1]
                estimate = self.Heuristic_dist(position, goal)
                path = list(parent.path)
                direct = list(parent.direct)
                path.append(position)
                direct.append(2)
                node = self.Node(actual, estimate, path, direct, position)
                current.append(node)
                if position not in histPath:
                    histPath.append(position)
            #Left
            if parent.position[0]-1 >= 0 and [parent.position[0]-1\
                               , parent.position[1]] not in histPath and [parent.position[0]-1, parent.position[1]] not in block:
                parent = current[0]
                position = [parent.position[0]-1, parent.position[1]]
                estimate = self.Heuristic_dist(position, goal)
                path = list(parent.path)
                direct = list(parent.direct)
                path.append(position)
                direct.append(3)
                node = self.Node(actual, estimate, path, direct, position)
                current.append(node)
                if position not in histPath:
                    histPath.append(position)
            #Right
            if parent.position[0]+1 <= x and [parent.position[0]+1\
                               , parent.position[1]] not in histPath and [parent.position[0]+1, parent.position[1]] not in block:
                parent = current[0]
                position = [parent.position[0]+1, parent.position[1]]
                estimate = self.Heuristic_dist(position, goal)
                path = list(parent.path)
                direct = list(parent.direct)
                path.append(position)
                direct.append(4)
                node = self.Node(actual, estimate, path, direct, position)
                current.append(node)
                if position not in histPath:
                    histPath.append(position)
            current.pop(0)
            if current == []:
                break
            #Sort
            for i in range(len(current)):
                if current[i].actual+current[i].estimate < current[0].actual\
                        +current[0].estimate:
                    current.insert(0, current.pop(i))
        if current != []:
            return current[0].path, current[0].direct, current[0].actual
        else:
            return [], [], 0