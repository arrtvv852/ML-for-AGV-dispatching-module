# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:09:18 2017

@author: CIMlab徐孟維
"""


#from ShopFloor import Vehicle, Job
from Routing import ShortestPath as sp

def VID(Controller, Vehicle, rule, routRule, Parameter):
    if Controller.job_num > 0:
        if Vehicle.startIdle:
            Vehicle.IdleTime += Controller.Time - Vehicle.startIdle
            Vehicle.startIdle = False
            
        candidate = Controller.jobs
        
        
        #STT
        if rule == 0:
            minTT = 999999
            index = 0
            for i in range(len(candidate)):
                if type(candidate[i].position.x) == type([]):
                    for j in range(len(candidate[i].position.x)):
                        path = sp([Vehicle.x, Vehicle.y], [candidate[i].position.x[j]\
                                            , candidate[i].position.y[j]]\
                                            , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, Vehicle, Parameter)
                        if path.distance < minTT:
                            minTT = path.distance
                            index = i
                else:
                    path = sp([Vehicle.x, Vehicle.y], [candidate[i].position.x\
                                            , candidate[i].position.y]\
                                            , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, Vehicle, Parameter)
                    if path.distance < minTT:
                        minTT = path.distance
                        index = i
        #Max Remaining Output Queue Space
        elif rule == 1:
            minOC = 9999999
            index = 0
            for i in range(len(candidate)):
                OC = candidate[i].position.out_num
                if OC < minOC:
                    minOC = OC
                    index = i
                    '''
        #LTT
        elif rule == 2:
            maxTT = 0
            index = 0
            for i in range(len(candidate)):
                if type(candidate[i].position.x) == type([]):
                    for j in range(len(candidate[i].position.x)):
                        path = sp([Vehicle.x, Vehicle.y], [candidate[i].position.x[j]\
                                            , candidate[i].position.y[j]]\
                                            , routRule, Controller.x-1, Controller.y-1, [], Vehicle, Parameter)
                        if path.distance > maxTT:
                            maxTT = path.distance
                            index = i
                else:
                    path = sp([Vehicle.x, Vehicle.y], [candidate[i].position.x\
                                            , candidate[i].position.y]\
                                            , routRule, Controller.x-1, Controller.y-1, [], Vehicle, Parameter)
                    if path.distance > maxTT:
                        maxTT = path.distance
                        index = i
                        '''
        #MFCFS
        elif rule == 2:
            index = 0
        #DS            
        elif rule == 3:
            minST = 999999
            index = 0
            for i in range(len(candidate)):
                rdd = candidate[i].DueDate - Controller.Time
                rpt = 0
                rtt = 0
                for j in range(candidate[i].current_seq, len(candidate[i].seq)):
                    rpt += candidate[i].PT[j]
                for j in range(candidate[i].current_seq, len(candidate[i].seq)-1):
                    f,t = candidate[i].seq[j], candidate[i].seq[j+1]
                    rtt += Controller.FTmatrix[f][t]
                rot = rpt + rtt
                ST = rdd - rot
                if ST < minST:
                    minST = ST
                    index = i
        #EDD
        elif rule == 4:
            minDD = 999999
            index = 0
            for i in range(len(candidate)):
                if candidate[i].DueDate < minDD:
                    minDD = candidate[i].DueDate
                    index = i
                    '''
        #Wl
        elif rule == 5:
            minWL = 999999
            index = 0
            for i in range(len(candidate)):
                ns = candidate[i].seq[candidate[i].current_seq+1]
                wl = 0
                for j in range(len(Controller.stations[ns].in_queue)):
                    job = Controller.stations[ns].in_queue[j]
                    wl += job.PT[job.current_seq]
                if wl < minWL:
                    minWL = wl
                    index = i
                    '''
        #CR            
        elif rule == 5:
            minCR = 999999
            index = 0
            for i in range(len(candidate)):
                rdd = candidate[i].DueDate - Controller.Time
                rpt = 0
                rtt = 0
                for j in range(candidate[i].current_seq, len(candidate[i].seq)):
                    rpt += candidate[i].PT[j]
                for j in range(candidate[i].current_seq, len(candidate[i].seq)-1):
                    f,t = candidate[i].seq[j], candidate[i].seq[j+1]
                    rtt += Controller.FTmatrix[f][t]
                rot = rpt + rtt
                CR = rdd/rot
                if CR < minCR:
                    minCR = CR
                    index = i
            
        else:
            print("ERROR")
        
        
        Controller.job_num -= 1
        Vehicle.Goal = candidate[index]
        
        
        if type(candidate[index].position.x) == type([]):
            
            minTT = 999999
            indexj = 0
            for j in range(len(candidate[index].position.x)):
                path = sp([Vehicle.x, Vehicle.y], [candidate[index].position.x[j]\
                                            , candidate[index].position.y[j]]\
                                            , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, Vehicle, Parameter)
                if path.distance < minTT:
                    minTT = path.distance
                    indexj = j
            Vehicle.path = sp([Vehicle.x, Vehicle.y], [candidate[index].position.x[indexj]\
                                        , candidate[index].position.y[indexj]]\
                                        , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, Vehicle, Parameter)
        else:
            Vehicle.path = sp([Vehicle.x, Vehicle.y], [candidate[index].position.x\
                                        , candidate[index].position.y]\
                                        , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, Vehicle, Parameter)
        Vehicle.status = Vehicle.path.direct
        nextWS = Controller.stations[candidate[index].seq[candidate[index].current_seq+1]]
        
        if type(nextWS.x) == type([]):
            minTT = 999999
            index2 = 0
            for j in range(len(nextWS.x)):
                if type(candidate[index].position.x) == type([]):
                    path = sp([candidate[index].position.x[indexj], candidate[index].position.y[indexj]], [nextWS.x[j], nextWS.y[j]]\
                                , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, candidate[index], Parameter)
                    if path.distance < minTT:
                        minTT = path.distance
                        index2 = j
                else:
                    path = sp([candidate[index].position.x, candidate[index].position.y], [nextWS.x[j], nextWS.y[j]]\
                                , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, candidate[index], Parameter)
                    if path.distance < minTT:
                        minTT = path.distance
                        index2 = j
            nextpos = [nextWS.x[index2], nextWS.y[index2]]
        else:
            nextpos = [nextWS.x, nextWS.y]
        
        if type(candidate[index].position.x) == type([]):
            Vehicle.nextpath = sp([candidate[index].position.x[indexj], candidate[index].position.y[indexj]], nextpos\
                                        , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, Vehicle, Parameter)
        else:
            Vehicle.nextpath = sp([candidate[index].position.x, candidate[index].position.y], nextpos\
                                        , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, Vehicle, Parameter)
        
        candidate.pop(index)
        Controller.IV.remove(Vehicle)
        Controller.IV_num -= 1
        
    elif [Vehicle.x, Vehicle.y] == [Controller.stations[Controller.WS_num-1].x, Controller.stations[Controller.WS_num-1].y]:
    #else:
        Vehicle.path = sp([Vehicle.x, Vehicle.y], [Controller.Cstation.x\
                                        , Controller.Cstation.y]\
                                        , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, Vehicle, Parameter)
        Vehicle.status = Vehicle.path.direct
        Vehicle.Parking = True
        
        Controller.IV.remove(Vehicle)
        Controller.IV_num -= 1
        
        
        
def WID(Controller, Job, rule, routRule, Parameter):
    if Controller.IV_num > 0:
        candidate = Controller.IV
        #NV
        if rule == 0:
            minTT = 999999
            index = 0
            if type(Job.position.x) == type([]):
                for j in range(len(Job.position.x)):
                    for i in range(len(candidate)):
                        path = sp([Job.position.x[j], Job.position.y[j]]\
                                , [candidate[i].x, candidate[i].y]\
                                , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, candidate[i], Parameter)
                        if path.distance < minTT:
                            minTT = path.distance
                            index = i
            else:
                for i in range(len(candidate)):
                    path = sp([Job.position.x, Job.position.y]\
                            , [candidate[i].x, candidate[i].y]\
                            , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, candidate[i], Parameter)
                    if path.distance < minTT:
                        minTT = path.distance
                        index = i
                        '''
        #FV
        elif rule == 1:
            maxTT = 0
            index = 0
            if type(Job.position.x) == type([]):
                for j in range(len(Job.position.x)):
                    for i in range(len(candidate)):
                        path = sp([Job.position.x[j], Job.position.y[j]]\
                                , [candidate[i].x[j], candidate[i].y[j]]\
                                , routRule, Controller.x-1, Controller.y-1, [], candidate[i], Parameter)
                        if path.distance > maxTT:
                            maxTT = path.distance
                            index = i
            else:
                for i in range(len(candidate)):
                    path = sp([Job.position.x, Job.position.y]\
                            , [candidate[i].x, candidate[i].y]\
                            , routRule, Controller.x-1, Controller.y-1, [], candidate[i], Parameter)
                    if path.distance > maxTT:
                        maxTT = path.distance
                        index = i
                        '''
        #LIV
        elif rule == 1:
            maxIT = 0
            index = 0
            for i in range(len(candidate)):
                if candidate[i].Idling:
                    IT = Controller.Time - candidate[i].Idling
                else:
                    IT = 0
                if IT > maxIT:
                    maxIT = IT
                    index = i
        #LU
        elif rule == 2:
            minUT = 999999
            index = 0
            for i in range(len(candidate)):
                if candidate[i].startIdle:
                    UT = Controller.Time - candidate[i].startIdle + candidate[i].IdleTime
                else:
                    UT = candidate[i].IdleTime
                if UT < minUT:
                    minUT = UT
                    index = i
        else:
            print("ERROR")
            
        if candidate[index].startIdle:
            candidate[index].IdleTime += Controller.Time - candidate[index].startIdle
            candidate[index].startIdle = False
        
        if type(Job.position.x) == type([]):
            minTT = 999999
            for j in range(len(Job.position.x)):
                path = sp([Job.position.x[j], Job.position.y[j]]\
                            , [candidate[index].x[j], candidate[index].y[j]]\
                            , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, candidate[i], Parameter)
                if path.distance < minTT:
                    minTT = path.distance
                    pos = [Job.position.x[j], Job.position.y[j]]
        else:
            pos = [Job.position.x, Job.position.y]
    
        Controller.IV_num -= 1
        candidate[index].Goal = Job
        candidate[index].path = sp([candidate[index].x, candidate[index].y], pos\
                                        , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, candidate[index], Parameter)
        candidate[index].status = candidate[index].path.direct
        nextWS = Controller.stations[Job.seq[Job.current_seq+1]]
        if type(nextWS.x) == type([]):
            minTT = 999999
            index2 = 0
            for j in range(len(nextWS.x)):
                path = sp(pos, [nextWS.x[j], nextWS.y[j]]\
                            , routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, candidate[index], Parameter)
                if path.distance < minTT:
                    minTT = path.distance
                    index2 = j
            nextpos = [nextWS.x[index2], nextWS.y[index2]]
        else:
            nextpos = [nextWS.x, nextWS.y]
        candidate[index].nextpath = sp(pos, nextpos, routRule, Controller.x-1, Controller.y-1, Controller.BLOCK, candidate[index], Parameter)
        
        candidate.pop(index)
        Controller.jobs.remove(Job)
        Controller.job_num -= 1



            