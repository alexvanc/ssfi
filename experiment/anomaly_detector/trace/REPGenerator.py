# coding=utf-8
"""
parse the traceData.dat
"""
import pymysql
import sys
import os
import json
import pickle

# class BatchREPGenerator(object):
#     def __init__(self,dataDir):
#         self.db = pymysql.connect("localhost", "root", "fake_passwd", "new_injection5", charset="utf8")
#         self.cursor = self.db.cursor()
#         self.dataDir=dataDir

#     def close(self):
#         self.cursor.close()
#         self.db.close()

#rtype=24 27 28 29
def start(dataDir):
    if os.path.exists(dataDir):
        dirs=os.listdir(dataDir)
        counter=0
        for fi in dirs:
            generator=REPGenerator(dataDir+"/"+fi) 
            generator.start()
            counter=counter+1
            if counter % 10 == 0:
                print("Processed: " + str(counter))

    else:
        print("Invalid folder")

def compareLTime(event):
    return event['ltime']


class REPGenerator(object):
    def __init__(self,dataDir):
        self.dataDir=dataDir
        self.dataPath=dataDir+"/logs/traceData.dat"
        self.eventCounter=0
        self.allOriginalEvent=[]
        self.allEvents=[]
        self.componentThreadMap=None
    
    def start(self):
        if os.path.exists(self.dataPath):
            with open(self.dataPath,'r') as traceDataFile:
                lines=traceDataFile.readlines()
                for line in lines:
                    event=self.parseOneTraceLine(line.strip())
                    self.allEvents.append(event)
                    if event['id']==-1:
                        print("Parse one line data error: "+self.dataPath+" "+line)
                        sys.exit()
                    self.eventCounter=self.eventCounter+1
            self.allOriginalEvent=self.allEvents
            print("Load traceData done!")
            self.generateThreadTree()
            self.assignComponentToEvents()
            print("Generating thread tree done!")
            self.generateREP()
            print("Generating REP done")

            
        else:
            print("No trace data found: "+self.dataPath)
    
    def generateThreadTree(self):
        allThreadEvents=[]
        
        
        for event in self.allEvents:
            if event['rtype']==27:
                allThreadEvents.append(event)
        sorted(allThreadEvents,key=compareLTime,reverse=True)
        print("All threadEvents: "+str(len(allThreadEvents)))
        
        for event in allThreadEvents:
            if event.get('tparent',-2)==-2:
                for event2 in allThreadEvents:
                    if event['pktid']==event2['ktid'] and event['ltime']>event2['ltime']:
                        event['tparent']=event2['id']
                        # event2.get('tchildren',[]).append(event['id'])
                        if event2.get('tchildren',None) is None:
                            event2['tchildren']=[event['id']]
                        else:
                            event2['tchildren']=event2['tchildren'].append(event['id'])

        print("Deciding children done!")
        self.assignThreadsToComponents(allThreadEvents)
    
    def assignComponentToEvents(self):
        counter={}
        for event in self.allEvents:
            if event['rtype']==27:
                event['component']=self.getComponentByKtid(event['pktid'])
                counter[event['component']]=counter.get(event['component'],0)+1
            else:
                event['component']=self.getComponentByKtid(event['ktid'])
                counter[event['component']]=counter.get(event['component'],0)+1
        
        print(counter)

    def assignThreadsToComponents(self,allThreadEvents):
        allStartThreadEvents=[]

        for event in allThreadEvents:
            if event.get('tparent',-2)==-2:
                allStartThreadEvents.append(event)

        allThreadGroups=[]

        for startThreadEvent in allStartThreadEvents:
            allThreadGroups.append([startThreadEvent['pktid']])
        
        
        for event in allThreadEvents:
            parentThreadEventID=event.get('tparent',-2)
            if parentThreadEventID!=-2:
                parentThreadEvent=self.getEventByID(parentThreadEventID)
                for threadGroup in allThreadGroups:
                    if parentThreadEvent['pktid'] in threadGroup:
                        threadGroup.append(event['pktid'])
                        break

        allComponents=self.loadAllComponents()
        print(allComponents)
        no_foundCounter=0
        for threadGroup in allThreadGroups:
            found=False
            # print(len(threadGroup))
            for ktid in threadGroup:
                if str(ktid) in allComponents:
                    found=True
                    self.assigneAComponentToThreads(allComponents[str(ktid)],threadGroup)
                    break

        # print("No Component thread: {}".format(no_foundCounter))
    
    def getEventByID(self,eventID):
        for event in self.allEvents:
            if event['id']==eventID:
                return event
        return None
    
    def getComponentByKtid(self,ktid):
        if self.componentThreadMap is None:
            with open("com_th_map.dict") as mfile:
                self.componentThreadMap==pickle.load(mfile)
        
        for component in self.componentThreadMap:
            allThreads=self.componentThreadMap[component]
            if ktid in allThreads:
                return component
        return "unknown"
        

    
    def assigneAComponentToThreads(self,componentName,threadGroup):
        if self.componentThreadMap is None:
            self.componentThreadMap={}
        if componentName not in self.componentThreadMap:
            self.componentThreadMap[componentName]=[]
        for threadKtid in threadGroup:
            self.componentThreadMap[componentName].append(threadKtid)



    
    def loadAllComponents(self):
        allComponents={}
        componentFilePath=self.dataDir+"/startresult.log"
        if os.path.exists(componentFilePath):
            with open(componentFilePath,'r') as cfile:
                lines=cfile.readlines()
                for line in lines:
                    items=line.strip().split(" ")
                    allComponents[str(items[0])]=items[1]
        return allComponents


    
    def parseOneTraceLine(self,content):
        event={}
        event['parent']=-2
        event['children']=None
        event['id']=self.eventCounter
        attr = content.strip().split("&")
        for item in attr:
            pair = item.strip().split("=",1)
            if(len(pair)==2):
                key=pair[0]
                if key=='ttime':
                    key='ltime'
                value=pair[1]
                if key in ['rtype',"pid","ktid","tid","ftype","ltime","ttime","length","supposed_length","rlength","dtype","process_id","k_thread_id","thread_id","p_process_id","pktid","p_thread_id"]:
                    value=int(pair[1])
                if key in ["uuid","unit_uuid"] and value.strip()=="":
                    value=None
                event[key] = value
            else:
                event['id']=-2
                print("Parse Event Error!")
                return event
        return event
        
    
    def generateREP(self):
        self.filterWhiteListEvent()
        self.filterFailureEvent()
        self.aggregateUnit()
        self.aggregateUnits()
        self.generateReport()

    def generateReport(self):
        print("Total events: "+str(self.eventCounter))
        print("Total valid events: "+str(len(self.allEvents)))
        unitStartEvents=self.getUnitStartEvents()
        print("Total unit start events: "+str(len(unitStartEvents)))
        # print("last one".format(unitStartEvents[-1]))
        for event in unitStartEvents:
            print(event)
            print("----\n")
        checkEvents=self.getAllEventOrderByTime()
        for event in checkEvents:
            print(event)
            print("----\n")

    
    def getUnitStartEvents(self):
        unitStartEvents=[]
        for event in self.allEvents:
            if event.get("l2_index",-1)==0 and event.get("l2_parent",None) is None:
                unitStartEvents.append(event)
        return unitStartEvents
    
    #for debugging
    def getAllEventOrderByTime(self):
        allEvents=[]
        for event in self.allEvents:
            if event['rtype']==27 and event['ktid']==10789:
                allEvents.append(event)
        return sorted(allEvents,key=compareLTime,reverse=False)

    
    def filterWhiteListEvent(self):
        tmpAllEvents=[]
        for event in self.allEvents:
            #rtype==24 mean it's a network related event
            if event['rtype']==24:
                if event['dtype']==4 or event['dtype']==10:
                    continue
            tmpAllEvents.append(event)
        self.allEvents=tmpAllEvents

    def filterFailureEvent(self):
        tmpAllEvents=[]
        for event in self.allEvents:
            #rtype==24 mean it's a network related event
            if event['rtype']==24:
                if event['ftype']%2==0 and (event['rlength']<=0 or event['uuid'] is None):
                    continue
            tmpAllEvents.append(event)
        self.allEvents=tmpAllEvents

    
    def aggregateUnit(self):
        tmpUniqueDict={}
        for event in self.allEvents:
            key=event['unit_uuid']+str(self.getKtidForEvent(event))

            if key in tmpUniqueDict:
                tmpUniqueDict[key].append(event)
            else:
                tmpUniqueDict[key]=[event]
        
        for key in tmpUniqueDict:
            allUnitEvents=tmpUniqueDict[key]
            sorted(allUnitEvents,key=compareLTime,reverse=False)

            earliestLTime=0
            counter=0
            for event in allUnitEvents:
                event['l2_index']=counter
                event['l2_length']=len(allUnitEvents)
                if counter==0:
                    earliestLTime=event['ltime']
                    event['l2_top_alt']=1
                elif event['ltime']==earliestLTime:
                    #whether to set the parent in this   
                    print("Same ltime found! "+key)
                    if event['rtype']==24 and event['ftype']%2==0 and event['uuid'] is None:
                        print("here")
                    print(event)
                    print(allUnitEvents[counter-1])
                    event['l2_top_alt']=1
                    event['parent']=allUnitEvents[counter-1]['id']
                    if allUnitEvents[counter-1].get('children',None) is None:
                        allUnitEvents[counter-1]['children']=[event['id']]
                    else:
                        allUnitEvents[counter-1]['children']=allUnitEvents[counter-1]['children'].append(event['id'])
                else:
                    event['parent']=allUnitEvents[counter-1]['id']
                    if allUnitEvents[counter-1].get('children',None) is None:
                        allUnitEvents[counter-1]['children']=[event['id']]
                    else:
                        allUnitEvents[counter-1]['children']=allUnitEvents[counter-1]['children'].append(event['id'])
                
                counter=counter+1

    def aggregateUnits(self):
        unitStartEventsList=[]
        for event in self.allEvents:
            if event.get("l2_top_alt",0)==1:
                unitStartEventsList.append(event)

        
        for event in unitStartEventsList:
            data_unit_id=event['unit_uuid']
            ktid=self.getKtidForEvent(event)
            
            for event2 in self.allEvents:
                #write events
                if event2['rtype']==24 and event2['ftype']%2==1 and event2['uuid']==data_unit_id and event2['ktid']==ktid:
                    event['l2_parent']=event2['id']
                    if event2.get('l2_children',None) is None:
                        event2['l2_children']=[event['id']]
                    else:
                        event2['l2_children']=event2['l2_children'].append(event['id'])
                #thread events
                elif event2['rtype']==27 and event2['ktid']==ktid and event2['unit_uuid']==data_unit_id:
                    event['l2_parent']=event2['id']
                    if event2.get('l2_children',None) is None:
                        event2['l2_children']=[event['id']]
                    else:
                        event2['l2_children']=event2['l2_children'].append(event['id'])
                #network events  
                elif event['rtype']==24 and event['ftype']%2==0 and event2['rtype']==24 and event2['ftype']%2==1 and event2['uuid']==event2['uuid']:
                    event['l2_parent']=event2['id']
                    if event2.get('l2_children',None) is None:
                        event2['l2_children']=[event['id']]
                    else:
                        event2['l2_children']=event2['l2_children'].append(event['id'])
    




            
    
    def getKtidForEvent(self,event):
        if event['rtype']==24 or event['rtype']==29 or event['rtype']==28:
            return event['ktid']
        elif event['rtype']==27:
            return event['pktid']
        else: # for log
            return event['ktid']



if __name__ == '__main__':
    start(sys.argv[1])
