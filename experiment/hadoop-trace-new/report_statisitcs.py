# coding=utf-8
import sys
import MySQLdb
import os
from datetime import datetime

class Analyzer(object):
    def __init__(self,ID,runningTime):
        self.ID=ID
        self.runningTime=runningTime
        self.db = MySQLdb.connect("39.99.169.20", "root", "fake_passwd", "detection", charset="utf8")
        self.cursor = self.db.cursor()
        self.activationFile=activationFile
        self.sqlDict={}
    
    def analyze(self):
        analyzeReporting()
        # analyzeDeepLog()
        # analyzeMetrics()
        # analyzeTrace()
        
            
                
    def analyzeReporting(self):
        sql="select fault_id,class,component,error_class,error_component,error_time,failure_type from injection_record_hadoop where failure_type<>'' and failure_type<>'Silent Benign' and with_bug=0 and error_log=1"
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        counterDict={}
        for result in results:
            failure_type=result[-1]
            if failure_type not in counterDict:
                counterDict[failure_type]={'counter':1}
            
    
    def checkRightClass(self,fi_class,found_class):
        if fi_class=="" or found_class=="":
            return False
        if fi_class.find(found_class)!=-1 or found_class.find(fi_class)!=-1:
            return True
    def checkRightComponent(self,fi_component,found_component):
        if fi_component=="" or found_component=="":
            return -1
        
        components=found_component.strip().split('\n')
        if len(components)==1:
            if components[0]==fi_component:
                return 0 #accurately found
            else:
                return -1
        else:
            if fi_component in components:
                return 1 # found but also propogate
            else:
                return 2 # propagate and not found

        return -1
    
    def getCheckLatency(self,act_time,found_time):
        if act_time>found_time: #invalid
            return -1
        else:
            diffrence=(found_time-act_time).seconds
            return diffrence

    
    def parseActivation(self,lines):
        tmpLines=[]
        #to avoid analyzing too many lines
        if len(lines)>100:
            tmpLines=lines[len(lines)-100:]
        else:
            tmpLines=lines
        
        counter=0
        millSeconds=0
        try:
            for line in tmpLines:
                tuples=line.strip().split(":")
                if tuples[1].strip()==self.ID:
                    counter=counter+1
                    millSeconds=int(tuples[0])
        except:
            print(line)
            return (0,0)
        return (counter,millSeconds)
            
    def parseInjection(self,lastLine):
        tuples=lastLine.split("\t")
        for item in tuples:
            key=item.split(":")[0]
            value=item.split(":")[1]
            columnKey=""
            
            if(key=="ID"):
                columnKey="fault_id"
                if value!=self.ID:
                    print("ID not equal")
                    sys.exit(1)
                    
            elif key=="FaultType" :
                columnKey="fault_type"
            elif key=="ActivationMode":
                columnKey="activation_mode"
            elif key=="Package":
                columnKey="package"
            elif key=="Class":
                columnKey="class"
            elif key=="Method":
                columnKey="method"
            elif key=="Action":
                columnKey="action"
            elif key=="VariableScope":
                columnKey="scope"
            elif key=="VariableName":
                columnKey="variable"
            elif key=="VariableType":
                columnKey="variable_type"
            elif key=="VariableValue":
                columnKey="variable_value"
            elif key=="ExeIndex":
                columnKey="exe_index"
            elif key=="JarName":
                columnKey="jar_file"
            elif key=="ComponentName":
                columnKey="component"
            else:
                print("Unexpected key: "+key)
                continue
                
            self.sqlDict[columnKey]=item.split(":")[1].replace("'",'"')
            
    
    def close(self):
        self.cursor.close()
        self.db.close()


if __name__ == '__main__':
    if len(sys.argv)==4:
        analyzer=Analyzer(sys.argv[1],sys.argv[2],sys.argv[3])
        analyzer.analyze()
        analyzer.close()
        

