# coding=utf-8
import sys
import MySQLdb
import os

class Analyzer(object):
    def __init__(self,ID,runningTime,activationFile):
        self.ID=ID
        self.runningTime=runningTime
        self.db = MySQLdb.connect("137.82.252.59", "root", "fake_passwd", "injection", charset="utf8")
        self.cursor = self.db.cursor()
        self.activationFile=activationFile
        self.sqlDict={}
    
    def analyze(self):
        if os.path.exists("/tmp/injection.log"):
            injectionFile=open("/tmp/injection.log",'r')
            lines=injectionFile.readlines()
            injectionFile.close()
            lastLine=lines[len(lines)-1]
            self.parseInjection(lastLine)
            
            #get activation info
            if os.path.exists(self.activationFile):
                activationFile=open(self.activationFile)
                aLines=activationFile.readlines()
                activationFile.close()
                activationNumber=self.parseActivation(aLines)
                if activationNumber!=0:
                    self.sqlDict['activated']=str(1)
                    self.sqlDict['activated_number']=str(activationNumber)
                    self.parseFailure()
            
            self.insertIntoDB()
            
                
    def insertIntoDB(self):
        
        tableColumns=""
        valueTml=""
        valueHolder=[]
        values=[]
        
        allKeys=self.sqlDict.keys()

        for key in allKeys: 
            value=self.sqlDict[key].strip()
            values.append(value)
#             print(value)
            if value.isdigit():
                valueHolder.append("%s")
            else:
                valueHolder.append("'%s'")
        
        tableColumns=",".join(allKeys)
        valueTml=",".join(valueHolder)
        insertTemp="insert into injection_record_weka(%s) values(%s)" % (tableColumns, valueTml)
        insertSQL=insertTemp % tuple(values)
        self.cursor.execute(insertSQL)
        self.db.commit()
    
    def parseFailure(self):
        crashed=False 
        rightOutput=False
        exceptions=False
        timeOut=False
        if os.path.exists("/tmp/runResult.txt"):
            outputFile=open("/tmp/runResult.txt",'r')
            lines=outputFile.readlines()
            outputFile.close()
            self.sqlDict['running_output']="".join(lines)
            if len(lines)==0:
                crashed=True
            else:
                if len(lines)==21 and lines[5].strip().split("\t")==4:
		    items=lines[5].strip().split("\t")
		    if item[1]=="4" and lines[2]=="80":
                        rightOutput=True
        else:
            crashed=True
        if os.path.exists("/tmp/runError.txt"):
            errFile=open("/tmp/runError.txt",'r')
            lines=errFile.readlines()
            errFile.close()
            self.sqlDict['running_error']="".join(lines)
            self.sqlDict['running_error']=self.sqlDict['running_error'].replace('"','-')
            self.sqlDict['running_error']=self.sqlDict['running_error'].replace("'","-")
            if len(lines)>3:
                exceptions=True
        rtime=int(self.runningTime)
        if rtime>15000:
            timeOut=True
        self.sqlDict['running_time']=self.runningTime

        if timeOut:
            if exceptions:
                self.sqlDict['failure_type']="Detected Hang"
            else:
                self.sqlDict['failure_type']="Silent Hang"
        elif crashed:
            if exceptions:
                self.sqlDict['failure_type']="Detected Early Exit"
            else:
                self.sqlDict['failure_type']="Silent Early Exit"
        else:
            if rightOutput:
                if exceptions:
                    self.sqlDict['failure_type']="Benign"
                else:
                    self.sqlDict['failure_type']="No effect"
            else:
                if exceptions:
                    self.sqlDict['failure_type']="Detected Data Corruption"
                else:
                    self.sqlDict['failure_type']="Silent Data Corruption"

    
    def parseActivation(self,lines):
        tmpLines=[]
        #to avoid analyzing too many lines
        if len(lines)>100:
            tmpLines=lines[len(lines)-100:]
        else:
            tmpLines=lines
        
        counter=0
        for line in tmpLines:
            if line.strip()==self.ID:
                counter=counter+1
        return counter
            
            
            
        
    
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
            else:
                print("Unexpected key: "+key)
                continue               
            self.sqlDict[columnKey]=item.split(":")[1]
            
    
    def close(self):
        self.cursor.close();
        self.db.close()

            


if __name__ == '__main__':
    if len(sys.argv)==4:
        analyzer=Analyzer(sys.argv[1],sys.argv[2],sys.argv[3])
        analyzer.analyze()
        analyzer.close()
        

