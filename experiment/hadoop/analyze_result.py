# coding=utf-8
import sys
import MySQLdb
import os

class Analyzer(object):
    def __init__(self,ID,runningTime,activationFile):
        self.ID=ID
        self.runningTime=runningTime
        self.db = MySQLdb.connect("10.58.0.200", "root", "test1234", "injection", charset="utf8")
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
        insertTemp="insert into injection_record_hadoop(%s) values(%s)" % (tableColumns, valueTml)
        insertSQL=insertTemp % tuple(values)
        self.cursor.execute(insertSQL)
        self.db.commit()
    
    def parseFailure(self):
        crashed=False 
        rightOutput=False
        exceptions=False
        timeOut=False
        if os.path.exists("/tmp/startresult.log"):
            outputFile=open("/tmp/startresult.log",'r')
            lines=outputFile.readlines()
            outputFile.close()
            if len(lines)!=7:
                crashed=True
        else:
            crashed=True
        
        if os.path.exists("/tmp/runResult.txt"):
            resultFile=open("/tmp/runResult.txt")
            lines=resultFile.readlines()
            self.sqlDict['running_output']="".join(lines)
            resultFile.close()
            if len(lines)==0:
                crashed=True
            elif len(lines)==3:
                resultDict={}
                for line in lines:
                    kvp=line.strip().split("\t")
                    print(line)
                    if len(kvp)!=2:
                        break
                    else:
                        resultDict[kvp[0]]=kvp[1]
                if resultDict.has_key('Docker') and resultDict['Docker']=='1' and resultDict.has_key('Hadoop') and resultDict['Hadoop']=='1' and resultDict.has_key('Hello') and resultDict['Hello']=='2':
                    rightOutput=True
        else:
            crashed=True
        
        
        self.sqlDict['running_time']=self.runningTime
        rtime=int(self.runningTime)
        if rtime>240000:
            timeOut=True
            #check hadoop running logs after being killed
            directory="/tmp/hadoop/logs/"+self.ID+"/klogs"
            for dirpath,subdirs,files in os.walk(directory):
                for filename in files:
                    if filename.endswith(".log") or filename.endswith("audit") or filename=="syslog" or filename=="stderr" or filename=="stdout":
                        logFile=open(os.path.join(dirpath,filename))
                        lines=logFile.readlines()
                        if len(lines)!=0:
                            exceptions=True
                            self.sqlDict["running_error"]=self.sqlDict.get("running_error","")+"".join(lines)
                            self.sqlDict["error_file"]=self.sqlDict.get("error_file","")+os.path.join(dirpath,filename)+"\n"
                    elif filename.endswith(".out"):
                        outFile=open(os.path.join(dirpath,filename))
                        lines=outFile.readlines()
                        if len(lines)!=0 and not lines[0].startswith("ulimit"):
                            exceptions=True
                            self.sqlDict["running_error"]=self.sqlDict.get("running_error","")+"".join(lines)
                            self.sqlDict["error_file"]=self.sqlDict.get("error_file","")+os.path.join(dirpath,filename)+"\n"
                    else:
                        #unexpected log file means error here
                        exceptions=True
                        self.sqlDict["error_file"]=self.sqlDict.get("error_file","")+os.path.join(dirpath,filename)+"\n"
                        
        else:
            directory="/tmp/hadoop/logs/"+self.ID+"/logs"
            for dirpath,subdirs,files in os.walk(directory):
                for filename in files:
                    if filename.endswith(".log") or filename.endswith("audit") or filename=="syslog" or filename=="stderr" or filename=="stdout":
                        logFile=open(os.path.join(dirpath,filename))
                        lines=logFile.readlines()
                        if len(lines)!=0:
                            exceptions=True
                            self.sqlDict["running_error"]=self.sqlDict.get("running_error","")+"".join(lines)
                            self.sqlDict["error_file"]=self.sqlDict.get("error_file","")+os.path.join(dirpath,filename)+"\n"
                    elif filename.endswith(".out"):
                        outFile=open(os.path.join(dirpath,filename))
                        lines=outFile.readlines()
                        if len(lines)!=0 and not lines[0].startswith("ulimit"):
                            exceptions=True
                            self.sqlDict["running_error"]=self.sqlDict.get("running_error","")+"".join(lines)
                            self.sqlDict["error_file"]=self.sqlDict.get("error_file","")+os.path.join(dirpath,filename)+"\n"
                    else:
                        #unexpected log file means error here
                        exceptions=True
                        self.sqlDict["error_file"]=self.sqlDict.get("error_file","")+os.path.join(dirpath,filename)+"\n"
        if self.sqlDict.has_key("running_error"):
            self.sqlDict["running_error"]=self.sqlDict["running_error"].replace("'","-")
            self.sqlDict["running_error"]=self.sqlDict["running_error"].replace('"','-')
            self.sqlDict["running_error"]=self.sqlDict["running_error"].replace('%','-')        

        if timeOut:
            if exceptions:
                self.sqlDict['failure_type']="Detected Hang"
            else:
                self.sqlDict['failure_type']="Hang"
        elif crashed:
            if exceptions:
                self.sqlDict['failure_type']="Detected Crash"
            else:
                self.sqlDict['failure_type']="Silent Crash"
        else:
            if rightOutput:
                if exceptions:
                    self.sqlDict['failure_type']="Benign"
                else:
                    self.sqlDict['failure_type']="No effect"
            else:
                if exceptions:
                    self.sqlDict['failure_type']="Data Corruption"
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
            elif key=="JarName":
                columnKey="jar_file"
            elif key=="ComponentName":
                columnKey="component"
            else:
                print("Unexpected key: "+key)
                sys.exit(1)
                
            self.sqlDict[columnKey]=item.split(":")[1]
            
    
    def close(self):
        self.cursor.close();
        self.db.close()


if __name__ == '__main__':
    if len(sys.argv)==4:
        analyzer=Analyzer(sys.argv[1],sys.argv[2],sys.argv[3])
        analyzer.analyze()
        analyzer.close()
        

