# coding=utf-8
import sys
import MySQLdb
import os

class Analyzer(object):
    def __init__(self,ID,runningTime,activationFile):
        self.ID=ID
        self.runningTime=runningTime
        self.db = MySQLdb.connect("10.0.0.3", "root", "Finelab123...", "injection", charset="utf8")
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
                    self.sqlDict['running_time']=self.runningTime
            
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
        

