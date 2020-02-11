# coding=utf-8
import sys
import MySQLdb
import os

class Analyzer(object):
    def __init__(self,injectionFile,activationFile):

        self.db = MySQLdb.connect("39.99.169.20", "alex", "fake_passwd", "injection_recovery", charset="utf8")
        self.cursor = self.db.cursor()
        self.activationFile=activationFile
        self.injectionFile=injectionFile
        self.sqlDict={}
        self.countDict={}
    
    def recoverInjection(self):
        if os.path.exists(self.injectionFile):
            injectionFile=open("/tmp/injection.log",'r')
            lines=injectionFile.readlines()
            injectionFile.close()
            for line in lines:
                self.parseInjection(line)
                self.insertIntoDB()
                self.sqlDict={}
    
    def recoverActivation(self):
        #get activation info
        if os.path.exists(self.activationFile):
            activationFile=open(self.activationFile)
            aLines=activationFile.readlines()
            activationFile.close()
            for line in aLines:
                idKey=line.strip()
                if self.countDict.has_key(idKey):
                    self.countDict[idKey]=self.countDict[idKey]+1
                else:
                    self.countDict[idKey]=1
        
        self.updateDB()
            
                
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
        try:
            self.cursor.execute(insertSQL)
            self.db.commit()
        except expression as identifier:
            print(identifier)
        
    
    def updateDB(self):
        allKeys=self.countDict.keys()
        for key in allKeys:
            value=self.countDict[key]
            sql="update injection_record_hadoop set activated=1, activated_number=%s where fault_id='%s'" % (value,key)
            self.cursor.execute(sql)
            self.db.commit()
      
            
            
            
        
    
    def parseInjection(self,lastLine):
        tuples=lastLine.split("\t")
        for item in tuples:
            key=item.split(":")[0]
            value=item.split(":")[1]
            columnKey=""
            
            if(key=="ID"):
                columnKey="fault_id"                    
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
                
            self.sqlDict[columnKey]=item.split(":")[1].replace("'","`")
            
    
    def close(self):
        self.cursor.close()
        self.db.close()


if __name__ == '__main__':
    
    analyzer=Analyzer("/tmp/injection.log","/work/hadoop/hadoop-output/activation.log")
    analyzer.recoverInjection()
    analyzer.recoverActivation()
    analyzer.close()
        

