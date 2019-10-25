# coding=utf-8
import sys
import pymysql
import os
import re
import csv
import numpy as np
import pandas as pd
import hashlib
import datetime as dt
from datetime import datetime
from dateutil import parser as ps
import matplotlib.pyplot as plt
import time
from shutil import copyfile
import log_parser

class Processor(object):
    def __init__(self,dataDir,prefix):
        self.db = pymysql.connect("10.58.0.200", "root", "test1234", "injection_merge", charset="utf8")
        self.cursor = self.db.cursor()
        self.dataDir=dataDir
        self.prefix=prefix
        self.sqlDict={}
        self.containerDict={}
    
    def start(self):
        if os.path.exists(self.dataDir):
            self.load_containers()
            # self.load_resource()
            dirs=os.listdir(self.dataDir+"/fi")
            for fi in dirs:
                self.processOneFI(fi)
    
    def load_containers(self):
        containerFile=open(self.dataDir+"/container.log")
        lines=containerFile.readlines()
        containerFile.close()
        lineCounter=0
        for line in lines:
            if lineCounter%2==0:
                container_name=line.strip()
                container_id=lines[lineCounter+1].strip()
                self.containerDict[container_name]=container_id
            lineCounter=lineCounter+1

    # def load_resource(self):
    #     monitorFile=open(self.dataDir+"/monitor.log")
    #     content=monitorFile.read()
    #     monitorFile.close()
    #     records=content.split("\n\n")
    #     print(len(records))
        

    def getContainerID(self,ID):
        sql="select fault_type from injection_merge_hadoop where fault_id='%s'" % ID
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        fault_type=results[0][0]
        containerID=""
        if fault_type=="ATTRIBUTE_SHADOWED_FAULT":
            containerID=self.containerDict['always_shadow']
        elif fault_type=="CONDITION_BORDER_FAULT":
            containerID=self.containerDict['always_border']
        elif fault_type=="CONDITION_INVERSED_FAULT":
            containerID=self.containerDict['always_inverse']
        elif fault_type=="EXCEPTION_SHORTCIRCUIT_FAULT":
            containerID=self.containerDict['always_short']
        elif fault_type=="EXCEPTION_UNCAUGHT_FAULT":
            containerID=self.containerDict['always_uncaught']
        elif fault_type=="EXCEPTION_UNHANDLED_FAULT":
            containerID=self.containerDict['always_unhandle']
        elif fault_type=="NULL_FAULT":
            containerID=self.containerDict['always_null']
        elif fault_type=="SWITCH_FALLTHROUGH_FAULT":
            containerID=self.containerDict['always_fallthrough']
        elif fault_type=="SWITCH_MISS_DEFAULT_FAULT":
            containerID=self.containerDict['always_default']
        elif fault_type=="UNUSED_INVOKE_REMOVED_FAULT":
            containerID=self.containerDict['always_remove']
        elif fault_type=="Value_FAULT":
            containerID=self.containerDict['always_value']
        else:
            print("Cannot find containerID") 
        
        return containerID

    
    def processOneFI(self,fi):
        if not fi.lower().startswith(self.prefix):
            return
        ID=fi
        containerID=""
        try:
            containerID=self.getContainerID(ID)
        except:
            print("Cannot find container for fault: "+ID)
            return
        
        timeResult=self.getRunningTime(ID)
        failure_type=""
        if self.IDActivated(ID):
            crashed=False
            rightOutput=False
            exceptions=False
            timeOut=False
            #check whether all hadoop services started
            if os.path.exists(self.dataDir+'/fi/'+ID+'/'+'startresult.log'):
                outputFile=open(self.dataDir+'/fi/'+ID+'/'+'startresult.log','r')
                lines=outputFile.readlines()
                outputFile.close()
                if len(lines)!=7:
                    crashed=True
            else:
                crashed=True
            #check whether running result is right
            if os.path.exists(self.dataDir+'/fi/'+ID+'/'+'runResult.txt'):
                resultFile=open(self.dataDir+'/fi/'+ID+'/'+'runResult.txt')
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
                    if ('Docker' in resultDict) and resultDict['Docker']=='1' and ('Hadoop' in resultDict) and resultDict['Hadoop']=='1' and ('Hello' in resultDict) and resultDict['Hello']=='2':
                        rightOutput=True
            else:
                crashed=True
            
            if timeResult['flag']=='ok' and timeResult['last_time']>180000:
                timeOut=True
            
            #check log for system being killed
            kExceptionResult=self.checkExceptionInLogs(ID,"klogs")
            if kExceptionResult['error_found']:
                exceptions=True

            #check log for normal system logs
            nExceptionResult=self.checkExceptionInLogs(ID,"logs")
            if nExceptionResult['error_found']:
                exceptions=True
            

            #decide the final failure_type
            if timeOut:
                if exceptions:
                    failure_type="Detected Hang"
                else:
                    failure_type="Silent Hang"
            elif crashed:
                if exceptions:
                    failure_type="Detected Early Exit"
                else:
                    failure_type="Silent Early Exit"
            else:
                if rightOutput:
                    if exceptions:
                        failure_type="Benign"
                    else:
                        failure_type="No effect"
                else:
                    if exceptions:
                        failure_type="Detected Data Corruption"
                    else:
                        failure_type="Silent Data Corruption"

        else:
            sExceptionResult=self.checkExceptionInLogs(ID,"logs")
            if sExceptionResult['error_found']:
                print(sExceptionResult)
        
        sql="update injection_merge_hadoop set start_time='%s', end_time='%s',last_time=%s,failure_type='%s',container_id='%s' where fault_id='%s'" % (timeResult.get("start_time",""),timeResult.get("end_time",""),timeResult.get("last_time",0),failure_type,containerID,ID)
        self.cursor.execute(sql)
        self.db.commit()

    
    def checkExceptionInLogs(self,ID,logFolder):
        resultDict={}
        resultDict['error_found']=False
        resultDict['error_file']=""
        resultDict['error_class']=""
        #template for userlogs
        log_format = '<UUID> <PID> <Date> <Time> <Level> <Class>: <Content>'
        #template for service logs
        log_format2 = '<UUID> <PID> <Date> <Time> <Level> <Unknown> <Class>: <Content>'
        #template for stderr
        log_format3='<UUID> <PID> <log>:<Level> <Content>'
        #rtemplate for audit log
        log_format4=''

        directory=self.dataDir+"/fi/"+ID+"/"+logFolder
        for dirpath,subdirs,files in os.walk(directory):
            for filename in files:
                if filename.endswith(".log"):
                    tempFile=open(os.path.join(dirpath,filename))
                    lines=tempFile.readlines()
                    tempFile.close()
                    if len(lines)!=0:
                        errorDict=self.parseALogFile(filename,dirpath,log_format)
                        if errorDict['error_found']:
                            resultDict['error_found']=True
                            resultDict["error_file"]=resultDict.get("error_file","")+errorDict["error_file"]+"\n"
                            resultDict["error_class"]=resultDict.get("error_class","")+errorDict["error_class"]+"\n"
                elif filename=="syslog" or filename.endswith(".audit"):
                    tempFile=open(os.path.join(dirpath,filename))
                    lines=tempFile.readlines()
                    tempFile.close()
                    if len(lines)!=0:
                        errorDict=self.parseALogFile(filename,dirpath,log_format)
                        if errorDict['error_found']:
                            resultDict['error_found']=True
                            resultDict["error_file"]=resultDict.get("error_file","")+errorDict["error_file"]+"\n"
                            resultDict["error_class"]=resultDict.get("error_class","")+errorDict["error_class"]+"\n"
                elif filename=="stderr":
                    tempFile=open(os.path.join(dirpath,filename))
                    lines=tempFile.readlines()
                    tempFile.close()
                    if len(lines)!=0:
                        errorDict=self.parseALogFile(filename,dirpath,log_format3)
                        if errorDict['error_found']:
                            resultDict['error_found']=True
                            resultDict["error_file"]=resultDict.get("error_file","")+errorDict["error_file"]+"\n"
                            resultDict["error_class"]=resultDict.get("error_class","")+errorDict["error_class"]+"\n" 
                elif filename=="stdout":
                    logFile=open(os.path.join(dirpath,filename))
                    lines=logFile.readlines()
                    logFile.close()
                    if len(lines)!=0:
                        resultDict['error_found']=True
                        resultDict["error_file"]=resultDict.get("error_file","")+os.path.join(dirpath,filename)+"\n"
                elif filename.endswith(".out"):
                    outFile=open(os.path.join(dirpath,filename))
                    lines=outFile.readlines()
                    outFile.close()
                    if len(lines)!=0 and not lines[0].startswith("ulimit"):
                        resultDict['error_found']=True
                        resultDict["error_file"]=resultDict.get("error_file","")+os.path.join(dirpath,filename)+"\n"
                elif filename.endswith(".dat") or filename.endswith(".csv") or filename.endswith(".sequence"):
                    #currently not used
                    fullPath=os.path.join(dirpath,filename)
                else:
                    #unexpected log file means error here
                    resultDict['error_found']=True
                    resultDict["error_file"]=resultDict.get("error_file","")+os.path.join(dirpath,filename)+"\n"
        if resultDict["error_found"]:
            sql="update injection_merge_hadoop set error_file='%s',error_class='%s' where fault_id='%s'" %(resultDict["error_file"],resultDict["error_class"].replace("'","`"),ID)
            self.cursor.execute(sql)
            self.db.commit()
        return resultDict

                    
    
    def parseALogFile(self,filename,parentFolderPath,template):
        input_dir = parentFolderPath  # The input directory of log file
        output_dir = parentFolderPath  # The output directory of parsing results

        # Regular expression list for optional preprocessing (default: [])
        regex = [
                r'@[a-z0-9]+$',
                r'[[a-z0-9\-\/]+]',
                r'{.+}',
                # r'blk_(|-)[0-9]+',  # block id
                # r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
                r'(\d+\.){3}\d+',
                r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        st = 0.5  # Similarity threshold
        depth = 4  # Depth of all leaf nodes

        parser = log_parser.LogParser(template, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
        parser.parse(filename)
        errorLogDict=self.parseLogSequence(filename,parentFolderPath)
        return errorLogDict
    
    #generate a log sequence for a log file and report error and fatal level logs
    def parseLogSequence(self,filename,parentFolderPath):
        parseResultDict={}
        parseResultDict["error_found"]=False
        templateIDSequence=[]
        processedLogFilePath=parentFolderPath+"/"+filename+"_structured.csv"
        processedLogFile=open(processedLogFilePath)
        allLines=processedLogFile.readlines()
        processedLogFile.close()

        headLine=True
        levelIndex=0
        templateIDIndex=0
        classIndex=0
        for line in csv.reader(allLines, quotechar='"', delimiter=',',quoting=csv.QUOTE_ALL, skipinitialspace=True):
            if headLine:
                indexCounter=0
                for item in line:
                    if item=="Level":
                        levelIndex=indexCounter
                    elif item=="EventId":
                        templateIDIndex=indexCounter
                    elif item=="Class":
                        classIndex=indexCounter
                    indexCounter=indexCounter+1
                headLine=False
            else:
                level=line[levelIndex]
                template_hash=line[templateIDIndex]
                className=line[classIndex]
                if level=="ERROR" or level=="FATAL":
                    if not parseResultDict['error_found']:
                        parseResultDict['error_found']=True
                        parseResultDict['error_class']=className
                        parseResultDict['error_file']=parentFolderPath+"/"+filename
                template_id=self.getTemplateID(template_hash)
                templateIDSequence.append(template_id)
        sequenceFile=open(parentFolderPath+"/"+filename+".sequence","w+")
        sequenceFile.write(" ".join(str(x) for x in templateIDSequence)+"\n")
        sequenceFile.close()
        return parseResultDict

        
        
    def getTemplateID(self,template_hash):
        sql="select id from hadoop_log_template where hash_key='%s'" % template_hash
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        if len(results)==0:
            try:
                sql2="insert into hadoop_log_template(hash_key) values ('%s')" % template_hash
                self.cursor.execute(sql2)
                self.db.commit()
            except:
                print("log key already existed! "+template_hash)
            self.cursor.execute(sql)
            results2=self.cursor.fetchall()
            return results2[0][0]
        else:
            return results[0][0]



    
    def getRunningTime(self,ID):
        resultDict={}
        timeFilePath=self.dataDir+"/fi/"+ID+"/timeCounter.txt"
        if os.path.exists(timeFilePath):
            timeFile=open(self.dataDir+"/fi/"+ID+"/timeCounter.txt")
            lines=timeFile.readlines()
            timeFile.close()
            if len(lines)==1:
                resultDict['flag']="single"
                start_time=dt.datetime.strptime(lines[0].strip(),"%Y-%m-%d_%H:%M:%S")
                resultDict['start_time']=start_time.strftime("%Y-%m-%d %H:%M:%S")

            elif len(lines)!=2:
                resultDict['flag']="no time file"
                # print(lines)
            else:
                resultDict['flag']="ok"
                start_time=dt.datetime.strptime(lines[0].strip(),"%Y-%m-%d_%H:%M:%S")
                end_time=dt.datetime.strptime(lines[1].strip(),"%Y-%m-%d_%H:%M:%S")
                time_diffrence=(end_time-start_time).total_seconds()*1000
                resultDict['start_time']=start_time.strftime("%Y-%m-%d %H:%M:%S")
                resultDict['end_time']=end_time.strftime("%Y-%m-%d %H:%M:%S")
                resultDict['last_time']=int(time_diffrence)
        else:
            resultDict['flag']="no time file"
        return resultDict


    
    def IDActivated(self,ID):
        sql="select * from injection_merge_hadoop where fault_id='%s' and activated=1" % ID
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        if len(results)==0:
            return False
        else:
            return True
        

            
    
    def close(self):
        self.cursor.close();
        self.db.close()




if __name__ == '__main__':
        processor=Processor("data",sys.argv[1])
        processor.start()
        processor.close()
        

