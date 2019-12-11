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
import new_log_parser
import json


class Processor(object):
    def __init__(self,dataDir):
        self.db = pymysql.connect("39.99.169.20", "alex", "test1234", "new_injection3", charset="utf8")
        self.cursor = self.db.cursor()
        self.dataDir=dataDir
        self.outputDict={}
        self.sqlDict={}
        self.containerDict={}
    
    def start(self,prefix):
        self.prefix=prefix
        if os.path.exists(self.dataDir):
            self.load_rightOutput()
            # self.load_resource()
        input_dir = self.dataDir  # The input directory of log file
        output_dir = self.dataDir   # The output directory of parsing results

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

        parser = new_log_parser.LogParser(log_format=None, indir=None, outdir=None, depth=depth, st=st, rex=regex)

        if self.prefix=='act':
            #only process all activated faults
            sql="select fault_id from injection_record_spark where activated=1"
            self.cursor.execute(sql)
            results=self.cursor.fetchall()
            for result in results:
                self.processOneFI(result[0],parser)

        else:
            dirs=os.listdir(self.dataDir)
            for fi in dirs:
                self.processOneFI(fi,parser)
    
    # to parse the wordcount file
    def load_rightOutput(self):
        wordCountFile=open("input_data.txt",'r')
        lines=wordCountFile.readlines()
        wordCountFile.close()
        lineCounter=len(lines)
        oneLine=lines[0].strip()
        allWords=oneLine.split(" ")
        for word in allWords:
            self.outputDict[word]=lineCounter

    
    def processOneFI(self,fi,parser):
        if self.prefix!='act' and self.prefix!='all' and not fi.lower().startswith(self.prefix):
            return
        ID=fi
        containerID=""

        if not os.path.exists(self.dataDir+"/"+ID):
            return

        print("start: "+ID)
        # from log drain bugs
        if ID=='HpEmurOQdMkrdxYnhRXX' or ID=='TWiGcylXnUXeJlKJdopq':
            return
        running_time=self.getRunningTime(ID)
        failure_type=""
        if self.IDActivated(ID):
            crashed=False
            rightOutput=False
            exceptions=False
            timeOut=False
            #check whether all hadoop services started
            if os.path.exists(self.dataDir+'/'+ID+'/'+'startresult.log'):
                outputFile=open(self.dataDir+'/'+ID+'/'+'startresult.log','r')
                lines=outputFile.readlines()
                outputFile.close()
                if len(lines)!=7:
                    crashed=True
            else:
                crashed=True
            #check whether running result is right
            if os.path.exists(self.dataDir+'/'+ID+'/'+'runResult.txt'):
                resultFile=open(self.dataDir+'/'+ID+'/'+'runResult.txt','r')
                lines=resultFile.readlines()
                self.sqlDict['running_output']="".join(lines)
                resultFile.close()
                if len(lines)==0:
                    crashed=True
                else:
                    try:
                        rightOutput=self.checkRightOutput(lines)
                    except:
                        print("Invalid output: "+ID)
            else:
                crashed=True
            
            if running_time>100000:
                timeOut=True
            
            #check log for system being killed
            kExceptionResult=self.checkExceptionInLogs(ID,"klogs",parser)
            if kExceptionResult['error_found']:
                exceptions=True

            #check log for normal system logs
            nExceptionResult=self.checkExceptionInLogs(ID,"logs",parser)
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
                        failure_type="Detected Benign"
                    else:
                        failure_type="Silent Benign"
                else:
                    if exceptions:
                        failure_type="Detected Data Corruption"
                    else:
                        failure_type="Silent Data Corruption"

        else:
            sExceptionResult=self.checkExceptionInLogs(ID,"logs",parser)
            if sExceptionResult['error_found']:
                print(sExceptionResult)
        
        sql="update injection_record_spark set failure_type='%s' where fault_id='%s'" % (failure_type,ID)
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except:
            print(sql)
            # sys.exit()
        print("stop: "+ID)

    def checkRightOutput(self, lines):
        resultDict={}
        for line in lines:
            kvp=line.strip().split(":")
            if len(kvp)!=2:
                print(line)
                break
            else:
                resultDict[kvp[0]]=int(kvp[1].strip())
        
        allKeys=self.outputDict.keys()
        for key in allKeys:
            if key in resultDict and resultDict[key]==self.outputDict[key]:
                continue
            else:
                return False
        return True

    
    def checkExceptionInLogs(self,ID,logFolder,parser):
        resultDict={}
        resultDict['error_found']=False
        resultDict['error_file']=""
        resultDict['error_class']=""
        #template for service logs
        log_format = '<Date> <Time> <Level> <Class>: <Content>'
        #template for user logs
        log_format2 = '<Date> <Time> <Level> <Unknown> <Class>: <Content>'
        #template for stderr
        log_format3=' <log>:<Level> <Content>'
        #rtemplate for audit log
        log_format4=''

        directory=self.dataDir+'/'+ID+"/"+logFolder
        for dirpath,subdirs,files in os.walk(directory):
            for filename in files:
                
                if filename=="syslog" or filename.endswith(".audit") or filename=="spark.log":
                    tempFile=open(os.path.join(dirpath,filename),'r')
                    lines=tempFile.readlines()
                    tempFile.close()
                    if len(lines)!=0:
                        errorDict=self.parseALogFile(filename,dirpath,log_format2,ID,parser)
                        if errorDict['error_found']:
                            resultDict['error_found']=True
                            resultDict["error_file"]=resultDict.get("error_file","")+errorDict["error_file"]+"\n"
                            resultDict["error_class"]=resultDict.get("error_class","")+errorDict["error_class"]+"\n"
                elif filename.endswith(".log"):
                    tempFile=open(os.path.join(dirpath,filename),'r')
                    lines=tempFile.readlines()
                    tempFile.close()
                    if len(lines)!=0:
                        errorDict=self.parseALogFile(filename,dirpath,log_format,ID,parser)
                        if errorDict['error_found']:
                            resultDict['error_found']=True
                            resultDict["error_file"]=resultDict.get("error_file","")+errorDict["error_file"]+"\n"
                            resultDict["error_class"]=resultDict.get("error_class","")+errorDict["error_class"]+"\n"
                elif filename=="stderr":
                    tempFile=open(os.path.join(dirpath,filename),'r')
                    lines=tempFile.readlines()
                    tempFile.close()
                    if len(lines)!=0:
                        errorDict=self.parseALogFile(filename,dirpath,log_format3,ID,parser)
                        if errorDict['error_found']:
                            resultDict['error_found']=True
                            resultDict["error_file"]=resultDict.get("error_file","")+errorDict["error_file"]+"\n"
                            resultDict["error_class"]=resultDict.get("error_class","")+errorDict["error_class"]+"\n" 
                elif filename=="stdout":
                    logFile=open(os.path.join(dirpath,filename),'r')
                    lines=logFile.readlines()
                    logFile.close()
                    if len(lines)!=0:
                        resultDict['error_found']=True
                        resultDict["error_file"]=resultDict.get("error_file","")+os.path.join(dirpath,filename)+"\n"
                elif filename.endswith(".out"):
                    outFile=open(os.path.join(dirpath,filename),'r')
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
        # here we only check runError.txt, for outputError.txt, we have to confirm whether it's caused by reading result
        if os.path.exists(self.dataDir+'/'+ID+'/'+'runError.txt'):
                outputFile=open(self.dataDir+'/'+ID+'/'+'runError.txt','r')
                lines=outputFile.readlines()
                outputFile.close()
                if len(lines)!=0:
                    resultDict['error_found']=True
                    resultDict["error_file"]=resultDict.get("error_file","")+self.dataDir+'/'+ID+'/'+'runError.txt'+"\n"
                    resultDict['error_class']='null'
                else:
                    crashed=True
        if resultDict["error_found"]:
            sql="update injection_record_spark set error_file='%s',error_class='%s' where fault_id='%s'" %(resultDict["error_file"],resultDict["error_class"].replace("'","`"),ID)
            try:
                self.cursor.execute(sql)
                self.db.commit()
            except:
                print(sql)
        return resultDict

                    
    
    def parseALogFile(self,filename,parentFolderPath,template,ID,parser):
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

        parser.parse_logfile(parentFolderPath,filename,template)
        errorLogDict=self.parseLogSequence(filename,parentFolderPath,ID)
        return errorLogDict
    
        #generate a log sequence for a log file and report error and fatal level logs
    def parseLogSequence(self,filename,parentFolderPath,ID):
        # self.uploadLogTemplates(filename,parentFolderPath,ID)

        parseResultDict={}
        parseResultDict["error_found"]=False
        processedLogFilePath=parentFolderPath+"/"+filename+"_structured.csv"
        processedLogFile=open(processedLogFilePath,'r')
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
                    elif item=="Tmpl_HASH":
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
                        parseResultDict['error_class']=className.replace("'",'"')
                        parseResultDict['error_file']=parentFolderPath+"/"+filename
        return parseResultDict
    
    def getRunningTime(self,ID):

        sql="select running_time from injection_record_spark where fault_id='%s'" % ID
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        if results[0][0] is None:
            print("Running_time error1!")
        else:
            return results[0][0]
        

    
    def IDActivated(self,ID):
        sql="select * from injection_record_spark where fault_id='%s' and activated=1" % ID
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        if len(results)==0:
            return False
        else:
            return True
        

            
    
    def close(self):
        self.cursor.close()
        self.db.close()




if __name__ == '__main__':
        processor=Processor(sys.argv[1])
        processor.start(sys.argv[2])
        processor.close()
        

