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
    
    def start_from_zero(self):
        if os.path.exists(self.dataDir):
            self.load_rightOutput()
            self.load_containers()
            # self.load_resource()
            #iterate all the logFiles to train a log template extraction model
            self.trainLogTree()
        else:
            print("Invalid folder")

    def start_remain(self):
        self.prefix='all'
        self.load_rightOutput()
        self.load_containers()
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
        parser.loadData()
        
        sql="select fault_id from injection_record_hadoop where failure_type is null"
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            self.processOneFI(result[0],parser)
    
    def start_resource(self):
        self.load_resource()
        dirs=os.listdir(self.dataDir)
        for fi in dirs:
            self.processOneFIResource(fi)
        
    
    def load_resource(self):
        resourceFile=open("resource.dict","r")
        self.resourceDict=json.load(resourceFile)
        resourceFile.close()
    
    def processOneFIResource(self,ID):
        sql="select start_time,running_time,end_time,container_id from injection_record_hadoop where fault_id='%s'" % ID
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        if len(results)!=1:
            print("Error 1:"+ID)
        else:
            if results[0][0] is None or results[0][0]==results[0][2]:
                print("Error 2:"+ID)
            else:
                allTimeData=self.resourceDict[results[0][3][0:12]]
                running_time=results[0][1]
                dataList=[]
                jobStartTime=results[0][0]
                jobEndTime=jobStartTime+dt.timedelta(milliseconds=running_time)
                if ID=='AjoeHkiaaDaliGCDReUc':
                    # print(allTimeData)
                    print(jobStartTime)
                    print(jobEndTime)
                for timeData in allTimeData:
                    resourceTime=dt.datetime.strptime(timeData['time'],"%Y-%m-%d %H:%M:%S")
                    # jobStartTime=dt.datetime.strptime(results[0][0],"%Y-%m-%d %H:%M:%S")
                    if ID=='AjoeHkiaaDaliGCDReUc':
                        print(resourceTime)
                    if resourceTime>=jobStartTime and resourceTime<=jobEndTime:
                        dataList.append(timeData)
                resourceDataFile=open(self.dataDir+"/"+ID+"/resource.dat","w")
                json.dump(dataList,resourceDataFile)
                
                resourceDataFile.close()                        
    
    
    def start(self,prefix):
        self.prefix=prefix
        if os.path.exists(self.dataDir):
            self.load_rightOutput()
            self.load_containers()
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
        parser.loadData()
        #upload all the template in the log tree
        self.uploadLogTemplates(parser)
        if self.prefix=='act':
            #only process all activated faults
            sql="select fault_id from injection_record_hadoop where activated=1"
            self.cursor.execute(sql)
            results=self.cursor.fetchall()
            for result in results:
                self.processOneFI(result[0],parser)

        else:
            dirs=os.listdir(self.dataDir)
            for fi in dirs:
                self.processOneFI(fi,parser)


    
    def trainLogTree(self):
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
        parser.loadData()
        dirs=os.listdir(self.dataDir)
        counter=0
        for fi in dirs:
            self.iterateOneFIForLog(fi,parser)
            counter=counter+1
            if counter % 50 == 0:
                print("Processed: " + str(counter))
        parser.saveData()
        #TO DO: save the built template tree model
    
    def iterateOneFIForLog(self,ID,parser):
        #template for service logs
        log_format = '<UUID> <PID> <Date> <Time> <Level> <Class>: <Content>'
        #template for userlogs
        log_format2 = '<UUID> <PID> <Date> <Time> <Level> <Unknown> <Class>: <Content>'
        #template for stderr
        log_format3='<UUID> <PID> <log>:<Level> <Content>'
        #rtemplate for audit log
        log_format4=''
        for dirpath,subdirs,files in os.walk(self.dataDir+"/"+ID):
            for filename in files:
                if filename.endswith(".log"):
                    parser.new_parse(dirpath+"/"+filename,log_format)

                elif filename=="syslog" or filename.endswith(".audit"):
                    parser.new_parse(dirpath+"/"+filename,log_format2)
                
                elif filename=="stderr":
                    parser.new_parse(dirpath+"/"+filename,log_format3)

                    # parser.new_parse(log_format)

                elif filename=="stdout":
                    fileHandler=open(dirpath+"/"+filename,'r')
                    content=fileHandler.readlines()
                    fileHandler.close()
                    if len(content)!=0:
                        print(dirpath+"/"+filename)
                        print(content)
                    # parser.new_parse(log_format)
    
    # def processOneLogFile(self,filename, folderPath,log_format):
    
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

    def load_containers(self):
        containerFile=open("container.log",'r')
        lines=containerFile.readlines()
        containerFile.close()
        lineCounter=0
        for line in lines:
            if lineCounter%2==0:
                container_name=line.strip()
                container_id=lines[lineCounter+1].strip()
                self.containerDict[container_name]=container_id
            lineCounter=lineCounter+1
        # self.upload_allContainers()
    
    def upload_allContainers(self):
        allKeys=self.containerDict.keys()
        for key in allKeys:
            fault_type=key.split('_')[1]
            activation_mode=key.split('_')[0]
            full_name=self.containerDict[key]
            short_name=full_name[0:12]
            sql="insert into hadoop_containers(fault_type,activation_mode,full_name,short_name) values ('%s','%s','%s','%s')" % (fault_type,activation_mode,full_name,short_name)
            try:
                self.cursor.execute(sql)
                self.db.commit()
            except:
                continue


        

    def getContainerID(self,ID):
        sql="select fault_type,activation_mode from injection_record_hadoop where fault_id='%s'" % ID
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        fault_type=results[0][0]
        activated_mode=results[0][1]
        containerID=""
        if fault_type=="ATTRIBUTE_SHADOWED_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_shadow']
            else:
                containerID=self.containerDict['random_shadow']
        elif fault_type=="CONDITION_BORDER_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_border']
            else:
                containerID=self.containerDict['random_border']
        elif fault_type=="CONDITION_INVERSED_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_inverse']
            else:
                containerID=self.containerDict['random_inverse']
        elif fault_type=="EXCEPTION_SHORTCIRCUIT_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_short']
            else:
                containerID=self.containerDict['random_short']
        elif fault_type=="EXCEPTION_UNCAUGHT_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_uncaught']
            else:
                containerID=self.containerDict['random_uncaught']
        elif fault_type=="EXCEPTION_UNHANDLED_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_unhandle']
            else:
                containerID=self.containerDict['random_unhandle']
        elif fault_type=="NULL_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_null']
            else:
                containerID=self.containerDict['random_null']
        elif fault_type=="SWITCH_FALLTHROUGH_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_fallthrough']
            else:
                containerID=self.containerDict['random_fallthrough']
        elif fault_type=="SWITCH_MISS_DEFAULT_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_default']
            else:
                containerID=self.containerDict['random_default']
        elif fault_type=="UNUSED_INVOKE_REMOVED_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_remove']
            else:
                containerID=self.containerDict['random_remove']
        elif fault_type=="Value_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_value']
            else:
                containerID=self.containerDict['random_value']
        else:
            print("Cannot find containerID") 
        #temporary process
        sql="update injection_record_hadoop set container_id='%s' where fault_id='%s'" % (containerID,ID)
        self.cursor.execute(sql)
        self.db.commit()
        
        return containerID

    
    def processOneFI(self,fi,parser):
        if self.prefix!='act' and self.prefix!='all' and not fi.lower().startswith(self.prefix):
            return
        ID=fi
        containerID=""
        try:
            containerID=self.getContainerID(ID)
        except:
            print("Cannot find container for fault: "+ID)
            return
        if not os.path.exists(self.dataDir+"/"+ID):
            return
        
        timeResult=self.getRunningTime(ID)
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
                    rightOutput=self.checkRightOutput(lines)
            else:
                crashed=True
            
            if timeResult['running_time']>100000:
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
        
        sql="update injection_record_hadoop set start_time='%s', end_time='%s',last_time=%s,failure_type='%s',container_id='%s' where fault_id='%s'" % (timeResult.get("start_time",""),timeResult.get("end_time","2020-10-10 10:10:10"),timeResult.get("last_time",0),failure_type,containerID,ID)
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except:
            print(sql)
            # sys.exit()

    def checkRightOutput(self, lines):
        resultDict={}
        for line in lines:
            kvp=line.strip().split("\t")
            if len(kvp)!=2:
                print(line)
                break
            else:
                resultDict[kvp[0]]=int(kvp[1])
                
        allKeys=resultDict.keys()
        for key in allKeys:
            if key in self.outputDict and resultDict[key]==self.outputDict[key]:
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
        log_format = '<UUID> <PID> <Date> <Time> <Level> <Class>: <Content>'
        #template for user logs
        log_format2 = '<UUID> <PID> <Date> <Time> <Level> <Unknown> <Class>: <Content>'
        #template for stderr
        log_format3='<UUID> <PID> <log>:<Level> <Content>'
        #rtemplate for audit log
        log_format4=''

        directory=self.dataDir+'/'+ID+"/"+logFolder
        for dirpath,subdirs,files in os.walk(directory):
            for filename in files:
                if filename.endswith(".log"):
                    tempFile=open(os.path.join(dirpath,filename),'r')
                    lines=tempFile.readlines()
                    tempFile.close()
                    if len(lines)!=0:
                        errorDict=self.parseALogFile(filename,dirpath,log_format,ID,parser)
                        if errorDict['error_found']:
                            resultDict['error_found']=True
                            resultDict["error_file"]=resultDict.get("error_file","")+errorDict["error_file"]+"\n"
                            resultDict["error_class"]=resultDict.get("error_class","")+errorDict["error_class"]+"\n"
                elif filename=="syslog" or filename.endswith(".audit"):
                    tempFile=open(os.path.join(dirpath,filename),'r')
                    lines=tempFile.readlines()
                    tempFile.close()
                    if len(lines)!=0:
                        errorDict=self.parseALogFile(filename,dirpath,log_format2,ID,parser)
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
                if len(lines)!=3:
                    resultDict['error_found']=True
                    resultDict["error_file"]=resultDict.get("error_file","")+self.dataDir+'/'+ID+'/'+'runError.txt'+"\n"
                    resultDict['error_class']='null'
                else:
                    crashed=True
        if resultDict["error_found"]:
            sql="update injection_record_hadoop set error_file='%s',error_class='%s' where fault_id='%s'" %(resultDict["error_file"],resultDict["error_class"].replace("'","`"),ID)
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
        templateIDSequence=[]
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
                # get the tmpl ID from database
                template_id=self.getTemplateID(template_hash)
                templateIDSequence.append(template_id)
        sequenceFile=open(parentFolderPath+"/"+filename+".sequence","w+")
        sequenceFile.write(" ".join(str(x) for x in templateIDSequence)+"\n")
        sequenceFile.close()
        return parseResultDict
    
    #get all the log cluster in the log Tree and upload each cluster
    def uploadLogTemplates(self,parser):
        allClusters=parser.getAllCluster()
        for cluster in allClusters:
            template_str=cluster['tmpl']
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            template_file=cluster['source_file'].replace("'",'"')
            sql="insert into hadoop_log_template(hash_key,tmpl_content,source_file,level,length,token) values('%s','%s','%s','%s',%s,'%s')" % (template_id,template_str.replace("'",'"'),template_file,cluster['level'],cluster['length'],cluster['token'])

            try:
                self.cursor.execute(sql)
                self.db.commit()
            except:
                print(template_id)
                print(template_str)
        

        


        
        
    def getTemplateID(self,template_hash):
        sql="select id from hadoop_log_template where hash_key='%s'" % template_hash
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        if len(results)==0:
            print("log template doesn't exist!")
        else:
            return results[0][0]



    
    def getRunningTime(self,ID):
        resultDict={}
        timeFilePath=self.dataDir+'/'+ID+"/timeCounter.txt"
        if os.path.exists(timeFilePath):
            timeFile=open(self.dataDir+'/'+ID+"/timeCounter.txt")
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
        sql="select running_time from injection_record_hadoop where fault_id='%s'" % ID
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        if len(results)==0:
            resultDict['running_time']=-1
        else:
            if results[0][0] is None:
                resultDict['running_time']=-1
            else:
                resultDict['running_time']=int(results[0][0])
        
        return resultDict


    
    def IDActivated(self,ID):
        sql="select * from injection_record_hadoop where fault_id='%s' and activated=1" % ID
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
        # processor.start(sys.argv[2])
        # processor.start_from_zero()

        #this method should be merged into start
        # processor.start_remain()
        #this method should be merged into processOneFI
        processor.start_resource()
        processor.close()
        

