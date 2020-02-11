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
import extract_resource


class Processor(object):
    def __init__(self,dataDir):
        self.db = pymysql.connect("10.0.0.49", "alex", "fake_passwd", "new_injection3", charset="utf8")
        self.cursor = self.db.cursor()
        self.dataDir=dataDir
        self.outputDict={}
        self.sqlDict={}
        self.containerDict={}
    
    #extract log template
    def start_from_zero(self,prefix):
        self.prefix=prefix
        if os.path.exists(self.dataDir):
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

            fis=self.get_fi_to_process()
            
            counter=0
            for fi in fis:
                self.preprocessOneFI(fi,parser)
                counter=counter+1
                if counter % 50 == 0:
                    print("Processed: " + str(counter))
            parser.saveData()
        else:
            print("Invalid folder")
    

    def get_fi_to_process(self):
        ids=[]
        if self.prefix=='all':# process based on the directories in the dataDirFoler
            dirs=os.listdir(self.dataDir)
            ids=dirs
        elif self.prefix=='db':
            sql="select fault_id from injection_record_hadoop"
            self.cursor.execute(sql)
            results=self.cursor.fetchall()
            for result in results:
                ids.append(result[0])
        elif self.prefix=='act':
            sql="select fault_id from injection_record_hadoop where activated=1"
            self.cursor.execute(sql)
            results=self.cursor.fetchall(sql)
            for result in results:
                ids.append(result[0])
        elif len(self.prefix)==1:
            # in distributed process mode
            dirs=os.listdir(self.dataDir)
            for direc in dirs:
                if direc.lower().startswith(self.prefix):
                    ids.append(direc)
        else:
            print("Fatal! Unknown prefix! "+self.prefix)
        return ids

    #upload template
    #check failures types
    #parse resource usage for each job
    #generate log template sequence for each job
    def start(self,prefix):
        self.prefix=prefix
        if os.path.exists(self.dataDir):
            self.load_rightOutput()
            # self.load_containers()
            # self.load_resource()
            # print("Load resource done!")
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
            print("Upload templates done!")

            fis=self.get_fi_to_process()

            counter=0
            for fi in fis:
                self.processOneFI(fi,parser)
                counter=counter+1
                if counter % 50 == 0:
                    print("Processed: " + str(counter))                 
        else:
            print("Invalid folder")
   
    def start_resource(self,prefix):
        self.load_resource()
        self.load_containers()
        self.prefix=prefix
        if os.path.exists(self.dataDir):
            
            fis=self.get_fi_to_process()

            counter=0
            for fi in fis:
                containerID=self.getContainerID(fi)
                self.processOneFIResource(fi,containerID)
                counter=counter+1
                if counter % 50 == 0:
                    print("Processed: " + str(counter))                 
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
            
   
    def preprocessOneFI(self,ID,parser):
        if not os.path.exists(self.dataDir+"/"+ID):
            print("This fault doesn't exist! "+ID)
            return
        else:
            self.preprocessOneFILog(self.dataDir+"/"+ID+"/"+"logs",parser)
            self.preprocessOneFILog(self.dataDir+"/"+ID+"/"+"klogs",parser)


    def preprocessOneFILog(self,logdir,parser):
        
        #template for service logs
        log_format = '<UUID> <PID> <Date> <Time> <Level> <Class>: <Content>'
        #template for userlogs
        log_format2 = '<UUID> <PID> <Date> <Time> <Level> <Unknown> <Class>: <Content>'
        #template for stderr
        log_format3='<UUID> <PID> <Unknown>:<Level> <Content>'
        #rtemplate for audit log
        log_format4=''
        for dirpath,subdirs,files in os.walk(logdir):
            for filename in files:
                if filename.endswith(".log"):
                    parser.new_parse(dirpath+"/"+filename,log_format)

                elif filename=="syslog" or filename.endswith(".audit"):
                    #currently not security audit log found
                    parser.new_parse(dirpath+"/"+filename,log_format2)
                
                elif filename=="stderr":
                    fileHandler=open(dirpath+"/"+filename,'r')
                    content=fileHandler.readlines()
                    fileHandler.close()
                    #log4j configuration warning will be the first 3 lines
                    if len(content)!=0 and len(content)!=3 and len(content)!=6:
                        print("stderr found!")
                        print(dirpath+"/"+filename)
                        print(content)
                        # sys.exit()
                    parser.new_parse(dirpath+"/"+filename,log_format3)

                    # parser.new_parse(log_format)

                elif filename=="stdout":
                    fileHandler=open(dirpath+"/"+filename,'r')
                    content=fileHandler.readlines()
                    fileHandler.close()
                    if len(content)!=0:
                        print("stdout found!")
                        print(dirpath+"/"+filename)
                        print(content)
                    # parser.new_parse(log_format)
    
    # def processOneLogFile(self,filename, folderPath,log_format):
    
    # to parse the wordcount file
    def load_rightOutput(self):
        wordCountFile=open("data/input_data.txt",'r')
        lines=wordCountFile.readlines()
        wordCountFile.close()
        lineCounter=len(lines)
        oneLine=lines[0].strip()
        allWords=oneLine.split(" ")
        for word in allWords:
            self.outputDict[word]=lineCounter

    def load_containers(self):
        current=os.getcwd()
        all_files=os.listdir(current)
        for tmp_file in all_files:
            if tmp_file.startswith("hadoop") and tmp_file.endswith(".log"):
                containerFile=open(tmp_file,'r')
                fault_type=tmp_file.split(".")[0].split("_")[1]
                mode=tmp_file.split(".")[0].split("_")[0]
                allContainers=[]
                lines=containerFile.readlines()
                containerFile.close()
                for line in lines:
                    allContainers.append(line.strip())
                self.containerDict[mode+"_"+fault_type]=allContainers
    
     
    def load_resource(self):
        if not os.path.exists("resource.dict"):
            resource_parser=extract_resource.ResourceParser("monitor.log")
            resource_parser.parse()
        resourceFile=open("resource.dict","r")
        self.resourceDict=json.load(resourceFile)
        resourceFile.close()
    
    
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
        # modes=['random','always']
        # fault_types=["ATTRIBUTE_SHADOWED_FAULT","CONDITION_BORDER_FAULT","CONDITION_INVERSED_FAULT","EXCEPTION_SHORTCIRCUIT_FAULT",
        # "EXCEPTION_UNCAUGHT_FAULT","EXCEPTION_UNHANDLED_FAULT","NULL_FAULT","SWITCH_FALLTHROUGH_FAULT","SWITCH_MISS_DEFAULT_FAULT",
        # "UNUSED_INVOKE_REMOVED_FAULT","Value_FAULT","SYNC_FAULT"]

        # for mode in modes:
        #     for fault_type in fault_types:
        #         sql

        sql="select fault_type,activation_mode from injection_record_hadoop where fault_id='%s'" % ID
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        fault_type=results[0][0]
        activated_mode=results[0][1]

        sql2="select fault_id from injection_record_hadoop where fault_type='%s' and activation_mode='%s' order by start_time asc" % (fault_type,activated_mode)
        self.cursor.execute(sql2)
        results=self.cursor.fetchall()
        index=-1
        counter=0
        for result in results:
            if result[0]==ID:
                index=counter
                break
            counter=counter+1
        if index==-1:
            print(sql2)
            print("Error! Cannot find this fault_id in the db! "+ID)
            sys.exit()

        containerID=""
        if fault_type=="ATTRIBUTE_SHADOWED_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_shadow'][index]
            else:
                containerID=self.containerDict['random_shadow'][index]
        elif fault_type=="CONDITION_BORDER_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_border'][index]
            else:
                containerID=self.containerDict['random_border'][index]
        elif fault_type=="CONDITION_INVERSED_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_inverse'][index]
            else:
                containerID=self.containerDict['random_inverse'][index]
        elif fault_type=="EXCEPTION_SHORTCIRCUIT_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_short'][index]
            else:
                containerID=self.containerDict['random_short'][index]
        elif fault_type=="EXCEPTION_UNCAUGHT_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_uncaught'][index]
            else:
                containerID=self.containerDict['random_uncaught'][index]
        elif fault_type=="EXCEPTION_UNHANDLED_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_unhandle'][index]
            else:
                containerID=self.containerDict['random_unhandle'][index]
        elif fault_type=="NULL_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_null'][index]
            else:
                containerID=self.containerDict['random_null'][index]
        elif fault_type=="SWITCH_FALLTHROUGH_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_fallthrough'][index]
            else:
                containerID=self.containerDict['random_fallthrough'][index]
        elif fault_type=="SWITCH_MISS_DEFAULT_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_default'][index]
            else:
                containerID=self.containerDict['random_default'][index]
        elif fault_type=="UNUSED_INVOKE_REMOVED_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_remove'][index]
            else:
                containerID=self.containerDict['random_remove'][index]
        elif fault_type=="Value_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_value'][index]
            else:
                containerID=self.containerDict['random_value'][index]
        elif fault_type=="SYNC_FAULT":
            if activated_mode=='always':
                containerID=self.containerDict['always_sync'][index]
            else:
                containerID=self.containerDict['random_sync'][index]
        else:
            print("Cannot find containerID") 
        # update the container ID for each fi 
        sql="update injection_record_hadoop set container_id='%s' where fault_id='%s'" % (containerID,ID)
        # print(sql)
        self.cursor.execute(sql)
        self.db.commit()
        
        return containerID

    
    def processOneFI(self,fi,parser):
        
        ID=fi
  
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
            # if sExceptionResult['error_found']:
            #     print(sExceptionResult)
        
        sql="update injection_record_hadoop set start_time='%s', end_time='%s',last_time=%s,failure_type='%s' where fault_id='%s'" % (timeResult.get("start_time","2020-10-10 10:10:10"),timeResult.get("end_time","2020-10-10 10:10:10"),timeResult.get("last_time",0),failure_type,ID)
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except:
            print(sql)
            # sys.exit()
    
    def processOneFIResource(self,ID, container_id):
        sql="select start_time,running_time,end_time,container_id from injection_record_hadoop where fault_id='%s'" % ID
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        if len(results)!=1:
            print("Error 1:"+ID)
        else:
            # if results[0][0] is None or results[0][0]==results[0][2]:
            #     print("Error 2:"+ID)
            # else:
            allTimeData=self.resourceDict[container_id[0:12]]
            if len(allTimeData)==0:
                print("No resource usage data found for container: "+results[0][3][0:12])
            fakeTime=dt.datetime.strptime("2020-10-10 10:10:10","%Y-%m-%d %H:%M:%S")
            jobStartTime=results[0][0]
            jobEndTime=results[0][2]
            running_time=results[0][1]
            if jobStartTime==fakeTime:
                print("No start time record: "+ID)
                sql="update injection_record_hadoop set resource_bug_flag=%s where fault_id='%s'" % (1,ID)
                self.cursor.execute(sql)
                self.db.commit()
                return
                #mark this job as a job with bug
                #return
            if jobEndTime==fakeTime:
                print("No end time record: "+ID)
                jobEndTime=jobStartTime+dt.timedelta(milliseconds=running_time)

            
            dataList=[]

            for timeData in allTimeData:
                # resourceTime=dt.datetime.strptime(timeData['time'],"%Y-%m-%d %H:%M:%S")-dt.timedelta(hours=8)
                resourceTime=dt.datetime.strptime(timeData['time'],"%Y-%m-%d %H:%M:%S")
                # jobStartTime=dt.datetime.strptime(results[0][0],"%Y-%m-%d %H:%M:%S")
                jobStartTime=results[0][0]
                if resourceTime>=jobStartTime and resourceTime<=jobEndTime:
                    dataList.append(timeData)
            if len(dataList)==0:
                print("No resource data: "+ID)
                sql="update injection_record_hadoop set resource_bug_flag=%s where fault_id='%s'" % (2,ID)
                self.cursor.execute(sql)
                self.db.commit()
            resourceDataFile=open(self.dataDir+"/"+ID+"/resource.dat","w")
            json.dump(dataList,resourceDataFile)
            
            resourceDataFile.close()  

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
        resultDict['error_date']="2020-10-10"
        resultDict['error_time']="10:10:10.000"
        error_from_log=0
        

        #template for service logs
        log_format = '<UUID> <PID> <Date> <Time> <Level> <Class>: <Content>'
        #template for user logs
        log_format2 = '<UUID> <PID> <Date> <Time> <Level> <Unknown> <Class>: <Content>'
        #template for stderr
        log_format3='<UUID> <PID> <Unknown>:<Level> <Content>'
        #rtemplate for audit log
        log_format4=''

        directory=self.dataDir+'/'+ID+"/"+logFolder
        for dirpath,_,files in os.walk(directory):
            for filename in files:
                if filename.endswith(".log"):
                    tempFile=open(os.path.join(dirpath,filename),'r')
                    lines=tempFile.readlines()
                    tempFile.close()
                    if len(lines)!=0:
                        errorDict=self.parseALogFile(filename,dirpath,log_format,ID,parser)
                        if errorDict['error_found']:
                            resultDict['error_found']=True
                            if errorDict['error_log']:
                                error_from_log=1
                            resultDict["error_file"]=resultDict.get("error_file","")+errorDict["error_file"]+"\n"
                            resultDict["error_class"]=resultDict.get("error_class","")+errorDict["error_class"]+"\n"
                            resultDict["error_component"]=resultDict.get("error_component","")+errorDict["error_component"]+"\n"
                            found_time=dt.datetime.strptime(errorDict.get("error_date","2020-10-10")+"_"+errorDict.get("error_time","10:10:10,000"),"%Y-%m-%d_%H:%M:%S.%f")
                            previous_time=dt.datetime.strptime(resultDict.get("error_date","2020-10-10")+"_"+resultDict.get("error_time","10:10:10,000"),"%Y-%m-%d_%H:%M:%S.%f")
                            if found_time<previous_time:
                                resultDict['error_date']=errorDict['error_date']
                                resultDict['error_time']=errorDict['error_time']

                elif filename=="syslog" or filename.endswith(".audit"):
                    tempFile=open(os.path.join(dirpath,filename),'r')
                    lines=tempFile.readlines()
                    tempFile.close()
                    if len(lines)!=0:
                        errorDict=self.parseALogFile(filename,dirpath,log_format2,ID,parser)
                        if errorDict['error_found']:
                            resultDict['error_found']=True
                            if errorDict['error_log']:
                                error_from_log=1
                            resultDict["error_file"]=resultDict.get("error_file","")+errorDict["error_file"]+"\n"
                            resultDict["error_class"]=resultDict.get("error_class","")+errorDict["error_class"]+"\n"
                            resultDict["error_component"]=resultDict.get("error_component","")+errorDict["error_component"]+"\n"
                            found_time=dt.datetime.strptime(errorDict.get("error_date","2020-10-10")+"_"+errorDict.get("error_time","10:10:10,000"),"%Y-%m-%d_%H:%M:%S.%f")
                            previous_time=dt.datetime.strptime(resultDict.get("error_date","2020-10-10")+"_"+resultDict.get("error_time","10:10:10,000"),"%Y-%m-%d_%H:%M:%S.%f")
                            if found_time<previous_time:
                                resultDict['error_date']=errorDict['error_date']
                                resultDict['error_time']=errorDict['error_time']
                elif filename=="stderr":
                    tempFile=open(os.path.join(dirpath,filename),'r')
                    lines=tempFile.readlines()
                    tempFile.close()
                    if len(lines)!=0:
                        errorDict=self.parseALogFile(filename,dirpath,log_format3,ID,parser)
                        if errorDict['error_found']:
                            resultDict['error_found']=True
                            if errorDict['error_log']:
                                error_from_log=1
                            resultDict["error_file"]=resultDict.get("error_file","")+errorDict["error_file"]+"\n"
                            resultDict["error_class"]=resultDict.get("error_class","")+errorDict["error_class"]+"\n" 
                            resultDict["error_component"]=resultDict.get("error_component","")+errorDict["error_component"]+"\n"
                            found_time=dt.datetime.strptime(errorDict.get("error_date","2020-10-10")+"_"+errorDict.get("error_time","10:10:10,000"),"%Y-%m-%d_%H:%M:%S.%f")
                            previous_time=dt.datetime.strptime(resultDict.get("error_date","2020-10-10")+"_"+resultDict.get("error_time","10:10:10,000"),"%Y-%m-%d_%H:%M:%S.%f")
                            if found_time<previous_time:
                                resultDict['error_date']=errorDict['error_date']
                                resultDict['error_time']=errorDict['error_time']
                elif filename=="stdout":
                    logFile=open(os.path.join(dirpath,filename),'r')
                    lines=logFile.readlines()
                    logFile.close()
                    if len(lines)!=0:
                        resultDict['error_found']=True
                        error_from_log=1
                        resultDict["error_file"]=resultDict.get("error_file","")+os.path.join(dirpath,filename)+"\n"
                elif filename.endswith(".out"):
                    outFile=open(os.path.join(dirpath,filename),'r')
                    lines=outFile.readlines()
                    outFile.close()
                    if len(lines)!=0 and not lines[0].startswith("ulimit"):
                        resultDict['error_found']=True
                        resultDict["error_file"]=resultDict.get("error_file","")+os.path.join(dirpath,filename)+"\n"
                elif filename.endswith(".dat") or filename.endswith(".csv") or filename.endswith(".sequence") or filename.endswith(".sequence2"):
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
        if resultDict["error_found"]:
            error_time=dt.datetime.strptime(resultDict.get("error_date","2020-10-10")+"_"+resultDict.get("error_time","10:10:10.000"),"%Y-%m-%d_%H:%M:%S.%f")
            sql="update injection_record_hadoop set error_file='%s',error_class='%s',error_time='%s', error_log=%s, error_component='%s' where fault_id='%s'" %(resultDict["error_file"],resultDict["error_class"].replace("'","`"),error_time.strftime("%Y-%m-%d %H:%M:%S"),error_from_log,resultDict.get("error_component",""),ID)
            try:
                self.cursor.execute(sql)
                self.db.commit()
            except:
                print(sql)
        return resultDict

                    
    
    def parseALogFile(self,filename,parentFolderPath,template,ID,parser):

        parser.parse_logfile(parentFolderPath,filename,template)
        try:
            errorLogDict=self.parseLogSequence(filename,parentFolderPath,ID)
            return errorLogDict
        except:
            print(parentFolderPath+"/"+filename+"_structured.csv")
            return None
    
    #generate a log sequence for a log file and report error and fatal level logs
    def parseLogSequence(self,filename,parentFolderPath,ID):
        # self.uploadLogTemplates(filename,parentFolderPath,ID)

        parseResultDict={}
        parseResultDict["error_found"]=False
        parseResultDict["error_log"]=False
        parseResultDict["error_component"]=""
        templateIDSequence=[]
        processedLogFilePath=parentFolderPath+"/"+filename+"_structured.csv"
        processedLogFile=open(processedLogFilePath,'r')
        allLines=processedLogFile.readlines()
        processedLogFile.close()

        headLine=True
        levelIndex=0
        templateIDIndex=0
        UUIDIndex=0
        classIndex=0
        dateIndex=0
        timeIndex=0
        component_name=self.getComponentName(filename)
        sequenceFile=open(parentFolderPath+"/"+filename+".sequence","w+")
        for line in csv.reader((tmp_line.replace('\0','') for tmp_line in allLines), quotechar='"', delimiter=',',quoting=csv.QUOTE_ALL, skipinitialspace=True):
            if headLine:
                indexCounter=0
                for item in line:
                    if item=="Level":
                        levelIndex=indexCounter
                    elif item=="Tmpl_HASH":
                        templateIDIndex=indexCounter
                    elif item=="UUID":
                        UUIDIndex=indexCounter
                    elif item=="Class":
                        classIndex=indexCounter
                    elif item=="Date":
                        dateIndex=indexCounter
                    elif item=="Time":
                        timeIndex=indexCounter
                    indexCounter=indexCounter+1
                headLine=False
            else:
                level=line[levelIndex]
                template_hash=line[templateIDIndex]
                className=line[classIndex].replace("'",'"')
                UUID=line[UUIDIndex]
                
                if level=="ERROR" or level=="FATAL" or level not in ['INFO','WARN','DEBUG']:
                    if not parseResultDict['error_found']:
                        parseResultDict['error_found']=True
                        if level=="ERROR" or level=='FATAL':
                            parseResultDict['error_log']=True
                        parseResultDict['error_class']=className
                        parseResultDict['error_file']=parentFolderPath+"/"+filename
                        parseResultDict['error_date']=line[dateIndex]
                        parseResultDict['error_time']=line[timeIndex]
                        parseResultDict['error_component']=component_name
                        try:
                            dt.datetime.strptime(parseResultDict.get("error_date","2020-10-10")+"_"+parseResultDict.get("error_time","10:10:10.000"),"%Y-%m-%d_%H:%M:%S.%f")
                        except:
                            parseResultDict['error_date']="2020-10-10"
                            parseResultDict['error_time']="10:10:10.000"
                            # print("Error time info!"+ID+"-"+parentFolderPath+"/"+filename)
                            # print(parseResultDict.get("error_date","2020-10-10")+" "+parseResultDict.get("error_time","10:10:10.000"))
                template_id=self.getTemplateID(template_hash)
                if template_id==-1:
                    print(processedLogFilePath)
                    print(line[templateIDIndex])
                    sys.exit()
                templateIDSequence.append(template_id)
                # get the tmpl ID from database
                if level in ['INFO','WARN','DEBUG','ERROR','FATAL']:
                    sequenceFile.write(str(template_id)+" "+UUID+" "+className+" "+line[dateIndex]+"_"+line[timeIndex]+" "+component_name+"\n")        
        sequenceFile2=open(parentFolderPath+"/"+filename+".sequence2","w+")
        sequenceFile2.write(" ".join(str(x) for x in templateIDSequence)+"\n")
        sequenceFile2.close()
        sequenceFile.close()
        return parseResultDict
    
    def getComponentName(self,filename):
        componentName=""
        if filename.endswith(".log"):
            if filename.find("secondarynamenode")!=-1:
                componentName="secondary"
            elif filename.find("namenode")!=-1:
                componentName="namenode"
            elif filename.find("datanode")!=-1:
                componentName="datanode"
            elif filename.find("resourcemanager")!=-1:
                componentName='resourcemanager'
            elif filename.find("nodemanager"):
                componentName="nodemanager"
        elif filename=='stderr' or filename=='stdout' or filename=='syslog':
                componentName="mapreduce"
        else:
            componentName=""
        return componentName
    
    #get all the log cluster in the log Tree and upload each cluster
    def uploadLogTemplates(self,parser):
        allClusters=parser.getAllCluster()
        for cluster in allClusters:
            template_str=cluster['tmpl']
            template_id =cluster['hash']
            template_file=cluster['source_file'].replace("'",'"')
            sql="insert into hadoop_log_template(hash_key,tmpl_content,source_file,level,length,token) values('%s','%s','%s','%s',%s,'%s')" % (template_id,template_str.replace("'",'"'),template_file,cluster['level'],cluster['length'],cluster['token'])

            try:
                self.cursor.execute(sql)
                self.db.commit()
            except:
                pass
                # print("duplicate log templates")
                # print(template_id)
                # print(template_str)
        
        
    def getTemplateID(self,template_hash):
        sql="select id from hadoop_log_template where hash_key='%s'" % template_hash
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        if len(results)==0:
            print("log template doesn't exist!"+template_hash)
            return -1
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
        
        # processor.start_from_zero(sys.argv[2])

        # processor.start(sys.argv[2])

        #this method should be merged into start
        # processor.start_remain()
        #this method should be merged into processOneFI
        processor.start_resource(sys.argv[2])
        processor.close()
        

