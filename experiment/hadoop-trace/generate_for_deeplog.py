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

class Generator(object):
    def __init__(self,dataDir):
        self.db = pymysql.connect("10.0.0.3", "root", "Finelab123...", "injection_merge", charset="utf8")
        self.cursor = self.db.cursor()
        self.dataDir=dataDir
        self.sqlDict={}
        self.containerDict={}
    

    def generateForAll(self):
        dirs=os.listdir(self.dataDir+"/fi")
        for fi in dirs:
            self.generateForOneRequest(fi)


    

    def generateForOneRequest(self,ID):
        requestDataFile=open(self.dataDir+"/fi/"+ID+"/"+ID+".dat","w+")
        directory=self.dataDir+"/fi/"+ID
        for dirpath,subdirs,files in os.walk(directory):
            for filename in files:
                if filename.endswith(".sequence"):
                    sequenceFile=open(os.path.join(dirpath,filename))
                    oneSequence=sequenceFile.read()
                    sequenceFile.close()
                    requestDataFile.write(oneSequence)

        requestDataFile.close()
    
    def generateTrainData(self,trainNumber):
        sql="select fault_id from injection_merge_hadoop where failure_type=''"
        trainDataFile=open("data/train_"+str(trainNumber),"w+")
        trainDataRecordFile=open("data/train_"+str(trainNumber)+"_record","w+")
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        targets=results[0:trainNumber]
        for target in targets:
            fault_id=target[0]
            oneRequestFile=open(self.dataDir+"/"+fault_id+".dat")
            content=oneRequestFile.read()
            oneRequestFile.close()
            trainDataFile.write(content)
            trainDataRecordFile.write(fault_id+"\n")
        trainDataFile.close()
        trainDataRecordFile.close()
    
    def generateTrainData2(self,trainNumber):
        sql="select fault_id from injection_merge_hadoop where failure_type=''"
        trainDataFile=open("data/train_"+str(trainNumber),"w+")
        trainDataRecordFile=open("data/train_"+str(trainNumber)+"_record","w+")
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        targets=results[0:trainNumber]
        for target in targets:
            fault_id=target[0]
            oneRequestFile=open(self.dataDir+"/"+fault_id+".dat")
            lines=oneRequestFile.readlines()
            for line in lines:
                allIDs=line.split(" ")
                oneLine=" ".join(str(int(x)-1) for x in allIDs)
                trainDataFile.write(oneLine+"\n")
            oneRequestFile.close()
            trainDataRecordFile.write(fault_id+"\n")
        trainDataFile.close()
        trainDataRecordFile.close()
        
    
    def close(self):
        self.cursor.close()
        self.db.close()


if __name__ == '__main__':
        generator=Generator("for_deep")
        generator.generateTrainData2(sys.argv[1])
        # generator.generateForAll()
        generator.close()
        

