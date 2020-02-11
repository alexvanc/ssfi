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
import pickle
import random


#this script is used to generate all kind of datasets

class Generator(object):
    def __init__(self,dataDir):
        self.db = pymysql.connect("39.99.169.20", "alex", "fake_passwd", "new_injection5", charset="utf8")
        self.cursor = self.db.cursor()
        self.dataDir=dataDir
        self.trainSetDict={}
        self.allNormalIns=[]
    
    def start(self):
        self.generateAllNomal()
        self.generateAllAbnormal()
        self.trainSetDict['training']=self.allNormalIns
        self.moveFiles()
        allDataSet=open("/tmp/dataset.dat",'wb')
        pickle.dump(self.trainSetDict,allDataSet)
        allDataSet.close()

    
    def moveFiles(self):
        for key in self.trainSetDict:
            if key=='training':
                print(key+" "+str(len(self.trainSetDict[key])))
                for fault_id in self.trainSetDict[key]:
                    os.system("cp -r "+self.dataDir+"/all/"+fault_id+" "+self.dataDir+"/+all_training/")
            else:
                dest_dir=""
                if key=='Detected Benign':
                    dest_dir="all_DBE"
                elif key=='Silent Benign':
                    dest_dir="all_SBE"
                elif key=='Detected Early Exit':
                    dest_dir="all_DEE"
                elif key=='Silent Early Exit':
                    dest_dir="all_SEE"
                elif key=='Detected Hang':
                    dest_dir="all_DHANG"
                elif key=='Silent Hang':
                    dest_dir="all_SHANG"
                elif key=='Silent Data Corruption':
                    dest_dir="all_SDC"
                else:
                    print("Fatal Error!"+key)
                    return
                for fault_id in self.trainSetDict[key]['normal']:
                    os.system("cp -r "+self.dataDir+"/all/"+fault_id+" "+self.dataDir+"/"+dest_dir+"/normal/")
                for fault_id in self.trainSetDict[key]['anomaly']:
                    os.system("cp -r "+self.dataDir+"/all/"+fault_id+" "+self.dataDir+"/"+dest_dir+"/anomaly/")
                print(key+" normal "+str(len(self.trainSetDict[key]["normal"])))
                print(key+" anomaly "+str(len(self.trainSetDict[key]["anomaly"])))


    
    def generateAllAbnormal(self):
        sql="select fault_id,failure_type from injection_record_hadoop where with_bug=0 and resource_bug_flag=0 and failure_type is not null and failure_type<>''"
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            fault_id=result[0]
            f_type=result[1]
            self.generate_one_type(fault_id,f_type)
    
    def generateAllNomal(self):
        sql="select fault_id from injection_record_hadoop where with_bug=0 and resource_bug_flag=0 and activated=0 and failure_type=''"
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            self.allNormalIns.append(result[0])
    
    def generate_one_type(self,fault_id,failure_type):
        aNormalIns=self.getRandomNormalIns()
        if failure_type in self.trainSetDict:
            self.trainSetDict[failure_type]['anomaly'].append(fault_id)
            self.trainSetDict[failure_type]['normal'].append(aNormalIns)
        else:
            self.trainSetDict[failure_type]={'anomaly':[fault_id]}
            self.trainSetDict[failure_type]['normal']=[aNormalIns]
    
    def getRandomNormalIns(self):
        length=len(self.allNormalIns)
        index=random.randint(0,length-1)
        fault_id=self.allNormalIns[index]
        self.allNormalIns.remove(fault_id)
        return fault_id


    
    def close(self):
        self.cursor.close()
        self.db.close()

if __name__ == '__main__':
    generator=Generator(sys.argv[1])
    generator.start()
    generator.close()

        

