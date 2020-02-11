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


class Processor(object):
    def __init__(self,dataDir):
        self.db = pymysql.connect("39.99.169.20", "alex", "fake_passwd", "new_injection5", charset="utf8")
        self.cursor = self.db.cursor()
        self.dataDir=dataDir
        self.outputDict={}
        self.sqlDict={}
        self.containerDict={}

    #upload template
    #check failures types
    #parse resource usage for each job
    #generate log template sequence for each job
    def start(self,prefix):
        split_time=datetime.strptime("2019-12-17 13:00:00","%Y-%m-%d %H:%M:%S")
        self.prefix=prefix
        group1=0
        group2=0
        dirs=os.listdir(self.dataDir)
        for fi in dirs:
            ctime=os.path.(self.dataDir+"/"+fi)
            create_time=datetime.fromtimestamp(ctime)
            
            if create_time>split_time:
                group2=group2+1
            else:
                group1=group1+1
        print("For group2: "+str(group2))
        print("For group1: "+str(group1))

    def close(self):
        self.cursor.close()
        self.db.close()


            






if __name__ == '__main__':
        processor=Processor(sys.argv[1])

        processor.start(sys.argv[1])
        processor.close()
        

