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
import random
import pickle


#this script is used to mark the FI runs witn our SSFI bugs


class Processor(object):
    def __init__(self,dataDir):
        self.db = pymysql.connect("39.99.169.20", "alex", "fake_passwd", "new_injection3", charset="utf8")
        self.cursor = self.db.cursor()
        self.dataDir=dataDir
        self.outputDict={}
        self.sqlDict={}
        self.containerDict={}
    
    def start(self):
        sql="select fault_id,error_file,activated from injection_record_spark where error_file is not null"
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            self.markBug(result[0],result[1],result[2])
    
    def markBug(self,ID,error_files,activated):
        if os.path.exists(self.dataDir+"/"+ID):
            file_list=error_files.strip().split("\n")
            for error_file in file_list:
                bug_type=self.checkBug(error_file,activated)
                if bug_type!=0:
                    sql="update injection_record_spark set with_bug=%s where fault_id='%s'" % (bug_type,ID)
                    self.cursor.execute(sql)
                    self.db.commit()
                    break

    
    def checkBug(self,absFilePath,activated):
        errorFile=open(absFilePath,"r")
        content=errorFile.read()
        if content.find('VerifyError')!=-1:
            return 1
        elif activated==0 and content.find('<init> signature')!=-1:
            return 2
        elif activated==0:
            return 3
        else:
            return 0

    
    def close(self):
        self.cursor.close()
        self.db.close()


if __name__ == '__main__':
    processor=Processor(sys.argv[1])
    processor.start()
    processor.close()