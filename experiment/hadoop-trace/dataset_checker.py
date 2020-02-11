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
        self.db = pymysql.connect("127.0.0.1", "alex", "fake_passwd", "new_injection6", charset="utf8")
        self.cursor = self.db.cursor()
        self.dataDir=dataDir



    def check(self):
        dirs=os.listdir(self.dataDir)
        counter=0
        anomaly_counter=0
        normal_counter=0
        for fi in dirs:
            
            failure_type=self.get_failure_type(fi)
            if failure_type is None or failure_type=="":
                normal_counter=normal_counter+1
            
            if failure_type is not None and failure_type!="":
                anomaly_counter=anomaly_counter+1
            counter=counter+1

        print("anomaly:{}".format(anomaly_counter))
        print("normal:{}".format(normal_counter))
        print("all:{}".format(counter))
        
    def get_failure_type(self,ID):
        sql="select failure_type from injection_record_hadoop where fault_id='%s'" % ID
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        return results[0][0]

    def close(self):
        self.cursor.close()
        self.db.close()


            






if __name__ == '__main__':
        processor=Processor(sys.argv[1])

        processor.check()
        processor.close()
        

