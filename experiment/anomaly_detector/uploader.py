# coding=utf-8
import sys
import pymysql
import os
import datetime as dt
from datetime import datetime
from dateutil import parser as ps
import time


class Uploader(object):
    def __init__(self):
        self.db = pymysql.connect("39.99.169.20", "root", "fake_passwd", "detection", charset="utf8")
        self.cursor = self.db.cursor()
    

    def upload(self,hidden_size,batch_size,num_epochs,train_size,threshold,failure_type,false_positive,false_negative,true_positive,true_negative,precision,recall,f1,
        accuracy,false_positivate_rate):

        failure_type=failure_type.split("/")[-1]
        sql="insert into metrics_detect(hidden_size,batch_size,num_epochs,train_size,threshold,failure_type,false_positive,false_negative,true_positive,true_negative,p,r,f1, \
        accuracy,false_positivate_rate) values (%s,%s,%s,%s,%s,'%s',%s,%s,%s,%s,%s,%s,%s,%s,%s)" % (hidden_size,batch_size,num_epochs,train_size,threshold,failure_type,false_positive,false_negative,true_positive,true_negative,precision,recall,f1,
        accuracy,false_positivate_rate)
        self.cursor.execute(sql)
        self.db.commit()
    
    def markMetrics(self,fi):
        sql="update injection_record_hadoop set metrics_detect=1 where fault_id='%s' " % fi
        self.cursor.execute(sql)
        self.db.commit()
    
    def upload_deeplog(self,hidden_size,batch_size,num_epochs,train_size,num_candidates,failure_type,false_positive,false_negative,true_positive,true_negative,precision,recall,f1,
        accuracy,false_positivate_rate):
        failure_type=failure_type.split("/")[-1]
        sql="insert into deeplog_detect(hidden_size,batch_size,num_epochs,train_size,num_candidates,failure_type,false_positive,false_negative,true_positive,true_negative,p,r,f1, \
        accuracy,false_positivate_rate) values (%s,%s,%s,%s,%s,'%s',%s,%s,%s,%s,%s,%s,%s,%s,%s)" % (hidden_size,batch_size,num_epochs,train_size,num_candidates,failure_type,false_positive,false_negative,true_positive,true_negative,precision,recall,f1,
        accuracy,false_positivate_rate)
        self.cursor.execute(sql)
        self.db.commit()
    
    def markDeep(self,fi):
        sql="update injection_record_hadoop set deep_detect2=1 where fault_id='%s' " % fi
        self.cursor.execute(sql)
        self.db.commit()
    
    def mark_deep_detail(self,fi,info):
        fclass=info['class'].replace("'",'"')
        fcomponent=info['component']
        ftime=info['ltime']
        sql="select activation_time from injection_record_hadoop where fault_id='%s'" % fi
        self.cursor.execute(sql)
        atime=self.cursor.fetchall()[0][0]
        latency=0
        if (ftime<atime):
            print("Time dismatches in fault: {}".format(fi))
        else:
            latency=(ftime-atime).seconds

        sql="update injection_record_hadoop set deep_detect=1,deep_class='%s',deep_component='%s',deep_detect_time='%s', deep_detect_latency=%s where fault_id='%s'" % (fclass,fcomponent,ftime.strftime("%Y-%m-%d %H:%M:%S"),latency,fi)
        self.cursor.execute(sql)
        self.db.commit()

        
    
    def close(self):
        self.cursor.close()
        self.db.close()
