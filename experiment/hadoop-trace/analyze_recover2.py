# coding=utf-8
import sys
import MySQLdb
import os
from datetime import datetime 
import datetime as dt

class Analyzer(object):
    def __init__(self,activationFile):

        self.db = MySQLdb.connect("39.99.169.20", "alex", "fake_passwd", "detection", charset="utf8")
        self.cursor = self.db.cursor()
        self.activationFile=activationFile
        self.sqlDict={}
    

    
    def recoverActivation(self):
        tmpLines=[]
        if os.path.exists(self.activationFile):
            activationFile=open(self.activationFile)
            tmpLines=activationFile.readlines()
            activationFile.close()
            for line in tmpLines:
                tuples=line.strip().split(":")
                millSeconds=int(tuples[0])
                ID=tuples[1]
                if ID not in self.sqlDict:
                    self.sqlDict[ID]=datetime.fromtimestamp(millSeconds/1000.0)
        self.updateDB()
            
                
  
    
    def updateDB(self):
        for key in self.sqlDict:
            act_time=self.sqlDict[key]
            sql="update injection_record_hadoop set activation_time='%s' where fault_id='%s'" % (act_time.strftime("%Y-%m-%d %H:%M:%S"),key)
            self.cursor.execute(sql)
            self.db.commit()
    
    def close(self):
        self.cursor.close()
        self.db.close()


if __name__ == '__main__':
    
    analyzer=Analyzer("/work/hadoop/hadoop-output/activation.log")
    analyzer.recoverActivation()
    analyzer.close()
        

