# coding=utf-8
"""
put this script at the same folder of the log files
and then import the log lines into table userlogs
"""
import MySQLdb
import os
import re
import datetime
import sys




class process_log():
    def __init__(self,dbname,ip):
        self.db = MySQLdb.connect("localhost", "root", "fake_passwd", dbname, charset="utf8")
        self.cursor = self.db.cursor()
        self.ip=ip

    def processDir(self, dirname):
        list = os.listdir(dirname)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            fn = os.path.join(dirname, list[i])
            if os.path.isfile(fn):
                if list[i] == 'stderr':
                    self.processOneFile(fn, 1)
                elif list[i] == 'syslog':
                    self.processOneFile(fn, 2)
                elif list[i]=='stdout':
                    self.processOneFile(fn,3)
                elif list[i].endswith('log'): #hadopp system log, others are applocation log
                    self.processOneFile(fn,4)
                pass
            elif os.path.isdir(fn):
                self.processDir(fn)

    def processOneFile(self, fn, label):
        filelines = open(fn, 'r').readlines()
        sql = ""
        uuid=""
        level=""
        loginfo=""
        ktid=""
        time=""
        timestamp=0
        # stderr with no time
        if label == 1:
            for line in filelines: #parse rules should be made based on the log format configuration of log4j
                line = line.strip()
                if len(line)>=37 and line[0]=='@' and line[36]=='@':
                    if uuid!="":#not the first line
                        sql = "insert into userlogs(uuid,level,loginfo,label,filename,ktid,source_ip) values ('%s','%s','%s',%s,'%s',%s,'%s');" % (
                        uuid, level, loginfo, str(label),fn,ktid,self.ip)
                        self.cursor.execute(sql) 
                    uuid = line[0:37].strip()
                    ktid=line[37:].strip().split(" ")[0]
                    index=37+len(ktid)+2
                    loginfo = MySQLdb.escape_string(line[index:]).decode("utf-8")
                    level = line.split(" ")[2].split(":")[1]
                else:# an free text line, just append it to last line's text part
                    loginfo+='\n'+MySQLdb.escape_string(line).decode("utf-8")
            if uuid!="":#for the last line
                sql = "insert into userlogs(uuid,level,loginfo,label,filename,ktid,source_ip) values ('%s','%s','%s',%s,'%s',%s,'%s');" % (
                uuid, level, loginfo, str(label),fn,ktid,self.ip)
                self.cursor.execute(sql) 
        # syslog
        if label == 2:
            for line in filelines:
                line = line.strip()
                if len(line)>=37 and line[0]=='@' and line[36]=='@':
                    if uuid!="":#not the first line
                        sql = "insert into userlogs(uuid,time,ltime,level,loginfo,label,filename,ktid,source_ip) values ('%s','%s',%s,'%s','%s',%s,'%s',%s,'%s');" % (
                        uuid, time, timestamp,level, loginfo, str(label),fn,ktid,self.ip)
                        self.cursor.execute(sql) 
                    uuid = line[0:37].strip()
                    ktid=line[37:].strip().split(" ")[0]
                    index=37+len(ktid)+2
                    loginfo = MySQLdb.escape_string(line[index:]).decode("utf-8")  # str转义
                    time = line[index:index+23]
                    timestamp=int(float(datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S,%f').strftime("%s.%f"))*1000)
                    level = line.split(" ")[4]
                else:# an free text line, just append it to last line's text part
                    loginfo+='\n'+MySQLdb.escape_string(line).decode("utf-8")
            if uuid!="":#for the last line
                sql = "insert into userlogs(uuid,time,level,loginfo,label,filename,ltime,ktid,source_ip) values ('%s','%s','%s','%s',%s,'%s',%s,%s,'%s');" % (
                uuid, time,level, loginfo, str(label),fn,timestamp,ktid,self.ip)
                self.cursor.execute(sql) 
        #stdout
        if label==3:#curretly don't deal with stdout log file, because it's alwasy empty
            if len(filelines)>0:
                print "Found unempty stdout file"
                return
        #hadoop system log file
        if label==4:
            for line in filelines:
                line=line.strip()
                if len(line)>=37 and line[0]=='@' and line[36]=='@':
                    if uuid!="":#not the first line
                        result=re.search(r'\s([A-Z]{4,5})\s',loginfo)
                        if result:
                            level=result.group(1)
                        else:
                            print "Unknown log format of hadoop system log"
                        sql = "insert into userlogs(uuid,time,level,loginfo,label,filename,ltime,ktid,source_ip) values ('%s','%s','%s','%s',%s,'%s',%s,%s,'%s');" % (
                        uuid, time, level, loginfo, str(label),fn,timestamp,ktid,self.ip)
                        self.cursor.execute(sql) 
                    uuid = line[0:37].strip()
                    ktid=line[37:].strip().split(" ")[0]
                    index=37+len(ktid)+2
                    loginfo = MySQLdb.escape_string(line[index:]).decode("utf-8")  # str转义
                    time = line[index:index+23]
                    timestamp=int(float(datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S,%f').strftime("%s.%f"))*1000)
                    
                else:
                    loginfo+='\n'+MySQLdb.escape_string(line).decode("utf-8")
            if uuid!="":#not the first line
                result=re.search(r'\s([A-Z]{4,5})\s',loginfo)
                if result:
                    level=result.group(1)
                else:
                    print "Unknown log format of hadoop system log"
                sql = "insert into userlogs(uuid,time,level,loginfo,label,filename,ltime,ktid,source_ip) values ('%s','%s','%s','%s',%s,'%s',%s,%s,'%s');" % (
                uuid, time, level, loginfo, str(label),fn,timestamp,ktid,self.ip)
                self.cursor.execute(sql) 

        self.db.commit()

if __name__ == '__main__':
    dirname = "logs"
    tmp = process_log(sys.argv[1],sys.argv[2])
    tmp.processDir(dirname)
