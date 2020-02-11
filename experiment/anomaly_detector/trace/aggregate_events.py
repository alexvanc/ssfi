# coding=utf-8
"""
aggregate the trace events from table no_unix,other_event,threads
and import the aggregated trace events into table all_trace_events
"""
import MySQLdb
import sys


class Aggregator(object):
    def __init__(self,dbname):
        self.db = MySQLdb.connect("localhost", "root", "fake_passwd", dbname, charset="utf8")
        self.cursor = self.db.cursor()

    def close(self):
        self.cursor.close()
        self.db.close()

    
    def start(self):
        self.startTrace()
        self.startThreads()
        self.startOther()
    
    def startTrace(self):
        sql="select id,ftype,data_id,data_unit_id,pid,tid,ktid,ltime, rlength,source_ip from no_unix2"
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            sql2="insert into all_trace_events(source_type,event_id,ftype,data_id,data_unit_id,pid,tid,ktid,ltime,result,source_ip) values(%s,%s,%s,'%s','%s',%s,%s,%s,%s,%s,'%s')" % (1,result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9])
            self.cursor.execute(sql2)
        self.db.commit()
    
    def startThreads(self):
        sql="select id,data_unit_id,p_process_id,p_thread_id,pk_thread_id,process_id,ltime,source_ip from threads"
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            ftype=21
            data_id=None
            if result[2]==result[5]:
                ftype=23
            sql2="insert into all_trace_events(source_type,event_id,ftype,data_id,data_unit_id,pid,tid,ktid,ltime,result,source_ip) values(%s,%s,%s,'%s','%s',%s,%s,%s,%s,%s,'%s')" % (2,result[0],ftype,data_id,result[1],result[2],result[3],result[4],result[6],1,result[7])
            self.cursor.execute(sql2)
        self.db.commit()

    def startOther(self):
        sql="select id,ftype,data_unit_id,pid,tid,ktid,ltime,result,source_ip from other_event"
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            data_id=None
            sql2="insert into all_trace_events(source_type,event_id,ftype,data_id,data_unit_id,pid,tid,ktid,ltime,result,source_ip) values(%s,%s,%s,'%s','%s',%s,%s,%s,%s,%s,'%s')" % (3,result[0],result[1],data_id,result[2],result[3],result[4],result[5],result[6],result[7],result[8])
            self.cursor.execute(sql2)
        self.db.commit()
    
    def reOrder(self):
        sql="select source_type,event_id,ftype,data_id,data_unit_id,pid,tid,ktid,ltime,result,source_ip from all_trace_events order by ltime asc"
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            sql2="insert into combination1(source_type,event_id,ftype,data_id,data_unit_id,pid,tid,ktid,ltime,result,source_ip) values(%s,%s,%s,'%s','%s',%s,%s,%s,%s,%s,'%s')" %(result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10])
            self.cursor.execute(sql2)
        self.db.commit()

if __name__ == '__main__':
    if len(sys.argv)==2:
        r = Aggregator(sys.argv[1])
        r.start()
    elif len(sys.argv)==3:
        r = Aggregator(sys.argv[1])
        r.reOrder()
    else:
        print "Failure"
    