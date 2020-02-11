# coding=utf-8
import MySQLdb
import os
import re
import sys

# haven't taken the cluster trace data into considerartion
class Merge():
    def __init__(self,dbname,tableName):
        self.db = MySQLdb.connect("localhost", "root", "fake_passwd",dbname , charset="utf8")
        self.cursor = self.db.cursor()
        self.target=tableName

    def check_internal(self):
        sql="select distinct data_unit_id,ktid,source_ip from %s where data_unit_id is not null" % self.target
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            sql2="select id, ltime from %s where data_unit_id='%s' and ktid=%s and source_ip='%s' order by ltime asc" % (self.target,result[0],result[1],result[2])
            self.cursor.execute(sql2)
            results2=self.cursor.fetchall()
            couter=0
            ltime=0
            for result2 in results2:
                sql3=""
                if couter==0:
                    ltime=result2[1]
                    sql3="update %s set l2_index=%s where id=%s" % (self.target,couter,result2[0])
                elif result2[1]==ltime:
                    sql3="update %s set l2_index=%s,l2_index_alt=1 where id=%s" % (self.target,couter,result2[0])
                else:
                    sql3="update %s set l2_index=%s where id=%s" % (self.target,couter,result2[0])
                self.cursor.execute(sql3)
                self.db.commit()
                couter=couter+1
            sql3="update %s set l2_length=%s where data_unit_id='%s' and ktid=%s and source_ip='%s' and l2_index=0" % (self.target,couter,result[0],result[1],result[2])
            self.cursor.execute(sql3)
            self.db.commit()
        print "Done internal"
    
    def check_after_write(self):
        counter=0
        sql="select id,data_unit_id,ktid,source_ip from %s where l2_index=0 or l2_index_alt=1" % self.target
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            sql2="select id from %s where ktid=%s and data_unit_id<>'%s' and data_id='%s' and source_ip='%s'" % (self.target,result[2],result[1],result[1],result[3])
            self.cursor.execute(sql2)
            results2=self.cursor.fetchall()
            number=len(results2)
            counter=counter+len(results2)
            if number!=0:
                if number==1:
                    sql3="update %s set l2_parent=%s where id=%s" % (self.target,results2[0][0],result[0])
                    self.cursor.execute(sql3)
                    self.db.commit()
                else:
                    print "Error more than one parent"
        print counter
        print "Done write"
    
    
    def check_thread(self):
        counter=0
        sql="select id,data_unit_id,ktid,source_ip from %s where l2_index=0 or l2_index_alt=1" % self.target
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            sql2="select id from threads where k_thread_id=%s and data_unit_id='%s' and source_ip='%s'" % (result[2],result[1],result[3])
            self.cursor.execute(sql2)
            results2=self.cursor.fetchall()
            number=len(results2)
            if number==0:
                counter=counter+1
            elif number==1:
                sql3="select id from %s where source_type=2 and event_id=%s" % (self.target,results2[0][0]) 
                self.cursor.execute(sql3)
                results3=self.cursor.fetchall()
                if len(results3)!=1:
                    print "Error, Not found thread in combination table"
                else:
                    sql4="update %s set l2_parent=%s where id=%s" % (self.target,results3[0][0],result[0])
                    self.cursor.execute(sql4)
                    self.db.commit()
            else:
                print "Error more than one parent thread"
        print "Done thread"
    
    def check_network(self):
        sql="select id,ftype,data_id from %s where data_unit_id is not null and data_id is not null" % self.target
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            if result[1]%2<>0:
                sql2="update %s set l2_parent=%s where data_id='%s' and id<>%s" % (self.target,result[0],result[2],result[0])
                self.cursor.execute(sql2)
                self.db.commit()
        print "Done network"
    
    def check_timeOrder(self):
        sql="select distinct data_unit_id,ktid,source_ip from %s where data_unit_id is not null" % self.target
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            sql2="select id from %s where data_unit_id='%s' and ktid=%s and source_ip='%s' order by ltime asc" % (self.target,result[0],result[1],result[2])
            self.cursor.execute(sql2)
            results2=self.cursor.fetchall()
            time_parent_id=0
            for result2 in results2:
                if time_parent_id!=0:
                    sql3="update %s set time_parent=%s where id=%s " % (self.target,time_parent_id,result2[0])
                    self.cursor.execute(sql3)
                    self.db.commit()
                time_parent_id=result2[0]

            
    
    # check how many units with parent unit
    def check_trace(self):
        sql="select distinct data_unit_id,ktid,source_ip from %s where data_unit_id is not null" % self.target
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        total=len(results)
        counter=0
        for result in results:
            sql2="select id,l2_index,l2_index_alt,l2_parent from %s where data_unit_id='%s' and ktid=%s and source_ip='%s'" % (self.target,result[0],result[1],result[2])
            self.cursor.execute(sql2)
            results2=self.cursor.fetchall()
            for result2 in results2:
                if result2[3] is not None:
                    counter=counter+1
                    break
        print str(counter)+"/"+str(total)+"/"+str(total-counter)

    # Print the units without parents
    def check_trace2(self):
        sql="select distinct data_unit_id,ktid,source_ip from %s where data_unit_id is not null" % self.target
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        counter=0
        for result in results:
            sql2="select id,l2_index,l2_index_alt,l2_parent from %s where data_unit_id='%s' and ktid=%s and source_ip='%s' and l2_parent is not null" % (self.target,result[0],result[1],result[2])
            self.cursor.execute(sql2)
            results2=self.cursor.fetchall()
            if len(results2)==0:
                print result
                counter=counter+1
        print counter
    
    # check how many units have multiple parents
    def check_unit_parent(self):
        sql="select distinct data_unit_id,ktid,source_ip from %s" % self.target
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        counter=0
        for result in results:
            sql2="select id,l2_parent from %s where data_unit_id='%s' and ktid=%s and source_ip='%s' and l2_parent is not null" % (self.target,result[0],result[1],result[2])
            self.cursor.execute(sql2)
            results2=self.cursor.fetchall()
            if len(results2)>1:
                parent=results2[0][1]
                for result2 in results2:
                    if result2[1]!=parent:
                        counter=counter+1
                        print result
        print counter


    def close(self):
        self.cursor.close()
        self.db.close()

if __name__ == '__main__':
    tmp = Merge(sys.argv[1],sys.argv[2])
    #tmp.check_internal()
    #tmp.check_after_write()
    #tmp.check_thread()
    #tmp.check_network()
    #tmp.check_timeOrder()
    #tmp.check_trace()
    tmp.check_trace2()
    #tmp.check_unit_parent()
    tmp.close()
