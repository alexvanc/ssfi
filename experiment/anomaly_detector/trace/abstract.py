# coding=utf-8
import MySQLdb
import os
import re
import sys

# haven't taken the cluster trace data into considerartion
class Abstract():
    def __init__(self,dbname,sourceTable,targetTable):
        self.db = MySQLdb.connect("localhost", "root", "fake_passwd",dbname , charset="utf8")
        self.cursor = self.db.cursor()
        self.source=sourceTable
        self.target=targetTable

    def import_all(self):
        sql="select distinct data_unit_id,ktid,source_ip from %s where data_unit_id is not null" % self.source
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            sql2="select id,source_type,ftype from %s where data_unit_id='%s' and ktid=%s and source_ip='%s'" % (self.source,result[0],result[1],result[2])
            self.cursor.execute(sql2)
            results2=self.cursor.fetchall()
            internal_length=len(results2)
            child_number=0
            f_sequence=[]
            source_sequence=[]
            for result2 in results2:
                f_sequence.append(str(result2[2]))
                source_sequence.append(str(result2[1]))
                sql3="select count(*) as number from %s where l2_parent=%s" % (self.source,result2[0])
                self.cursor.execute(sql3)
                results3=self.cursor.fetchall()
                child_number=child_number+len(results3)
            sql4="insert into %s(data_unit_id,ktid,source_ip,internal_length,child_number,f_sequence,source_sequence) values('%s',%s,'%s',%s,%s,'%s','%s')" % (self.target,result[0],result[1],result[2],internal_length,child_number,"-".join(f_sequence),"-".join(source_sequence))
            self.cursor.execute(sql4)
            self.db.commit()
        print "Finish Import"
    
    def link_units(self):
        sql="select id,data_unit_id,ktid,source_ip from %s order by id desc" % self.target
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            sql2="select l2_parent from %s where data_unit_id='%s' and ktid=%s and source_ip='%s' and l2_parent is not null" % (self.source,result[1],result[2],result[3])
            self.cursor.execute(sql2)
            results2=self.cursor.fetchall()
            if len(results2)!=0:
                sql3="select data_unit_id,ktid,source_ip from %s where id=%s" % (self.source,results2[0][0])
                self.cursor.execute(sql3)
                results3=self.cursor.fetchall()
                if len(results3)!=1:
                    print "Error, no this ID"
                else:
                    sql4="select id from %s where data_unit_id='%s' and ktid=%s and source_ip='%s'" % (self.target,results3[0][0],results3[0][1],results3[0][2])
                    self.cursor.execute(sql4)
                    results4=self.cursor.fetchall()
                    if len(results4)!=1:
                        print "Error, cannot find it in this table"
                    else:
                        sql5="update %s set parent_unit=%s where id=%s" % (self.target,results4[0][0],result[0])
                        self.cursor.execute(sql5)
                        self.db.commit()
        print "Finish Link"
    
    def calc_length(self):
        sql="select id,internal_length from %s where parent_unit is null" % self.target
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            internal_length=result[1]
            total_length=internal_length
            total_sub_length=self.calc_sub_length(result[0])
            total_length=total_length+total_sub_length
            sql2="update %s set child_total=%s where id=%s" % (self.target,total_length,result[0])
            self.cursor.execute(sql2)
            self.db.commit()
        print "Finish calculate length"
    
    def calc_sub_length(self,parent_unit_id):
        sql="select id,internal_length from %s where parent_unit=%s" % (self.target,parent_unit_id)
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        total_length=0
        for result in results:
            current_total=result[1]+self.calc_sub_length(result[0])
            total_length+=current_total
            sql2="update %s set child_total=%s where id=%s " % (self.target,current_total,result[0])
            self.cursor.execute(sql2)
            self.db.commit()
        return total_length



    def close(self):
        self.cursor.close()
        self.db.close()

if __name__ == '__main__':
    tmp = Abstract(sys.argv[1],sys.argv[2],sys.argv[3])
    tmp.import_all()
    tmp.link_units()
    tmp.calc_length()
    tmp.close()
