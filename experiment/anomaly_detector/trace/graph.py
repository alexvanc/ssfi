# coding=utf-8
import py2neo
import MySQLdb
import os
import sys
from py2neo import Graph,Node,Relationship

class Drawer():
    def __init__(self,dbname,table):
        self.db = MySQLdb.connect("localhost", "root", "fake_passwd",dbname , charset="utf8")
        self.cursor = self.db.cursor()
        self.graph=Graph("http://localhost:7474", username="neo4j", password="fake_passwd")
        # self.traceNodes={}
        # self.logNodes={}
        # self.threadNodes={}
        self.source=table
        self.unit_dict={}
        # self.Node
    
    # create all the nodes from combination
    # store all the nodes in three dictionary
    def import_all(self):
        sql="select distinct data_unit_id,ktid,source_ip from %s where data_unit_id is not null" % self.source
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            sql2="select id,source_type,event_id,ftype,data_unit_id,ktid,ltime,l2_parent,source_ip from %s where data_unit_id='%s' and ktid=%s and source_ip='%s'" % (self.source,result[0],result[1],result[2])
            self.cursor.execute(sql2)
            results2=self.cursor.fetchall()
            self.create_unit_node(result,results2)
            # for result2 in results2:


    
    def create_unit_node(self,node_key,events):
        unit_key=node_key[0]+str(node_key[1])+node_key[2]
        if self.unit_dict.has_key(unit_key):
            return
        length=len(events)
        all_events=""
        parent_id=0
        for event in events:
            if event[7] is not None:
                parent_id=event[7]
            all_events=all_events+str(event[0])+","+str(event[1])+","+str(event[2])+","+str(event[3])+","+str(event[6])+","+str(event[7])+"\n"
        tmpNode=Node("Unit",value=unit_key,number=length,contents=all_events)
        self.unit_dict[unit_key]=tmpNode
        self.graph.create(tmpNode)
        if parent_id!=0:
            self.create_unit_relation(unit_key,parent_id)
    
    def create_unit_relation(self,child_node_key,parent_id):
        sql="select data_unit_id,ktid,source_ip from %s where id=%s" % (self.source,parent_id)
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        if len(results)==0:
            print "Error: No this ID!"
        unit_key=results[0][0]+str(results[0][1])+results[0][2]
        if self.unit_dict.has_key(unit_key):
            relation=Relationship(self.unit_dict[unit_key],"Cause",self.unit_dict[child_node_key])
            self.graph.create(relation)
        else:
            sql2="select id,source_type,event_id,ftype,data_unit_id,ktid,ltime,l2_parent,source_ip from %s where data_unit_id='%s' and ktid=%s and source_ip='%s'" % (self.source,results[0][0],results[0][1],results[0][2])
            self.cursor.execute(sql2)
            results2=self.cursor.fetchall()
            self.create_unit_node(results[0],results2)
            relation=Relationship(self.unit_dict[unit_key],"Cause",self.unit_dict[child_node_key])
            self.graph.create(relation)
    
    def close(self):
        self.cursor.close()
        self.db.close()
        


if __name__ == '__main__':
    tmp = Drawer(sys.argv[1],sys.argv[2])
    tmp.import_all()
    # tmp.draw_segments()
    tmp.close()
