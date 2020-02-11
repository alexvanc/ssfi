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
    
    # find all the root nodes
    def import_all(self):
        sql="select id,data_unit_id,ktid,child_total,child_total,source_sequence,source_ip from %s where parent_unit is null" % self.source
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            self.create_node(result)
            self.create_node_relation(result)
            # for result2 in results2:
    
    def create_node_relation(self,node):
        sql="select id,data_unit_id,ktid,child_total,child_number,source_sequence,source_ip from %s where parent_unit=%s" % (self.source,node[0])
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        for result in results:
            self.create_node(result)
            self.create_relation(node,result)
            self.create_node_relation(result)
    
    def create_node(self,node):
        
        # tmpNode=Node("Unit",value=node[1]+"-"+str(node[2])+"-"+node[6],number=node[3],content=node[5])
        unit_key=node[1]+"-"+str(node[2])+"-"+node[6]
        if self.unit_dict.has_key(unit_key):
            return
        tmpNode=Node("Unit",value=unit_key,number=node[3],content=node[5])
        self.unit_dict[unit_key]=tmpNode
        self.graph.create(tmpNode)
    
    def create_relation(self,parent,child):
        p_unit_key=parent[1]+"-"+str(parent[2])+"-"+parent[6]
        c_unit_key=child[1]+"-"+str(child[2])+"-"+child[6]
        relation=Relationship(self.unit_dict[p_unit_key],"Cause",self.unit_dict[c_unit_key])
        self.graph.create(relation)

    
    def close(self):
        self.cursor.close()
        self.db.close()
        


if __name__ == '__main__':
    tmp = Drawer(sys.argv[1],sys.argv[2])
    tmp.import_all()
    # tmp.draw_segments()
    tmp.close()
