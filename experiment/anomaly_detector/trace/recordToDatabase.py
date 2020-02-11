# coding=utf-8
"""
import /tmp/traceData.dat into table trace_info
"""
import MySQLdb
import sys

from warnings import filterwarnings
filterwarnings('error', category = MySQLdb.Warning)

class RecordToDatabase(object):
    def __init__(self,dbname,ip):
        self.db = MySQLdb.connect("localhost", "root", "fake_passwd", dbname, charset="utf8")
        self.cursor = self.db.cursor()
        self.ip=ip

    def close(self):
        self.cursor.close()
        self.db.close()

    def convertDataToDict(self, line):
        sqlDict = {}
        attr = line.strip().split("&")
        for item in attr:
            pair = item.strip().split("=",1)
            if(len(pair)==2):
                sqlDict[pair[0]] = pair[1]
            else:
                sqlDict['rtype']=1
        return sqlDict

    def recordToDatabase(self, filename):
        dataFile = open(filename, 'r')
        for line in dataFile:
            """
            rtype=24: trace_info
            rtype=27: threads
            rtype=28: thread_dep
            """
            sqlDict = self.convertDataToDict(line)
            # print(sqlDict.keys())
            if sqlDict['rtype'] == '24':
                sql = "insert into trace_info(on_ip,on_port,in_ip,in_port,data_id,data_unit_id,pid,ktid,tid,ftype,ltime,length,supposed_length,rlength,connect_id,dtype,message,message2,source_ip)" \
                      " values ('%s',%s,'%s',%s,'%s','%s',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'%s','%s','%s');" % \
                      (sqlDict['on_ip'], sqlDict['on_port'], sqlDict['in_ip'], sqlDict['in_port'], sqlDict['uuid'], sqlDict['unit_uuid'],
                       sqlDict['pid'],sqlDict['ktid'],sqlDict['tid'], sqlDict['ftype'], sqlDict['ttime'], sqlDict['length'], sqlDict['supposed_length'],
                       sqlDict['rlength'],sqlDict['socket'],sqlDict['dtype'], sqlDict['message'], '',self.ip) 

            elif sqlDict['rtype'] == '27':
                sql = "insert into threads(process_id,k_thread_id,thread_id,p_process_id,pk_thread_id,p_thread_id,ltime,data_unit_id,source_ip) values " \
                      "(%s,%s,%s,%s,%s,%s,%s,'%s','%s');" % (sqlDict['pid'],sqlDict['ktid'],sqlDict['tid'],sqlDict['ppid'],sqlDict['pktid'],sqlDict['ptid'],sqlDict['ttime'],sqlDict['unit_uuid'],self.ip)

            elif sqlDict['rtype'] == '28':
                sql = "insert into thread_dep(process_id,k_thread_id,d_thread_id,j_thread_id,ltime,source_ip,data_unit_id) values" \
                      "(%s,%s,%s,%s,%s,'%s','%s');" % (sqlDict['pid'],sqlDict['ktid'],sqlDict['dtid'],sqlDict['jtid'],sqlDict['ttime'],self.ip,sqlDict['unit_uuid'])
            elif sqlDict['rtype']=='29':
                sql="insert into other_event(ftype,result,ltime,pid,ktid,tid,data_unit_id,source_ip) values (%s,%s,%s,%s,%s,%s,'%s','%s')" % (sqlDict['ftype'],sqlDict['result'],sqlDict['ltime'],
                sqlDict['pid'],sqlDict['ktid'],sqlDict['tid'],sqlDict['unit_uuid'],self.ip)
            else:
                print sqlDict
                # self.db.commit()
                continue
            try:
                self.cursor.execute(sql)
            except:
                print sqlDict
                print sql
        self.db.commit()


if __name__ == '__main__':
    fn = "traceData.dat"
    r = RecordToDatabase(sys.argv[1],sys.argv[2])
    r.recordToDatabase(fn)
