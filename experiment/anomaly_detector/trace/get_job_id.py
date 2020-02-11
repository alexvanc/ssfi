import datetime
from calendar import monthrange
import MySQLdb
import re
import sys




class Ploter(object):
	def __init__(self,dbname):
		self.db=MySQLdb.connect("10.211.55.38","root","fake_passwd",dbname,charset="utf8")
		self.cursor=self.db.cursor()

	def process_id(self):
		sql1="select id,message,pid,ktid,trace_root from no_unix where message<>'' order by id"
		self.cursor.execute(sql1)
		results=self.cursor.fetchall()
		for result in results:
			self.extractID(result)

	def extractID(self,result):
		resultObj=re.search(r'job_\d{13}_\d{4}', result[1])
		if resultObj:
			# print resultObj.group()
			sql="update no_unix set message2='%s' where id=%s" % (resultObj.group(), result[0])
			self.cursor.execute(sql)
			#mark subtrace as a part of job or not
			sql2="update no_unix set job_id='%s' where id=%s" % (resultObj.group(),result[4])
			self.cursor.execute(sql2)
			self.db.commit()
		else:
			print "No Job ID"



	def speculate(self):
		sql1="select id,message,pid,ktid,trace_root from no_unix3 where message<>'' and job_id is null order by id"
		self.cursor.execute(sql1)
		results=self.cursor.fetchall()
		for result in results:
			self.speculateID(result)

	def speculateID(self,result):
		resultObj=re.search(r'job_\d+', result[1])
		if resultObj:
			# print resultObj.group()
			sql="update no_unix3 set job_id='%s' where id=%s" % ("suspicious", result[0])
			if result[4] is not None:
				sql2="update no_unix3 set job_id='%s' where job_id is null and id=%s" % ("suspicious",result[4])
				self.cursor.execute(sql2)
			self.cursor.execute(sql)
			self.db.commit()
		else:

			print "No suspicious Job ID"

	

	def close(self):
		self.cursor.close()
		self.db.close()

ploter=Ploter(sys.argv[1])
ploter.process_id()
# ploter.speculate()
ploter.close()
