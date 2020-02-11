$HADOOP_HOME/sbin/start-all.sh
jps > /tmp/startresult.log
date +%F_%T >/tmp/timeCounter.txt
$HADOOP_HOME/bin/hadoop dfsadmin -safemode leave
hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/sources/hadoop-mapreduce-examples-2.6.5-sources.jar org.apache.hadoop.examples.WordCount input output
hdfs dfs -cat output/part-r-00000 > /tmp/runResult.txt 2> /tmp/outputError.txt
date +%F_%T >>/tmp/timeCounter.txt
