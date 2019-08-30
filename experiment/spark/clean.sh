#! /bin/bash
cd $INPUT
rm test.jar

#put files back to hadoop
rm -rf /work/hadoop/hdfs/*
$HADOOP_HOME/bin/hdfs namenode -format -force
$HADOOP_HOME/sbin/start-dfs.sh
hadoop fs -mkdir -p input
hdfs dfs -put /tmp/input/* input
$HADOOP_HOME/sbin/stop-dfs.sh
rm -rf $HADOOP_HOME/logs
mv /tmp/runResult.txt /tmp/hadoop/logs/$ID/
mv /tmp/startresult.log /tmp/hadoop/logs/$ID/
mv /tmp/runError.txt /tmp/hadoop/logs/$ID/
mv /tmp/runNormal.txt /tmp/hadoop/logs/$ID/



