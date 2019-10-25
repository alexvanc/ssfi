#! /bin/bash
cd $INPUT
rm /usr/local/hadoop/share/hadoop/$COMPONENT/$JARNAME.jar
mv $OUTPUT/$JARNAME.jar.bak /usr/local/hadoop/share/hadoop/$COMPONENT/$JARNAME.jar

#put files back to hadoop
rm -rf /work/hadoop/hdfs/*
$HADOOP_HOME/bin/hdfs namenode -format -force
$HADOOP_HOME/sbin/start-dfs.sh
hadoop fs -mkdir -p input
hdfs dfs -put /tmp/input/* input
$HADOOP_HOME/sbin/stop-dfs.sh
rm -rf $HADOOP_HOME/logs
mv /tmp/*.txt /tmp/hadoop/logs/$ID/
mv /tmp/startresult.log /tmp/hadoop/logs/$ID/
rm /tmp/traceData.dat



