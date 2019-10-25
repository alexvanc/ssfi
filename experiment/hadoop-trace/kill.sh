#! /bin/bash
mkdir -p /tmp/hadoop/logs/$ID/klogs
mv $HADOOP_HOME/logs/* /tmp/hadoop/logs/$ID/klogs/
$HADOOP_HOME/sbin/stop-all.sh




