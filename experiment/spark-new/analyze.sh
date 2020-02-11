cd $OUTPUT
mkdir -p /tmp/hadoop/logs/$ID/logs
mv $HADOOP_HOME/logs/* /tmp/hadoop/logs/$ID/logs/
mv $SPARK_HOME/logs /tmp/hadoop/logs/$ID/slogs
python2 analyze_result.py $ID $RTIME $ACT_FILE
mv $ACT_FILE /tmp/hadoop/logs/$ID/


