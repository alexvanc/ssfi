cd $OUTPUT
mkdir -p /tmp/hadoop/logs/$ID/logs
mv $HADOOP_HOME/logs/* /tmp/hadoop/logs/$ID/logs/
python2 analyze_result.py $ID $RTIME $ACT_FILE


