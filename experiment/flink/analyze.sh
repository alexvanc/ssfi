cd $OUTPUT
mkdir -p /tmp/flink/logs/$ID/logs
mv $FLINK_HOME/log/* /tmp/flink/logs/$ID/logs/
python2 analyze_result.py $ID $RTIME $ACT_FILE


