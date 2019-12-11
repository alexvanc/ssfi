cd $OUTPUT
mkdir -p /tmp/search/logs/$ID/logs
mv /usr/share/elaticsearch/logs/* /tmp/search/logs/$ID/logs/
# mv /var/log/elasticsearch/* /tmp/search/logs/$ID/logs/
python2 analyze_result.py $ID $RTIME $ACT_FILE


