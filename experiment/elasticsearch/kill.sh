#! /bin/bash
mkdir -p /tmp/search/logs/$ID/klogs
#mv $ES_HOME/log/* /tmp/search/logs/$ID/klogs/
mv /var/log/elasticsearch/* /tmp/search/logs/$ID/klogs/
cd /work/search
#pkill -F pidfile



