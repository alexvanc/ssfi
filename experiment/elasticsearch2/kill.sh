#! /bin/bash
mkdir -p /tmp/search/logs/$ID/klogs
mv $ES_HOME/log/* /tmp/search/logs/$ID/klogs/
cd /work/search
kill $(jps |grep -Ev 'Application|Jps'| cut -d " " -f 1)


