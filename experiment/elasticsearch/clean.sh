#! /bin/bash
cd $INPUT
rm $ES_HOME/lib/$JARNAME.jar
mv $OUTPUT/$JARNAME.jar.bak $ES_HOME/lib/$JARNAME.jar

#put files back to elasticsearch
#rm -rf $ES_HOME/data/*
rm -rf /var/lib/elasticsearch/*
cd /work/search
#$ES_HOME/bin/elasticsearch -d -p pidfile
service elasticsearch start
sleep 16
curl -XPUT 'http://localhost:9200/twitter/_doc/1?pretty' -H 'Content-Type: application/json' -d '
{
    "user": "kimchy",
    "post_date": "2009-11-15T13:12:00",
    "message": "Trying out Elasticsearch, so far so good?"
}'
curl -XPUT 'http://localhost:9200/twitter/_doc/2?pretty' -H 'Content-Type: application/json' -d '
{
    "user": "kimchy",
    "post_date": "2009-11-15T14:12:12",
    "message": "Another tweet, will it be indexed?"
}'
curl -XPUT 'http://localhost:9200/twitter/_doc/3?pretty' -H 'Content-Type: application/json' -d '
{
    "user": "elastic",
    "post_date": "2010-01-15T01:46:38",
    "message": "Building the site, should be kewl"
}'
service elasticsearch stop
#pkill -F pidfile
#rm -rf $ES_HOME/logs/*
rm -rf /var/log/elasticsearch/*
mv /tmp/*.txt /tmp/search/logs/$ID/
mv /tmp/startresult.log /tmp/search/logs/$ID/



