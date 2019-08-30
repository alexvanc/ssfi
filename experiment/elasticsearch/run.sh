cd /work/search
service elasticsearch start
sleep 15
#$ES_HOME/bin/elasticsearch -d -p pidfile
jps > /tmp/startresult.log
curl -XGET 'http://localhost:9200/twitter/_search?q=user:kimchy&pretty=true' > runResult.txt 2 > outputError.txt
