#! /bin/bash
cd $INPUT
rm $ES_HOME/lib/$JARNAME.jar
mv $OUTPUT/$JARNAME.jar.bak $ES_HOME/lib/$JARNAME.jar

rm -rf /tmp/*elasticsearch*
cd /work/search
#rm -rf $ES_HOME/logs/*
rm -rf $ES_HOME/logs/*
rm -rf $ES_HOME/data/*
mv /tmp/*.txt /tmp/search/logs/$ID/
mv /tmp/startresult.log /tmp/search/logs/$ID/



