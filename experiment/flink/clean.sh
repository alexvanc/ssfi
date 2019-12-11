#! /bin/bash
cd $INPUT
rm /usr/local/flink/$COMPONENT/$JARNAME.jar
mv $OUTPUT/$JARNAME.jar.bak /usr/local/flink/$COMPONENT/$JARNAME.jar

rm -rf $FLINK_HOME/logs
mv /tmp/*.txt /tmp/flink/logs/$ID/
mv /tmp/startresult.log /tmp/flink/logs/$ID/







