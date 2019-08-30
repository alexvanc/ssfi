#! /bin/bash
cd $INPUT
rm weka.jar

cd $OUTPUT
mkdir /tmp/weka/logs/$ID

mv $INPUT/*.log /tmp/weka/logs/$ID/
mv /tmp/*Result.txt /tmp/weka/logs/$ID/
mv /tmp/*Normal.txt /tmp/weka/logs/$ID/
mv /tmp/*Error.txt /tmp/weka/logs/$ID/




