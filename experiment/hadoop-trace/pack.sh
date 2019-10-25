#! /bin/bash
cd $INPUT
mv $COMPONENT/$JARNAME/$TARGET $OUTPUT/$TARGET.bak
mv $OUTPUT/$TARGET  $COMPONENT/$JARNAME/$TARGET
cd $COMPONENT/$JARNAME
jar -cf $JARNAME.jar ./*
mv /usr/local/hadoop/share/hadoop/$COMPONENT/$JARNAME.jar $OUTPUT/$JARNAME.jar.bak
mv $JARNAME.jar /usr/local/hadoop/share/hadoop/$COMPONENT/
cd $INPUT
rm $COMPONENT/$JARNAME/$TARGET
mv $OUTPUT/$TARGET.bak $COMPONENT/$JARNAME/$TARGET
