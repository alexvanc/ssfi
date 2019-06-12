#! /bin/bash
cd $INPUT
mv $TARGET $OUTPUT/bak.class
mv $OUTPUT/$CLASS $TARGET
jar -cf test.jar ./*
java -cp test.jar com.alex.halo.App>>/tmp/halo_result.txt
rm test.jar
mv $TARGET $OUTPUT/$CLASS
mv $OUTPUT/bak.class $TARGET
cd $OUTPUT
python analyze_result.py halo_result.txt

