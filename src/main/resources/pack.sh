#! /bin/bash
cd $INPUT
rm test.jar
mv $TARGET $OUTPUT/$TARGET.bak
mv $OUTPUT/$TARGET $TARGET
jar -cf test.jar ./*
mv $TARGET $OUTPUT/$TARGET
mv $OUTPUT/$TARGET.bak $TARGET
