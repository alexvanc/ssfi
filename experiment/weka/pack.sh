#! /bin/bash
cd $INPUT
mv $TARGET $OUTPUT/$TARGET.bak
mv $OUTPUT/$TARGET $TARGET
jar -cf weka.jar ./*
mv $TARGET $OUTPUT/$TARGET
mv $OUTPUT/$TARGET.bak $TARGET
