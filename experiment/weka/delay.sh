#! /bin/bash
sleep 2
java -Xmx2024m -classpath $CLASSPATH:weka.jar weka.classifiers.functions.SGD -x 10  -v -do-not-output-per-class-statistics -o -t ./data/supermarket.arff -T ./data/supermarket.arff
