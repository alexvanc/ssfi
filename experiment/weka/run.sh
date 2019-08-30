cd $INPUT
java -Xmx2024m -classpath $CLASSPATH:weka.jar weka.classifiers.functions.SGD -x 10  -v -do-not-output-per-class-statistics -o -t /work/data/diabetes.head.arff -T /work/data/diabetes.tail.arff >/tmp/runResult.txt 2>/tmp/runError.txt
