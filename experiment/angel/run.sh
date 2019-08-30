$HADOOP_HOME/sbin/start-all.sh
jps > /tmp/startresult.log
$HADOOP_HOME/bin/hadoop dfsadmin -safemode leave
cd $SPARK_HOME
./bin/spark-submit --class org.apache.spark.examples.JavaWordCount --master yarn --deploy-mode client --driver-memory 1g --executor-memory 1g  --queue default examples/jars/spark-examples*.jar input >/tmp/runResult.txt 2> /tmp/runError.txt