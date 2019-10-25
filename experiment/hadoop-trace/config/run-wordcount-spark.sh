#!/bin/bash

# test the hadoop cluster by running wordcount

# create input files 
mkdir input
echo "Hello Docker" >input/file2.txt
echo "Hello Hadoop" >input/file1.txt

# create input directory on HDFS
hadoop fs -mkdir -p input

# put input files to HDFS
hdfs dfs -put ./input/* input
hadoop fs -rmr /output/*

# run wordcount 
cd $SPARK_HOME
./bin/spark-submit --class org.apache.spark.examples.JavaWordCount --master yarn --deploy-mode client --driver-memory 1g --executor-memory 1g  --queue default examples/jars/spark-examples*.jar input/ /output/1
