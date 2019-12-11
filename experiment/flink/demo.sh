#!/bin/sh
start-cluster.sh
flink run flink-1.9.1/examples/streaming/WordCount.jar --output wordcount.out

