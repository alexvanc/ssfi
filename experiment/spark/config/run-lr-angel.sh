#!/bin/bash

# test the angel by running lr

# create input directory on HDFS
hadoop fs -rm -r /test
hadoop fs -rm -r /test_data
hadoop fs -mkdir /test
hadoop fs -mkdir /test_data
hadoop fs -put $ANGEL_HOME/data/exampledata/LRLocalExampleData/a9a.train /test_data/

# run lr
cd $ANGEL_HOME/bin
./angel-submit --angel.app.submit.class com.tencent.angel.ml.classification.lr.LRRunner --angel.train.data.path hdfs:/test_data --angel.log.path hdfs:/test/log --angel.save.model.path hdfs:/test/model --action.type train --ml.data.type libsvm --ml.feature.num 1024 --angel.job.name LR_test --angel.am.memory.gb 1 --angel.worker.memory.gb 1 --angel.ps.memory.gb 1
