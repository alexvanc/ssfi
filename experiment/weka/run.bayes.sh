#! /bin/bash
cd $INPUT
java weka.classifiers.bayes.BayesNet -t /work/data/diabetes.head.arff -T /work/data/diabetes.tail.arff -D -x 10  -v -o -do-not-output-per-class-statistics -Q weka.classifiers.bayes.net.search.local.K2 -- -P 2 -S ENTROPY -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 1.0
