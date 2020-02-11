#! /bin/bash
hidden_size=$1
num_epochs=$2
batch_size=$3
train_size=$4

declare -a datasets=("/mnt/fi/hadoop-datasets/all"
                          "/mnt/fi/hadoop-datasets/all_DEE"
                          "/mnt/fi/hadoop-datasets/all_SEE"
                          "/mnt/fi/hadoop-datasets/all_DBE"
                          "/mnt/fi/hadoop-datasets/all_SBE"
                          "/mnt/fi/hadoop-datasets/all_DHANG"
                          "/mnt/fi/hadoop-datasets/all_SHANG"
                          )
declare -a thresholds=("0.5" "0.6" "0.7" "0.8" "0.9")

result_file="h_size:"$hidden_size"-n_epoch:"$num_epochs"-b_size:"$batch_size"-t_size:"$train_size"result.txt"

for threshold in "${thresholds[@]}"
do
    for dataset in "${datasets[@]}"
    do
        python MyRNN_predict.py -hidden_size=${hidden_size} -batch_size=${batch_size} -num_epochs=${num_epochs} -train_size=${train_size} -data_dir="$dataset" -threshold=$threshold >> "$result_file"
    done
done




