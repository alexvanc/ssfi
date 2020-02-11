#! /bin/bash
counter=$1
image_id=$2
mode=$3
fault=$4

while [ $counter -ge 1 ] 
do
    container_name=hadoop_"$mode"_"$fault"_"$counter"
    container_basic_name=hadoop_"$mode"_"$fault"
    docker run --rm --name "$container_name" -v $(pwd)/config.yaml.$fault:/etc/fi/config.yaml -v /mnt2/fi/hadoop:/tmp/hadoop/logs -d $image_id >>"$mode"_"$fault"_container.log 2>&1
    n=1
    while [ $n -ne 0 ]
    do
        sleep 2
        n=$(docker ps|grep "$container_basic_name"|wc -l)
        echo "$container_name"
        echo $n
    done
    counter=$[counter-1]
done