#! /bin/bash
counter=$1
image_id=$2
mode=$3
fault=$4

while [ $counter -ge 1 ] 
do
    container_name=spark_$mode_$fault
    docker run --rm --name $container_name -v $(pwd)/config.yaml.$fault:/etc/fi/config.yaml -v /data/hadoop2:/tmp/hadoop/logs -d $image_id >>$mode_$fault_container.log
    if [ $counter -eq $1 ]
    then
        ./monitor.sh &
    fi
    n=1
    while [ $n -ne 0 ]
    do
        n=$(docker ps|grep $container_name|wc -l)
        sleep 2
    done
    counter=$[counter-1]
done