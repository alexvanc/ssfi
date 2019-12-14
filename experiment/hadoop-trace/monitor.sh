#! /bin/bash
n=$(docker ps|wc -l)
while [ $n -ne 1 ]
do
    date +%F_%T >>monitor.log
    docker stats --no-stream >>monitor.log
    echo "" >>monitor.log
    n=$(docker ps|wc -l)
done

