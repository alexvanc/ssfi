#! /bin/bash
n=$(netstat -tuanlp|awk '{print $4}'|grep 9200|wc -l)
while [ $n -ne 1 ]
do 
        sleep 1
        n=$(netstat -tuanlp|awk '{print $4}'|grep 9200|wc -l)
done
