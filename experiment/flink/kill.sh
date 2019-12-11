#! /bin/bash
stop-cluster.sh
kill $(jps |grep -Ev 'Application|Jps'| cut -d " " -f 1)




