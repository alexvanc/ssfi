kill -9 $(jps |grep -Ev 'Application|Jps'| cut -d " " -f 1)
cd /work/search
