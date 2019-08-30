kill $(jps |grep -Ev 'Application|Jps'| cut -d " " -f 1)
cd /work/search
rm pidfile
