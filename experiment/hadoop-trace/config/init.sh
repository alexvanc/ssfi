mkdir /tmp/input
echo "Hello Docker" >/tmp/input/file2.txt
echo "Hello Hadoop" >/tmp/input/file1.txt

#for hadoop ssh service
# ssh without key
ssh-keygen -t rsa -f ~/.ssh/id_rsa -P ''
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
service ssh start

#compile trace library
cd /trace && ./recompile.sh && cd -

#import files into hdfs
$HADOOP_HOME/bin/hdfs namenode -format -force
$HADOOP_HOME/sbin/start-dfs.sh
hadoop fs -mkdir -p input
hdfs dfs -put /tmp/input/* input
$HADOOP_HOME/sbin/stop-dfs.sh
rm -rf $HADOOP_HOME/logs
cd /work
java -cp /work/ssfi.jar com.alex.ssfi.Application /etc/fi/config.yaml