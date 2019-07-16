#this is the Dockerfile only for hadoop
#without tracing complied and enabled
#without message ID tracing enabled
#without Hadoop tracing enabled

FROM rappdw/docker-java-python:latest

# install openssh-server, wget and gcc and other dependencies of tracing
RUN apt-get update && apt-get install -y openssh-server wget vim uuid-dev libconfig-dev libcurl4-openssl-dev build-essential

RUN mkdir -p /work/hadoop

WORKDIR /work/hadoop

# install hadoop 2.6.5
RUN wget http://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-2.6.5/hadoop-2.6.5.tar.gz && \
    tar -xzvf hadoop-2.6.5.tar.gz && \
    mv hadoop-2.6.5 /usr/local/hadoop && \
    rm hadoop-2.6.5.tar.gz

# set environment variable
ENV HADOOP_HOME=/usr/local/hadoop
ENV PATH=$PATH:/usr/local/hadoop/bin:/usr/local/hadoop/sbin

# ssh without key
RUN ssh-keygen -t rsa -f ~/.ssh/id_rsa -P '' && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

RUN mkdir -p hdfs/namenode && \
    mkdir -p hdfs/datanode && \
    mkdir $HADOOP_HOME/logs

COPY config/* /tmp/

RUN mv /tmp/ssh_config ~/.ssh/config && \
    mv /tmp/hadoop-env.sh $HADOOP_HOME/etc/hadoop/hadoop-env.sh && \
    mv /tmp/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml && \
    mv /tmp/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml && \
    mv /tmp/mapred-site.xml $HADOOP_HOME/etc/hadoop/mapred-site.xml && \
    mv /tmp/yarn-site.xml $HADOOP_HOME/etc/hadoop/yarn-site.xml && \
    mv /tmp/hadoop-config-notrace.sh $HADOOP_HOME/libexec/hadoop-config.sh && \
    cp /tmp/slaves $HADOOP_HOME/etc/hadoop/slaves && \
    mv /tmp/start-*.sh ~/ && \
    mv /tmp/run-*.sh ~/ && \
    mv /tmp/sysctl.conf /etc/sysctl.conf
    
#indicate hadoop-master ip
#RUN cat /tmp/hostnames >> /etc/hosts

RUN chmod +x ~/start-hadoop.sh && \
    chmod +x ~/run-wordcount.sh && \
    chmod +x $HADOOP_HOME/sbin/start-dfs.sh && \
    chmod +x $HADOOP_HOME/sbin/start-yarn.sh
#RUN sysctl -p

WORKDIR /usr/local/hadoop

# format namenode
#RUN /usr/local/hadoop/bin/hdfs namenode -format

CMD [ "sh", "-c", "service ssh start; bash"]