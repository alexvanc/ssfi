#this is the Dockerfile only for hadoop
#without tracing complied and enabled
#without message ID tracing enabled
#without Hadoop tracing enabled

FROM rappdw/docker-java-python:latest

# install openssh-server, wget and gcc and other dependencies of tracing
RUN apt-get update && apt-get install -y openssh-server wget vim uuid-dev libconfig-dev libcurl4-openssl-dev build-essential python-mysqldb

RUN mkdir -p /work/hadoop && mkdir /work/hadoop/hadoop-output
COPY *.sh *.py /work/hadoop/hadoop-output/

WORKDIR /work/hadoop

# install hadoop 2.6.5
RUN wget http://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-2.6.5/hadoop-2.6.5.tar.gz && \
    tar -xzvf hadoop-2.6.5.tar.gz && \
    mv hadoop-2.6.5 /usr/local/hadoop && \
    rm hadoop-2.6.5.tar.gz

# set environment variable
ENV HADOOP_HOME=/usr/local/hadoop
ENV PATH=$PATH:/usr/local/hadoop/bin:/usr/local/hadoop/sbin


RUN mkdir -p hdfs/namenode && \
    mkdir -p hdfs/datanode && \
    mkdir -p /tmp/hadoop/klogs && mkdir /tmp/hadoop/logs

#config hadoop cluster
COPY config/* /tmp/

#hadoop-env.sh --- for JAVA_HOME
#hadoop-config.sh --- for tracing
#*.xml for cluster cofig
#mapred-site.xml and hadoop-daemon.sh and yarn-daemon.sh for logging
#log4j.properties for security logs level
RUN mkdir ~/.ssh && mv /tmp/ssh_config ~/.ssh/config && mv /tmp/sshd_config /etc/ssh/sshd_config && \
    mv /tmp/hadoop-env.sh $HADOOP_HOME/etc/hadoop/hadoop-env.sh && \
    mv /tmp/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml && \
    mv /tmp/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml && \
    mv /tmp/mapred-site.xml $HADOOP_HOME/etc/hadoop/mapred-site.xml && \
    mv /tmp/yarn-site.xml $HADOOP_HOME/etc/hadoop/yarn-site.xml && \
    mv /tmp/hadoop-config-notrace.sh $HADOOP_HOME/libexec/hadoop-config.sh && \
    mv /tmp/yarn-daemon.sh $HADOOP_HOME/sbin/yarn-daemon.sh && \
    mv /tmp/hadoop-daemon.sh $HADOOP_HOME/sbin/hadoop-daemon.sh && \
    mv /tmp/log4j.properties $HADOOP_HOME/etc/hadoop/log4j.properties && \
    mv /tmp/slaves $HADOOP_HOME/etc/hadoop/slaves && \
    mv /tmp/start-*.sh ~/ && \
    mv /tmp/run-*.sh ~/ && \
    mv /tmp/sysctl.conf /etc/sysctl.conf && \
    mv /tmp/init.sh /work/
    
RUN chmod +x ~/start-hadoop.sh && \
    chmod +x ~/run-wordcount.sh && \
    chmod +x $HADOOP_HOME/sbin/start-dfs.sh && \
    chmod +x $HADOOP_HOME/sbin/start-yarn.sh && \
    chmod +x /work/init.sh

COPY ssfi.jar /work/
COPY hadoop.tar.bz /tmp/
RUN tar  -C /work/hadoop -xzf /tmp/hadoop.tar.bz


CMD [ "sh", "-c", "/work/init.sh; /bin/bash"]
