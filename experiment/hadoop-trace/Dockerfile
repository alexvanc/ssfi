#this is the Dockerfile only for hadoop
#without tracing complied and enabled
#without message ID tracing enabled
#without Hadoop tracing enabled

FROM rappdw/docker-java-python:latest

# install openssh-server, wget and gcc and other dependencies of tracing
RUN apt-get update && apt-get install -y openssh-server wget vim uuid-dev libconfig-dev libcurl4-openssl-dev build-essential python-mysqldb

RUN mkdir -p /work/hadoop && mkdir /work/hadoop/hadoop-output
RUN mkdir -p /trace
COPY trace/* /trace/
RUN mv /trace/trace.conf /etc/
COPY *.sh *.py /work/hadoop/hadoop-output/

WORKDIR /work/hadoop

# install hadoop 2.6.5 with network
#RUN wget https://archive.apache.org/dist/hadoop/core/hadoop-2.6.5/hadoop-2.6.5.tar.gz && \
#   tar -xzvf hadoop-2.6.5.tar.gz && \
#    mv hadoop-2.6.5 /usr/local/hadoop && \
#    rm hadoop-2.6.5.tar.gz

# install hadoop 2.6.5 with local hadoop file
COPY hadoop-2.6.5.tar.gz /work/hadoop/
RUN tar -xzvf hadoop-2.6.5.tar.gz && \
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

#copy wordcount data
RUN mkdir /tmp/input
COPY data/* /tmp/input/

#hadoop-env.sh --- for JAVA_HOME
#hadoop-config.sh --- for tracing
#*.xml for cluster cofig
#mapred-site.xml and hadoop-daemon.sh and yarn-daemon.sh for logging
#log4j.properties for security logs level
RUN mkdir ~/.ssh && mv /tmp/ssh_config ~/.ssh/config && \
    mv /tmp/hadoop-env.sh $HADOOP_HOME/etc/hadoop/hadoop-env.sh && \
    mv /tmp/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml && \
    mv /tmp/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml && \
    mv /tmp/mapred-site.xml $HADOOP_HOME/etc/hadoop/mapred-site.xml && \
    mv /tmp/yarn-site.xml $HADOOP_HOME/etc/hadoop/yarn-site.xml && \
    mv /tmp/hadoop-config-trace.sh $HADOOP_HOME/libexec/hadoop-config.sh && \
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
    chmod +x /trace/recompile.sh && \
    chmod +x /work/init.sh

COPY ssfi.jar /work/
COPY hadoop.tar.gz /tmp/
COPY hadoop-dependency.tar.gz /tmp/
#hadoop.tar.gz specify which jars to inject faults
RUN tar  -C /work/hadoop -xzf /tmp/hadoop.tar.gz
RUN tar  -C /work/hadoop -xzf /tmp/hadoop-dependency.tar.gz


CMD [ "sh", "-c", "/work/init.sh; /bin/bash"]
