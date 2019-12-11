# this is just a copy from the official file, no any changes
#this is the Dockerfile only for hadoop
#without tracing complied and enabled
#without message ID tracing enabled
#without Hadoop tracing enabled

FROM rappdw/docker-java-python:latest

# install openssh-server, wget and gcc and other dependencies of tracing
RUN apt-get update && apt-get install -y openssh-server wget vim uuid-dev libconfig-dev libcurl4-openssl-dev build-essential python-mysqldb

RUN mkdir -p /work/hadoop && mkdir -p /work/spark && mkdir /work/spark/spark-output 
COPY *.sh *.py /work/spark/spark-output/

WORKDIR /work/hadoop

# install hadoop 2.6.5 with network
# RUN wget http://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-2.6.5/hadoop-2.6.5.tar.gz && \
#     tar -xzvf hadoop-2.6.5.tar.gz && \
#     mv hadoop-2.6.5 /usr/local/hadoop && \
#     rm hadoop-2.6.5.tar.gz

# install hadoop 2.6.5 with local hadoop file
COPY hadoop-2.6.5.tar.gz /work/hadoop/
RUN tar -xzvf hadoop-2.6.5.tar.gz && \
    mv hadoop-2.6.5 /usr/local/hadoop && \
    rm hadoop-2.6.5.tar.gz
  
WORKDIR /work/spark  
#RUN wget https://archive.apache.org/dist/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.6.tgz && \
COPY spark-2.3.1-bin-hadoop2.6.tgz /work/spark/
RUN	tar -xzvf spark-2.3.1-bin-hadoop2.6.tgz && \
    mv spark-2.3.1-bin-hadoop2.6 /usr/local/spark && \
    rm /work/spark/spark-2.3.1-bin-hadoop2.6.tgz 


# set environment variable
ENV HADOOP_HOME=/usr/local/hadoop
ENV SPARK_HOME=/usr/local/spark
ENV PATH=$PATH:/usr/local/hadoop/bin:/usr/local/hadoop/sbin

WORKDIR /work/hadoop
RUN mkdir -p hdfs/namenode && \
    mkdir -p hdfs/datanode && \
    mkdir -p /tmp/logs/hadoop && \
    mkdir -p /tmp/logs/spark

#config hadoop cluster
COPY config/* /tmp/

#copy wordcount data
RUN mkdir /tmp/input
COPY data/* /tmp/input/

#hadoop-env.sh --- for JAVA_HOME
#*.xml for cluster cofig
#mapred-site.xml and hadoop-daemon.sh and yarn-daemon.sh for logging
#log4j.properties for security logs level
RUN mkdir ~/.ssh && mv /tmp/ssh_config ~/.ssh/config && \
    mv /tmp/hadoop-env.sh $HADOOP_HOME/etc/hadoop/hadoop-env.sh && \
    mv /tmp/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml && \
    mv /tmp/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml && \
    mv /tmp/mapred-site.xml $HADOOP_HOME/etc/hadoop/mapred-site.xml && \
    mv /tmp/yarn-site.xml $HADOOP_HOME/etc/hadoop/yarn-site.xml && \
    mv /tmp/hadoop-config-notrace.sh $HADOOP_HOME/libexec/hadoop-config.sh && \
    mv /tmp/log4j.properties $HADOOP_HOME/etc/hadoop/log4j.properties && \
    mv /tmp/log4j.properties.spark $SPARK_HOME/conf/log4j.properties && \
    cp /tmp/slaves $HADOOP_HOME/etc/hadoop/slaves && \
    mv /tmp/slaves $SPARK_HOME/conf/slaves && \
    mv /tmp/spark-env.sh $SPARK_HOME/conf/spark-env.sh && \
    mv /tmp/spark-defaults.conf $SPARK_HOME/conf/spark-defaults.conf && \
    mv /tmp/start-*.sh ~/ && \
    mv /tmp/sysctl.conf /etc/sysctl.conf && \
    mv /tmp/init.sh /work/
    
RUN chmod +x ~/start-hadoop.sh && \
    chmod +x ~/*.sh && \
    chmod +x $HADOOP_HOME/sbin/start-dfs.sh && \
    chmod +x $HADOOP_HOME/sbin/start-yarn.sh && \
    chmod +x /work/init.sh

COPY ssfi.jar /work/
COPY spark.tar.gz /tmp/
COPY spark-dependency.tar.gz /tmp/
RUN tar  -C /work/spark -xzf /tmp/spark.tar.gz

RUN tar  -C /work/spark -xzf /tmp/spark-dependency.tar.gz


CMD [ "sh", "-c", "/work/init.sh; /bin/bash"]
