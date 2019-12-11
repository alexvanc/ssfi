FROM rappdw/docker-java-python:latest
RUN apt-get update && apt-get install -y openssh-server wget vim uuid-dev libconfig-dev libcurl4-openssl-dev build-essential python-mysqldb

RUN mkdir -p /work/flink && mkdir /work/flink/flink-output 
COPY *.sh *.py /work/flink/flink-output/

WORKDIR /work/flink
#install flink with local files
COPY flink-1.9.1-bin-scala_2.11.tgz /work/flink/
RUN tar -xzvf flink-1.9.1-bin-scala_2.11.tgz && \
    mv flink-1.9.1 /usr/local/flink && \
    rm flink-1.9.1-bin-scala_2.11.tgz

# set environment variable
ENV FLINK_HOME=/usr/local/flink
ENV PATH=$PATH:/usr/local/flink/bin

#copy calculation data
RUN mkdir /tmp/input
COPY data/* /tmp/input/

#config flink cluster config
COPY config/* /tmp/

# config flink cluster
# change the default flink log format
RUN mv /tmp/log4j.properties $FLINK_HOME/conf/log4j.properties && \
    mv /tmp/init.sh /work/

RUN chmod +x /work/flink/flink-output/*.sh && \
    chmod +x /work/init.sh

COPY ssfi.jar /work/
COPY flink.tar.gz /tmp/
COPY flink-dependency.tar.gz /tmp/
RUN tar  -C /work/flink -xzf /tmp/flink.tar.gz

RUN tar  -C /work/flink -xzf /tmp/flink-dependency.tar.gz
CMD [ "sh", "-c", "/work/init.sh; /bin/bash"]
