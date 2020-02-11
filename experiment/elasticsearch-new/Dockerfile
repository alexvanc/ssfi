#this is the Dockerfile only for elasticsearch 
#without tracing complied and enabled

FROM docker.elastic.co/elasticsearch/elasticsearch:6.2.4

# install openssh-server, wget and gcc and other dependencies of tracing
#RUN yum update -y && yum install -y openssh-server wget vim uuid-dev libconfig-dev libcurl4-openssl-dev build-essential python-mysqldb net-tools
RUN yum install -y openssh-server wget vim uuid-dev libconfig-dev libcurl4-openssl-dev build-essential python-mysqldb net-tools
RUN yum install -y MySQL-python java-1.8.0-openjdk-devel.x86_64
ENV ES_HOME=/usr/share/elasticsearch
ENV JAVA_HOME=/usr/lib/jvm/java-1.8.0
ENV PATH=$PATH:$JAVA_HOME/bin

# install elasticsearch
RUN mkdir -p /work/search/search-output
COPY *.sh *.py /work/search/search-output/

WORKDIR /work/search

#config hadoop cluster
COPY ./* /tmp/
COPY ./init.sh /work/
COPY ./check.sh /work/search/
COPY ./log4j2.properties $ES_HOME/config/log4j2.properties
RUN chown elasticsearch $ES_HOME/config/log4j2.properties
RUN echo "bootstrap.system_call_filter: false" >>$ES_HOME/config/elasticsearch.yml

RUN chmod +x /work/search/search-output/*.sh && \
    chmod +x /work/init.sh

COPY ssfi.jar /work/
COPY search.tar.gz /tmp/
COPY search-dependency.tar.gz /tmp/
RUN tar  -C /work/search/ -xzf /tmp/search.tar.gz
RUN tar  -C /work/search/ -xzf /tmp/search-dependency.tar.gz
RUN chown -R elasticsearch /work/search/search-output


CMD [ "sh", "-c", "/work/init.sh; /bin/bash"]
