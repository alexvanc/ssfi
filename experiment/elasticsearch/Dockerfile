#this is the Dockerfile only for elasticsearch 
#without tracing complied and enabled

FROM rappdw/docker-java-python:latest

# install openssh-server, wget and gcc and other dependencies of tracing
RUN apt-get update && apt-get install -y openssh-server wget vim uuid-dev libconfig-dev libcurl4-openssl-dev build-essential python-mysqldb

# install elasticsearch
RUN mkdir -p /work/search/search-output
COPY *.sh *.py /work/search/search-output/
RUN wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add -
RUN apt-get -y install apt-transport-https
RUN echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | tee -a /etc/apt/sources.list.d/elastic-7.x.list
RUN apt-get update && apt-get -y install elasticsearch
ENV ES_HOME=/usr/share/elasticsearch

WORKDIR /work/search

#config hadoop cluster
COPY ./* /tmp/
COPY ./init.sh /work/
COPY ./log4j2.properties /etc/elasticsearch/log4j2.properties
RUN chown elasticsearch /etc/elasticsearch/log4j2.properties

RUN chmod +x /work/search/search-output/*.sh && \
    chmod +x /work/init.sh

COPY ssfi.jar /work/
COPY search.tar.gz /tmp/
RUN tar  -C /work/search/ -xzf /tmp/search.tar.gz


CMD [ "sh", "-c", "/work/init.sh; /bin/bash"]
