FROM rappdw/docker-java-python:latest
RUN apt-get update && apt-get -y install vim
RUN apt-get -y install mysql-server && apt-get -y install python-mysqldb
RUN mkdir -p /work/testInput && mkdir -p /etc/fi
RUN mkdir -p /work/testOutput
COPY *.sh *.py /work/testOutput/
CMD ["java","-cp","/work/ssfi.jar","com.alex.ssfi.Application","/etc/fi/config.xml"]

