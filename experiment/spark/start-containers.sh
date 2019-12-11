#! /bin/bash
docker run --name spark_random_null --hostname hadoop-master -v /home/alex/data/spark:/tmp/hadoop/logs -v $(pwd)/config.yaml.null:/etc/fi/config.yaml -d $1 
docker run --name spark_random_value --hostname hadoop-master -v /home/alex/data/spark:/tmp/hadoop/logs -v $(pwd)/config.yaml.value:/etc/fi/config.yaml -d $1 
docker run --name spark_random_short --hostname hadoop-master -v /home/alex/data/spark:/tmp/hadoop/logs -v $(pwd)/config.yaml.short:/etc/fi/config.yaml -d $1 
docker run --name spark_random_uncaught --hostname hadoop-master -v /home/alex/data/spark:/tmp/hadoop/logs -v $(pwd)/config.yaml.uncaught:/etc/fi/config.yaml -d $1 
docker run --name spark_random_unhandle --hostname hadoop-master -v /home/alex/data/spark:/tmp/hadoop/logs -v $(pwd)/config.yaml.unhandle:/etc/fi/config.yaml -d $1 
docker run --name spark_random_border --hostname hadoop-master -v /home/alex/data/spark:/tmp/hadoop/logs -v $(pwd)/config.yaml.border:/etc/fi/config.yaml -d $1 
docker run --name spark_random_inverse --hostname hadoop-master -v /home/alex/data/spark:/tmp/hadoop/logs -v $(pwd)/config.yaml.inverse:/etc/fi/config.yaml -d $1 
docker run --name spark_random_fallthrough --hostname hadoop-master -v /home/alex/data/spark:/tmp/hadoop/logs -v $(pwd)/config.yaml.fallthrough:/etc/fi/config.yaml -d $1 
docker run --name spark_random_default --hostname hadoop-master -v /home/alex/data/spark:/tmp/hadoop/logs -v $(pwd)/config.yaml.default:/etc/fi/config.yaml -d $1 
docker run --name spark_random_remove --hostname hadoop-master -v /home/alex/data/spark:/tmp/hadoop/logs -v $(pwd)/config.yaml.remove:/etc/fi/config.yaml -d $1 
docker run --name spark_random_shadow --hostname hadoop-master -v /home/alex/data/spark:/tmp/hadoop/logs -v $(pwd)/config.yaml.shadow:/etc/fi/config.yaml -d $1 
