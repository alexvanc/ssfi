#! /bin/bash
echo "always_null">>container.log
docker run --name hadoop_always_null --hostname hadoop-master -v $(pwd)/config.yaml.null:/etc/fi/config.yaml -d $1 >>container.log
echo "always_value">>container.log
docker run --name hadoop_always_value --hostname hadoop-master -v $(pwd)/config.yaml.value:/etc/fi/config.yaml -d $1 >>container.log
echo "always_short">>container.log
docker run --name hadoop_always_short --hostname hadoop-master -v $(pwd)/config.yaml.short:/etc/fi/config.yaml -d $1 >>container.log
echo "always_uncaught">>container.log
docker run --name hadoop_always_uncaught --hostname hadoop-master -v $(pwd)/config.yaml.uncaught:/etc/fi/config.yaml -d $1 >>container.log
echo "always_unhandle">>container.log
docker run --name hadoop_always_unhandle --hostname hadoop-master -v $(pwd)/config.yaml.unhandle:/etc/fi/config.yaml -d $1 >>container.log
echo "always_border">>container.log
docker run --name hadoop_always_border --hostname hadoop-master -v $(pwd)/config.yaml.border:/etc/fi/config.yaml -d $1 >>container.log
echo "always_inverse">>container.log
docker run --name hadoop_always_inverse --hostname hadoop-master -v $(pwd)/config.yaml.inverse:/etc/fi/config.yaml -d $1 >>container.log
echo "always_fallthrough">>container.log
docker run --name hadoop_always_fallthrough --hostname hadoop-master -v $(pwd)/config.yaml.fallthrough:/etc/fi/config.yaml -d $1 >>container.log
echo "always_default">>container.log
docker run --name hadoop_always_default --hostname hadoop-master -v $(pwd)/config.yaml.default:/etc/fi/config.yaml -d $1 >>container.log
echo "always_remove">>container.log
docker run --name hadoop_always_remove --hostname hadoop-master -v $(pwd)/config.yaml.remove:/etc/fi/config.yaml -d $1 >>container.log
echo "always_shadow">>container.log
docker run --name hadoop_always_shadow --hostname hadoop-master -v $(pwd)/config.yaml.shadow:/etc/fi/config.yaml -d $1 >>container.log
./monitor.sh &
