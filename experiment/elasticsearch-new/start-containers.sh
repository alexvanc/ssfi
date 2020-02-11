#!/bin/bash
docker run --name search_random_null -e "discovery.type=single-node" -v $(pwd)/config.yaml.null:/etc/fi/config.yaml -d $1
docker run --name search_random_value -e "discovery.type=single-node" -v $(pwd)/config.yaml.value:/etc/fi/config.yaml -d $1
docker run --name search_random_short -e "discovery.type=single-node" -v $(pwd)/config.yaml.short:/etc/fi/config.yaml -d $1
docker run --name search_random_uncaught -e "discovery.type=single-node" -v $(pwd)/config.yaml.uncaught:/etc/fi/config.yaml -d $1
docker run --name search_random_unhandle -e "discovery.type=single-node" -v $(pwd)/config.yaml.unhandle:/etc/fi/config.yaml -d $1
docker run --name search_random_border -e "discovery.type=single-node" -v $(pwd)/config.yaml.border:/etc/fi/config.yaml -d $1
docker run --name search_random_inverse -e "discovery.type=single-node" -v $(pwd)/config.yaml.inverse:/etc/fi/config.yaml -d $1
docker run --name search_random_fallthrough -e "discovery.type=single-node" -v $(pwd)/config.yaml.fallthrough:/etc/fi/config.yaml -d $1
docker run --name search_random_default -e "discovery.type=single-node" -v $(pwd)/config.yaml.default:/etc/fi/config.yaml -d $1
docker run --name search_random_remove -e "discovery.type=single-node" -v $(pwd)/config.yaml.remove:/etc/fi/config.yaml -d $1
docker run --name search_random_shadow -e "discovery.type=single-node" -v $(pwd)/config.yaml.shadow:/etc/fi/config.yaml -d $1
