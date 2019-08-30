#!/bin/bash
docker run --name search_always_null -v $(pwd)/config.yaml.null:/etc/fi/config.yaml -d $1
docker run --name search_always_value -v $(pwd)/config.yaml.value:/etc/fi/config.yaml -d $1
docker run --name search_always_short -v $(pwd)/config.yaml.short:/etc/fi/config.yaml -d $1
docker run --name search_always_uncaught -v $(pwd)/config.yaml.uncaught:/etc/fi/config.yaml -d $1
docker run --name search_always_unhandle -v $(pwd)/config.yaml.unhandle:/etc/fi/config.yaml -d $1
docker run --name search_always_border -v $(pwd)/config.yaml.border:/etc/fi/config.yaml -d $1
docker run --name search_always_inverse -v $(pwd)/config.yaml.inverse:/etc/fi/config.yaml -d $1
docker run --name search_always_fallthrough -v $(pwd)/config.yaml.fallthrough:/etc/fi/config.yaml -d $1
docker run --name search_always_default -v $(pwd)/config.yaml.default:/etc/fi/config.yaml -d $1
docker run --name search_always_remove -v $(pwd)/config.yaml.remove:/etc/fi/config.yaml -d $1
docker run --name search_always_shadow -v $(pwd)/config.yaml.shadow:/etc/fi/config.yaml -d $1
