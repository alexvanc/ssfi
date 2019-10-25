#! /bin/sh
rm hook.so
gcc -fPIC -shared -o hook.so tracer.c -ldl -luuid -pthread -lconfig
