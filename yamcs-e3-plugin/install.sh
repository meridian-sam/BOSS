#!/bin/sh
mvn install:install-file \
    -Dfile=target/yamcs-epsilon3-plugin-1.0.1.jar \
    -DgroupId=io.epsilon3.plugin \
    -DartifactId=yamcs-epsilon3-plugin \
    -Dversion=1.0.1 \
    -Dpackaging=jar \
    -DlocalRepositoryPath=../.m2/repository
