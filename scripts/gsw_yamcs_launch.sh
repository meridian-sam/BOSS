#!/bin/bash -i
#
# Convenience script for NOS3 development
#

CFG_BUILD_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#SCRIPT_DIR=$CFG_BUILD_DIR/../../scripts
SCRIPT_DIR=$CFG_BUILD_DIR/../scripts
source $SCRIPT_DIR/env.sh
export GSW="yamcs-operator_1"

echo "Yamcs launch..."
#gnome-terminal --tab --title="Yamcs" -- $DFLAGS -v $BASE_DIR:$BASE_DIR -v /tmp/nos3:/tmp/nos3 -w $BASE_DIR/gsw/yamcs --name yamcs-operator_1 --network=nos3_core -p 8090:8090 maven:3.9.9-eclipse-temurin-17 mvn -Dmaven.repo.local=$BASE_DIR/.m2/repository yamcs:run
$DFLAGS -v $BASE_DIR:$BASE_DIR -v /tmp/nos3:/tmp/nos3 -w $BASE_DIR/gsw/yamcs --name yamcs-operator_1 --network=nos3_sc_1 -p 8090:8090 maven:3.9.9-eclipse-temurin-17 mvn -Dmaven.repo.local=$BASE_DIR/.m2/repository yamcs:run
