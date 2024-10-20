#!/bin/bash
#Duplicate the UDP packets received by COSMOS so that both Yamcs and COSMOS can see the same TM

c1=cosmos_openc3-operator_1
c2=yamcs-operator_1

# Get the IPs of the containers
ip1=`docker inspect -f '{{.NetworkSettings.Networks.nos3_sc_1.IPAddress}}' $c1`
ip2=`docker inspect -f '{{.NetworkSettings.Networks.nos3_sc_1.IPAddress}}' $c2`


if [[ -z $ip1 || -z $ip2 ]]; then
  echo "Error: Unable to get IP addresses of the containers."
  exit 1
fi

pid1=$(docker inspect -f '{{.State.Pid}}' $c1)
pid2=$(docker inspect -f '{{.State.Pid}}' $c2)

if [[ -z $pid1 || -z $pid2 ]]; then
  echo "Error: Unable to get PIDs of the containers."
  exit 1
fi

echo "$c1:  $ip1 $pid1"
echo "$c2:  $ip2 $pid2"

port=5013

nsenter -t $pid2 -n iptables -t nat -F PREROUTING
nsenter -t $pid1 -n iptables -t mangle -F PREROUTING

for port in 5013 5111 6011; do
   echo "duplicating port $port"
   nsenter -n -t $pid2 iptables -t nat -A PREROUTING -p udp --dport $port -j DNAT --to-destination $ip2
   nsenter -n -t $pid1 iptables -t mangle -A PREROUTING -p udp --dport $port -j TEE --gateway $ip2
done 
nsenter -n -t $pid1 conntrack -F
nsenter -n -t $pid2 conntrack -F
