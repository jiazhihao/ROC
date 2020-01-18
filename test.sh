#!/bin/sh
old="$IFS"
IFS='_'
str="$*"
str=long0904_${str}_fix_reddit.txt
echo "$str"
IFS=$old
./gnn -ll:gpu 1 -ll:cpu 4 -ll:fsize 12000 -ll:zsize 30000 -lr $1 -decay $2 -decay-rate $3 -dropout $4 -layers $5 -file /scratch/users/zhihao/GNNDatasets/reddit-dgl/reddit-dgl -e $6 -seed $7> results/$str
