#!/bin/bash
 
model=$1
englishData=$2
dutchData=$3

python IBM.py -m $model -i 30 -e "$englishData" "$dutchData"
