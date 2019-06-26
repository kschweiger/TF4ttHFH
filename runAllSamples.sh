#!/bin/bash
COMMAND="python convertTree.py "
OUTFOLDER=$1
INPREFIX=$2

SEARCHFOLDER="data/*$INPREFIX*.cfg"

for FILE in ${SEARCHFOLDER}
do
    echo "${COMMAND} --output ${OUTFOLDER} --config ${FILE} &>> convert${INPREFIX}.log"
    ${COMMAND} --output ${OUTFOLDER} --config ${FILE} &>> convert${INPREFIX}.log
done
