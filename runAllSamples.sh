#!/bin/bash
COMMAND="python convertTree.py "
OUTFOLDER=$1
INFOLDER=$2
INPREFIX=$3


SEARCHFOLDER="$INFOLDER/*$INPREFIX*.cfg"

for FILE in ${SEARCHFOLDER}
do
    echo "${COMMAND} --output ${OUTFOLDER} --config ${FILE} >> convert${INPREFIX}.log 2>&1"
    ${COMMAND} --output ${OUTFOLDER} --config ${FILE} >> convert${INPREFIX}.log 2>&1
done
