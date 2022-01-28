#!/bin/bash

# Get the last geometry in a Gaussian optimisation or IRC
# Reads from the bottom of the file
# Author: Tristan Mackenzie

PATTERN="                         Standard orientation:"
PATTERN2=" -----"
SKIP=6
FILE=""

while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    --s)
      PATTERN="                         Standard orientation:"
      SKIP=6
      shift
      ;;
    --i)
      PATTERN="                          Input orientation:"
      SKIP=6
      shift
      ;;
    --f)
      PATTERN=" Center     Atomic             Integrated Forces"
      SKIP=4
      shift
      ;;
    *)
      FILE="$1"
      shift
      ;;
  esac
done

if [ -f "$FILE" ] ; then
  if command -v tac &> /dev/null; then
    tac "$FILE" | sed "/$PATTERN/q" | tac | tail -n +"$SKIP" | sed "/$PATTERN2/q" | sed '$d'
  else
    tail -r "$FILE" | sed "/$PATTERN/q" | tail -r | tail -n +"$SKIP" | sed "/$PATTERN2/q" | sed '$d'
  fi
else
  echo "File $FILE doesn't exist"
  exit 1
fi
