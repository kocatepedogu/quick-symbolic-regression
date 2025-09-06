#!/bin/sh
# Download script for dataset
wget https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip
unzip yearpredictionmsd.zip
mv YearPredictionMSD.txt YearPredictionMSD.csv
