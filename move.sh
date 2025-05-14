#!/bin/bash


base=~/Downloads/Semantic-mapping-for-Autonomous-Vehicles-main/annotations2


find "base" -type f -path "*/visualizations/*" -name '*_segmentation.png' -exec rm -v {} \;
