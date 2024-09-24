#!/bin/bash
cd ./code &&
python3 step5_metrics.py from_file=YES file_name=lfm-2b-subset &&
python3 step5_metrics.py from_file=YES file_name=food &&
python3 step5_metrics.py from_file=YES file_name=ml-20m
