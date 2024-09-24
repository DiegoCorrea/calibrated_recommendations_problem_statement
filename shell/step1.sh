#!/bin/bash
cd ./pierre_in_frame &&
python3 step1_preprocessing.py from_file=YES file_name=food &&
python3 step1_preprocessing.py from_file=YES file_name=ml-20m &&
python3 step1_preprocessing.py from_file=YES file_name=lfm-2b-subset &&
python3 step1_preprocessing.py from_file=YES file_name=food_distribution &&
python3 step1_preprocessing.py from_file=YES file_name=ml-20m_distribution &&
python3 step1_preprocessing.py from_file=YES file_name=lfm-2b-subset_distribution
python3 step1_preprocessing.py from_file=YES file_name=food_ohe &&
python3 step1_preprocessing.py from_file=YES file_name=ml-20m_ohe &&
python3 step1_preprocessing.py from_file=YES file_name=lfm-2b-subset_ohe