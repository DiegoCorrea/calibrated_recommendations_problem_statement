#!/bin/bash
cd ./pierre_in_frame &&
python3 step1_preprocessing.py from_file=YES file_name=food_cross &&
python3 step1_preprocessing.py from_file=YES file_name=food_cvtt &&
python3 step1_preprocessing.py from_file=YES file_name=ml-1m_cross &&
python3 step1_preprocessing.py from_file=YES file_name=ml-1m_cvtt &&
python3 step1_preprocessing.py from_file=YES file_name=yelp_cross &&
python3 step1_preprocessing.py from_file=YES file_name=yelp_cvtt &&
python3 step1_preprocessing.py from_file=YES file_name=food_distribution_cross &&
python3 step1_preprocessing.py from_file=YES file_name=food_distribution_cvtt &&
python3 step1_preprocessing.py from_file=YES file_name=ml-1m_distribution_cross &&
python3 step1_preprocessing.py from_file=YES file_name=ml-1m_distribution_cvtt &&
python3 step1_preprocessing.py from_file=YES file_name=yelp_distribution_cross &&
python3 step1_preprocessing.py from_file=YES file_name=yelp_distribution_cvtt &&
python3 step1_preprocessing.py from_file=YES file_name=food_ohe_cross &&
python3 step1_preprocessing.py from_file=YES file_name=food_ohe_cvtt &&
python3 step1_preprocessing.py from_file=YES file_name=ml-1m_ohe_cross &&
python3 step1_preprocessing.py from_file=YES file_name=ml-1m_ohe_cvtt &&
python3 step1_preprocessing.py from_file=YES file_name=yelp_ohe_cross &&
python3 step1_preprocessing.py from_file=YES file_name=yelp_ohe_cvtt