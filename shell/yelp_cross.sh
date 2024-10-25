#!/bin/bash
cd ./pierre_in_frame &&

python3 step1_preprocessing.py from_file=YES file_name=yelp_cross > ../logs/step1_yelp_cross.log &&
python3 step1_preprocessing.py from_file=YES file_name=yelp_distribution_cross > ../logs/step1_yelp_distribution_cross.log &&

python3 step2_searches.py from_file=YES file_name=yelp_cross > ../logs/step2_yelp_cross.log &&

python3 step3_processing.py from_file=YES file_name=yelp_cross > ../logs/step3_yelp_cross.log &&

python3 step4_postprocessing.py from_file=YES file_name=yelp_cross > ../logs/step4_yelp_cross.log