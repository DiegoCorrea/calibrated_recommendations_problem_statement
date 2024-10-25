#!/bin/bash
cd ./pierre_in_frame &&

python3 step1_preprocessing.py from_file=YES file_name=yelp_cross > ../logs/step1_yelp_cross.log &&
python3 step1_preprocessing.py from_file=YES file_name=yelp_cvtt > ../logs/step1_yelp_cvtt.log &&
python3 step1_preprocessing.py from_file=YES file_name=yelp_distribution_cross > ../logs/step1_yelp_distribution_cross.log &&
python3 step1_preprocessing.py from_file=YES file_name=yelp_distribution_cvtt > ../logs/step1_yelp_distribution_cvtt.log &&

python3 step2_searches.py from_file=YES file_name=yelp_cross > ../logs/step2_yelp_cross.log &&
python3 step2_searches.py from_file=YES file_name=yelp_cvtt > ../logs/step2_yelp_cvtt.log &&

python3 step3_processing.py from_file=YES file_name=yelp_cross > ../logs/step3_yelp_cross.log &&
python3 step3_processing.py from_file=YES file_name=yelp_cvtt > ../logs/step3_yelp_cvtt.log &&

python3 step4_postprocessing.py from_file=YES file_name=yelp_cross > ../logs/step4_yelp_cross.log &&
python3 step4_postprocessing.py from_file=YES file_name=yelp_cvtt > ../logs/step4_yelp_cvtt.log &&

python3 step5_metrics.py from_file=YES file_name=yelp > ../logs/step5_yelp.log