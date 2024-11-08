#!/bin/bash
cd ./pierre_in_frame &&
python3 step3_processing.py from_file=YES file_name=food_cross &&
python3 step3_processing.py from_file=YES file_name=food_cvtt &&
python3 step3_processing.py from_file=YES file_name=ml-1m_cross &&
python3 step3_processing.py from_file=YES file_name=ml-1m_cvtt &&
python3 step3_processing.py from_file=YES file_name=yelp_cross &&
python3 step3_processing.py from_file=YES file_name=yelp_cvtt