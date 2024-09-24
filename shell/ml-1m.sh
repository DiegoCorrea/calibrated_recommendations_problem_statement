#!/bin/bash
cd ./pierre_in_frame &&
python3 step1_preprocessing.py from_file=YES file_name=ml-1m &&
python3 step1_preprocessing.py from_file=YES file_name=ml-1m_distribution &&
python3 step1_preprocessing.py from_file=YES file_name=ml-1m_ohe &&
python3 step2_searches.py from_file=YES file_name=ml-1m &&
python3 step3_processing.py from_file=YES file_name=ml-1m &&
python3 step4_postprocessing.py from_file=YES file_name=ml-1m &&
python3 step5_metrics.py from_file=YES file_name=ml-1m_steck
