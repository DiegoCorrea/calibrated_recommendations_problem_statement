#!/bin/bash
cd ./pierre_in_frame &&
python3 step4_postprocessing.py from_file=YES file_name=food &&
python3 step4_postprocessing.py from_file=YES file_name=ml-20m &&
python3 step4_postprocessing.py from_file=YES file_name=lfm-2b-subset