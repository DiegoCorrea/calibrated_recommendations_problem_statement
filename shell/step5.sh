#!/bin/bash
cd ./pierre_in_frame &&

python3 step5_metrics.py from_file=YES file_name=steck_2018_food_cross &&
python3 step5_metrics.py from_file=YES file_name=steck_2018_food_cvtt &&
python3 step5_metrics.py from_file=YES file_name=steck_2018_ml-1m_cross &&
python3 step5_metrics.py from_file=YES file_name=steck_2018_ml-1m_cvtt &&
python3 step5_metrics.py from_file=YES file_name=steck_2018_yelp_cross &&
python3 step5_metrics.py from_file=YES file_name=steck_2018_yelp_cvtt &&

python3 step5_metrics.py from_file=YES file_name=silva_2023_food_cross &&
python3 step5_metrics.py from_file=YES file_name=silva_2023_food_cvtt &&
python3 step5_metrics.py from_file=YES file_name=silva_2023_ml-1m_cross &&
python3 step5_metrics.py from_file=YES file_name=silva_2023_ml-1m_cvtt &&
python3 step5_metrics.py from_file=YES file_name=silva_2023_yelp_cross &&
python3 step5_metrics.py from_file=YES file_name=silva_2023_yelp_cvtt &&

python3 step5_metrics.py from_file=YES file_name=silva_2025_food_cross &&
python3 step5_metrics.py from_file=YES file_name=silva_2025_food_cvtt &&
python3 step5_metrics.py from_file=YES file_name=silva_2025_ml-1m_cross &&
python3 step5_metrics.py from_file=YES file_name=silva_2025_ml-1m_cvtt &&
python3 step5_metrics.py from_file=YES file_name=silva_2025_yelp_cross &&
python3 step5_metrics.py from_file=YES file_name=silva_2025_yelp_cvtt &&

python3 step5_metrics.py from_file=YES file_name=silva_2022_food_cross &&
python3 step5_metrics.py from_file=YES file_name=silva_2022_food_cvtt &&
python3 step5_metrics.py from_file=YES file_name=silva_2022_ml-1m_cross &&
python3 step5_metrics.py from_file=YES file_name=silva_2022_ml-1m_cvtt &&
python3 step5_metrics.py from_file=YES file_name=silva_2022_yelp_cross &&
python3 step5_metrics.py from_file=YES file_name=silva_2022_yelp_cvtt &&

python3 step5_metrics.py from_file=YES file_name=souza_2024_food_cross &&
python3 step5_metrics.py from_file=YES file_name=souza_2024_food_cvtt &&
python3 step5_metrics.py from_file=YES file_name=souza_2024_ml-1m_cross &&
python3 step5_metrics.py from_file=YES file_name=souza_2024_ml-1m_cvtt &&
python3 step5_metrics.py from_file=YES file_name=souza_2024_yelp_cross &&
python3 step5_metrics.py from_file=YES file_name=souza_2024_yelp_cvtt &&

python3 step5_metrics.py from_file=YES file_name=abdollahpouri_2021_food_cross &&
python3 step5_metrics.py from_file=YES file_name=abdollahpouri_2021_food_cvtt &&
python3 step5_metrics.py from_file=YES file_name=abdollahpouri_2021_ml-1m_cross &&
python3 step5_metrics.py from_file=YES file_name=abdollahpouri_2021_ml-1m_cvtt &&
python3 step5_metrics.py from_file=YES file_name=abdollahpouri_2021_yelp_cross &&
python3 step5_metrics.py from_file=YES file_name=abdollahpouri_2021_yelp_cvtt