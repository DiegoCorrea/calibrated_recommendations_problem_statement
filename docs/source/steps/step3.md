### Step 3: Processing
The processing step uses the data pre-processed and the hyperparameters to train the chosen recommender algorithm.
The prediction generates a set of candidate items to be used by the post-processing.
All candidate items set are saved inside `data/experiment/{dataset}/candidate_items/{recommender}/trial-{trial}/fold-{fold}/candidate_items.csv`.

#### Run Examples
1. Movielens 20M   
   1.1. BPR + ALS: `python3 step3_processing.py from_file=YES file_name=ml-20m`    
   1.2. DeepAE: `python3 step3_processing.py from_file=YES file_name=deep_ae_ml-20m`    
2. Food.com Recipes    
   1.1. BPR + ALS: `python3 step3_processing.py from_file=YES file_name=food`    
   1.2. DeepAE: `python3 step3_processing.py from_file=YES file_name=deep_ae_food`     
3. Last.fm    
   1.1. BPR + ALS: `python3 step3_processing.py from_file=YES file_name=lfm-2b-subset`    
   1.2. DeepAE: `python3 step3_processing.py from_file=YES file_name=deep_ae_lfm-2b-subset`

In the directory `shell`, there are examples of script to run the step 3 for the three dataset cited above.
From the main project directory, you can run the command: `sh shell/step3.sh > ./logs/step3.log 2>&1 & disown`.
It will run all step 3 commands showed above.
