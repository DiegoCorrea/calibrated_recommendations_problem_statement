# Step 1: Pre-processing
The pre-processing step is implemented for cleaning, filtering and modeling the data.
The framework pre-processing the dataset to provide a structure to be used by all next steps.
It applies a data partition in k-fold by n-trials, which k and n are given by the command line.
The pre-processing script provides three parameters to config the experiment:

## Variables and they values
1. `dataset=<dataset_name>`: The dataset name declared in the dataset class. By default, the framework provides eight dataset pre-process:  
1.1. Movielens 1M: `ml-1m`  
1.2. Yahoo Movies: `yahoo-movies`  
1.3. Food.com Recipes: `food`   
1.4. Movielens 20M: `ml-20m`  
1.5. Last.fm 2B subset: `lfm-2b-subset`  
1.6. My Anime List: `mal`   
1.7. Taste Profile: `taste-profile`  
1.8. Twitter Movies: `twitter_movies`   
2. `n_folds=<number_of_folds>`: The number of folds the user's preferences will be split.
3. `n_trials=<number_of_trials>`: The number of trials, one trial is a k-fold, 2 trials are 2 different sets of k-fold.
4. `cut_value=<score_value>`: It is the score number that higher than this number is considered positive feedback.
5. `opt`
6. `item_cut_value`
7. `profile_len_cut_value`
8. `based_on`

## Example of file  
Example of the dataset Food.com Recipes.
```
{
  "opt": "SPLIT",
  "based_on": "CROSS_TRAIN_VALIDATION_TEST",
  "dataset": "food",
  "n_folds": 5,
  "n_trials": 1,
  "cut_value": 3,
  "item_cut_value": 1,
  "profile_len_cut_value": 30,
  "test_len_cut_value": 10
}
```
The train, validation and test file from each fold and trial are created inside:
1. Train: `data/datasets/clean/<dataset_name>/trial-<trial_number>/fold-<fold_number>/train.csv`
2. Validation: `data/datasets/clean/<dataset_name>/trial-<trial_number>/fold-<fold_number>/validation.csv`  
3. Test: `data/datasets/clean/<dataset_name>/trial-<trial_number>/fold-<fold_number>/test.csv`  


The pre-processing script provides three functionality to config the experiment:  

1. Pre-process based on validation method  
   1.1. Movielens 20M: `python3 step1_preprocessing.py from_file=YES file_name=ml-20m`  
   1.2. Food.com: `python3 step1_preprocessing.py from_file=YES file_name=food`  
   1.3. Last.fm 2B: `python3 step1_preprocessing.py from_file=YES file_name=lfm-2b-subset`  
2. Pre-Compute Distribution. It improves the execution time.  
   2.1. Movielens 20M: `python3 step1_preprocessing.py from_file=YES file_name=ml-20m_distribution`  
   2.2. Food.com: `python3 step1_preprocessing.py from_file=YES file_name=food_distribution`  
   2.3. Last.fm 2B: `python3 step1_preprocessing.py from_file=YES file_name=lfm-2b-subset_distribution`  
3. Pre-Compute One-Hot-Encode. It improves the execution time.  
   3.1. Movielens 20M: `python3 step1_preprocessing.py from_file=YES file_name=ml-20m_ohe`  
   3.2. Food.com: `python3 step1_preprocessing.py from_file=YES file_name=food_ohe`  
   3.3. Last.fm 2B: `python3 step1_preprocessing.py from_file=YES file_name=lfm-2b-subset_ohe`  

You can find the environment files from step 1 in the directory `environment/step1/`.

#### Run Examples of Split option
In the directory `shell`, there are examples of script to run the step 1 for the three dataset cited above.
From the main project directory, you can run the command: `sh shell/step1.sh > ./logs/step1.log 2>&1 & disown`.
It will run all step 1 commands showed above.