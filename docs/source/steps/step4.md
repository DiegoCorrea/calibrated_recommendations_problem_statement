### Step 4: Post-processing
Post-processing is the focus of this framework. We use the Scikit-Pierre to post-processing the candidate items given by the recommender algorithms provided by the Scikit-Surprise or other recommender library.
The parameters given from the command line are used to create one or more recommendation systems. So, you can change them and create a lot of different systems. It is possible to use the same candidate items set, given as entry to different post-processing formulations.
The recommendations produced by this step are saved in `data/experiment/<dataset_name>/recommendation_lists/`.
If some parameter is not specified, by default, the framework will apply all options registered to the void parameters.   
This step provides an interface with 11 command line parameters that are:
1. `--recommender=<recommender_name>`: The recommender algorithm name that will be used.
2. `--dataset=<dataset_name>`: The dataset name that will be loaded.
3. `--fold=<fold_number>`: The fold number that will be used to train.
4. `--trial=<trial_number>`: The trial number that will be used to train.

It is necessary to run the algorithm, dataset, fold and trial in step 3 to produce the candidate items set. These 4 parameters are dependent on the previous step, as all the steps. The next parameters will be used to configure the post-processing step.
5. `--tradeoff=<tradeoff_name>`: The tradeoff name that will be constructed based on the next parameters. The Scikit-Pierre provides 2 tradeoffs focused on calibration.
6. `--calibration=<measure_name>`: The measure name that will be used on the tradeoff as a fairness measure. The Scikit-Pierre provides 57 measures·
7. `--relevance=<relevance_name>`: The relevance name that will be used on the tradeoff. The Scikit-Pierre provides 2 measures·
8. `--weight=<weight_name>`: The tradeoff weight that will be used to bring balance between relevance and calibration. The Scikit-Pierre provides 2 personalized ways and accepts constants.
9. `--distribution=<distribution_name>`: The distribution name that will be used to extract the target and realized distributions. The Scikit-Pierre provides 2 distributions·
10. `--selector=<selector_name>`: The selector item algorithm name that will be used to choose the items and creates the recommendation list. The Scikit-Pierre provides the Surrogate Submodular algorithm·
11. `--list_size=<number_of_list_size>`: The number that defines the recommendation list size. The default is 10.

If any of these parameters are not given, all the registered or default options will be used in a multiprocessing way.
The number of CPUs allocated to this job and all jobs on the next steps is N-1.

#### Run Examples
1. Movielens 1M   
1.1. `python step4_postprocessing.py --dataset=ml-1m --trial=1 --fold=1 --recommender=SVD --tradeoff=LIN --relevance=NDCG --distribution=CWS --weight=VAR --calibration=COSINE`  
1.2. `python step4_postprocessing.py --dataset=ml-1m --trial=3 --fold=5 --recommender=SVDpp --tradeoff=LOG --relevance=SUM --distribution=WPS --weight=C@0.5 --calibration=KL`  
1.3. This example `weight` is not given by. The framework will produce 13 recommender systems, variety the weight between constant and personalized: `python step4_postprocessing.py --dataset=ml-1m --trial=3 --fold=5 --recommender=SVDpp --tradeoff=LOG --relevance=SUM --distribution=WPS --calibration=KL`  
1.4. This example `calibration` is not given by. The framework will produce 57 recommender systems, variety the measure between similarity and divergence: `python step4_postprocessing.py --dataset=ml-1m --trial=3 --fold=5 --recommender=SVDpp --tradeoff=LOG --relevance=SUM --distribution=WPS --weight=C@0.5`  

2. Yahoo Movies    
2.1. `python step4_postprocessing.py --dataset=yahoo-movies --trial=7 --fold=5 --recommender=SVD --tradeoff=LIN --relevance=NDCG --distribution=CWS --weight=VAR --calibration=COSINE`   
2.2. `python step4_postprocessing.py --dataset=yahoo-movies --trial=2 --fold=1 --recommender=USER_KNN_BASIC --tradeoff=LOG --relevance=SUM --distribution=WPS --weight=CGR --calibration=VICIS_EMANON2`  
2.3. This example `fold` and `trial` are not given by. The framework will produce 35 recommender systems, variety the train and test dataset: `python step4_postprocessing.py --dataset=yahoo-movies --recommender=SVDpp --tradeoff=LOG --relevance=SUM --distribution=WPS --calibration=KL`  
2.4. This example `tradeoff` is not given by. The framework will produce 2 recommender systems, variety the equation: `python step4_postprocessing.py --dataset=yahoo-movies --trial=3 --fold=5 --recommender=SVDpp--relevance=SUM --distribution=WPS --weight=C@0.5`  
