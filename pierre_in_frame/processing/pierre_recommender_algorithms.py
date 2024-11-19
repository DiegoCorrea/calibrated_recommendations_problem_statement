import logging

import pandas as pd
from recommender_pierre.autoencoders.CDAEModel import CDAEModel
from recommender_pierre.autoencoders.DeppAutoEncModel import DeppAutoEncModel
from recommender_pierre.baselines.Popularity import PopularityRecommender
from recommender_pierre.baselines.Random import RandomRecommender
from scikit_pierre.metrics.evaluation import MeanAveragePrecision

from datasets.registred_datasets import RegisteredDataset
from settings.labels import Label
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class PierreRecommenderAlgorithm:
    """
    Class to lead with the pierre recommender algorithms,
        generating the recommendation and saving in the results path
    """

    def __init__(
            self,
            experiment_name: str,
            recommender_name: str, dataset_name: str, fold: int, trial: int,
            list_size: int, split_methodology: str, metric: str = "map"
    ):
        """
        Class constructor.

        :param recommender_name: The recommender algorithm name to be load and fit.
        :param dataset_name:  The dataset name to be used by the recommender algorithm.
        :param fold: The fold number to be load.
        :param trial: The trial number to be load.
        :param list_size: The recommendation list size.
        """
        self.dataset = RegisteredDataset.load_dataset(dataset_name)
        self.recommender_name = recommender_name
        self.fold = fold
        self.trial = trial
        self.recommender = None
        self.list_size = list_size
        self.split_methodology = split_methodology
        self.experiment_name = experiment_name

        if self.recommender_name in Label.PIERRE_RECOMMENDERS_NO_PARAMS:
            if self.recommender_name == Label.POPULARITY_REC:
                self.recommender = PopularityRecommender(
                    list_size=int(self.list_size)
                )
            else:
                self.recommender = RandomRecommender(
                    list_size=int(self.list_size)
                )
        else:
            # Load the surprise recommender algorithm
            full_params = SaveAndLoad.load_hyperparameters_recommender(
                experiment_name=self.experiment_name, split_methodology=self.split_methodology,
                dataset=self.dataset.system_name, algorithm=self.recommender_name
            )
            if self.recommender_name == Label.DEEP_AE:
                self.recommender = DeppAutoEncModel(
                    factors=int(full_params["params"]["factors"]),
                    epochs=int(full_params["params"]["epochs"]),
                    dropout=int(full_params["params"]["dropout"]),
                    lr=int(full_params["params"]["lr"]),
                    reg=int(full_params["params"]["reg"]), list_size=int(self.list_size)
                )
            else:
                self.recommender = CDAEModel(
                    factors=int(full_params["params"]["factors"]),
                    epochs=int(full_params["params"]["epochs"]),
                    dropout=int(full_params["params"]["dropout"]),
                    lr=int(full_params["params"]["lr"]),
                    reg=int(full_params["params"]["reg"]), list_size=int(self.list_size)
                )

    def run(self):
        """
        Method to run the recommender algorithm, made and save the recommendation list
        """
        # fit the recommender algorithm
        logger.info(">>> Fit the recommender algorithm")
        self.dataset.set_environment(
            experiment_name=self.experiment_name,
            split_methodology=self.split_methodology
        )
        if self.split_methodology in Label.BASED_ON_VALIDATION:
            users_preferences = self.dataset.get_full_train_transactions(
                fold=self.fold, trial=self.trial
            )
        else:
            users_preferences = self.dataset.get_train_transactions(
                fold=self.fold, trial=self.trial
            )

        rec_lists_df = self.recommender.train_and_produce_rec_list(
            user_transactions_df=users_preferences
        )

        # Save all recommendation lists
        logger.info(">>> Saving...")
        SaveAndLoad.save_candidate_items(
            experiment_name=self.experiment_name, split_methodology=self.split_methodology,
            data=rec_lists_df,
            dataset=self.dataset.system_name, algorithm=self.recommender_name,
            fold=self.fold, trial=self.trial
        )
