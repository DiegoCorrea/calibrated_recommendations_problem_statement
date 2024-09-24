import logging

import pandas as pd
from surprise import SVD, NMF, KNNBasic
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.prediction_algorithms.matrix_factorization import SVDpp
from surprise.prediction_algorithms.slope_one import SlopeOne

from datasets.registred_datasets import RegisteredDataset
from processing.conversions.pandas_surprise import PandasSurprise
from settings.labels import Label
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class SurpriseRecommenderAlgorithm:
    """
    Class to lead with the surprise recommender algorithms,
        generating the recommendation and saving in the results path
    """

    def __init__(
            self,
            experiment_name: str,
            recommender_name: str, dataset_name: str, fold: int, trial: int,
            metric: str, based_on: str, list_size: int
    ):
        """
        Class constructor.

        :param recommender_name: The recommender algorithm name to be load and fit.
        :param dataset_name:  The dataset name to be used by the recommender algorithm.
        :param fold: The fold number to be load.
        :param trial: The trial number to be load.
        """
        self.dataset = RegisteredDataset.load_dataset(dataset_name)
        self.all_items = None
        self.all_items_ids = None
        self.recommender_name = recommender_name
        self.fold = fold
        self.trial = trial
        self.recommender = None
        self.list_size = list_size
        self.based_on = based_on
        self.experiment_name = experiment_name

        # Load the surprise recommender algorithm
        if self.recommender_name == Label.SLOPE:
            self.recommender = SlopeOne()
        else:
            full_params = SaveAndLoad.load_hyperparameters_recommender(
                experiment_name=self.experiment_name, based_on=self.based_on,
                dataset=self.dataset.system_name, algorithm=self.recommender_name
            )
            params = full_params[metric]
            if self.recommender_name == Label.SVD:
                self.recommender = SVD(
                    n_factors=params['n_factors'], n_epochs=params['n_epochs'],
                    lr_all=params['lr_all'], reg_all=params['reg_all'], biased=True,
                    random_state=42, verbose=True
                )
            elif self.recommender_name == Label.NMF:
                self.recommender = NMF(
                    n_factors=params['n_factors'], n_epochs=params['n_epochs'],
                    reg_bi=params['reg_bi'], reg_pu=params['reg_pu'],
                    reg_qi=params['reg_qi'], reg_bu=params['reg_bu'],
                    lr_bu=params['lr_bu'], lr_bi=params['lr_bi'],
                    biased=params['biased'],
                    random_state=42, verbose=True
                )
            elif self.recommender_name == Label.CO_CLUSTERING:
                self.recommender = CoClustering(
                    n_epochs=params['n_epochs'],
                    n_cltr_u=params['n_cltr_u'], n_cltr_i=params['n_cltr_i'],
                    verbose=True
                )
            elif self.recommender_name == Label.ITEM_KNN_BASIC:
                self.recommender = KNNBasic(
                    k=params['k'], sim_options=params['sim_options'], verbose=True
                )
            elif self.recommender_name == Label.USER_KNN_BASIC:
                self.recommender = KNNBasic(
                    k=params['k'], sim_options=params['sim_options'], verbose=True
                )
            elif self.recommender_name == Label.SVDpp:
                self.recommender = SVDpp(
                    n_factors=params['n_factors'], n_epochs=params['n_epochs'],
                    lr_all=params['lr_all'], reg_all=params['reg_all'],
                    random_state=42, verbose=True
                )

    def _user_unknown_items(self, users_preferences: pd.DataFrame, user_id: str) -> pd.DataFrame:
        """
        TODO: Docstring
        """
        user_unknown_items_ids = set(
            self.all_items_ids) - set(users_preferences['ITEM_ID'].unique().tolist())
        unk_df = pd.DataFrame()
        unk_df[Label.ITEM_ID] = list(user_unknown_items_ids)
        unk_df[Label.USER_ID] = user_id
        unk_df[Label.TRANSACTION_VALUE] = 0
        return unk_df

    def __predict(self, user_test_set: pd.DataFrame) -> pd.DataFrame:
        """
        Method to predict the rating to a user.

        :param user_test_set: A Pandas Dataframe with the user_id and item_id.
        :return: A Pandas Dataframe with the user_id, item_id and predicted_rating.
        """
        # Transform the pandas dataframe in a surprise dataset structure
        testset = PandasSurprise.pandas_transform_testset_to_surprise(testset_df=user_test_set)
        # Predict and transform surprise dataset structure in a pandas dataframe
        return PandasSurprise.surprise_to_pandas_get_candidates_items(
            predictions=self.recommender.test(testset=testset),
            n=self.list_size
        )

    def run(self):
        """
        Method to run the recommender algorithm, made and save the recommendation list
        """
        # fit the recommender algorithm
        logger.info(">>> Fit the recommender algorithm")
        self.dataset.set_environment(
            experiment_name=self.experiment_name,
            based_on=self.based_on
        )
        if self.based_on in Label.BASED_ON_VALIDATION:
            users_preferences = self.dataset.get_full_train_transactions(
                fold=self.fold, trial=self.trial
            )
        else:
            users_preferences = self.dataset.get_train_transactions(
                fold=self.fold, trial=self.trial
            )
        self.recommender.fit(
            PandasSurprise.pandas_transform_trainset_to_surprise(users_preferences)
        )

        # Load test data
        logger.info(">>> Get the test set")
        self.all_items = self.dataset.get_items()
        self.all_items_ids = self.all_items['ITEM_ID'].unique().tolist()
        user_list = users_preferences[Label.USER_ID].unique()

        # Predict the recommendation list
        logger.info(">>> Predicting...")
        result_list = [self.__predict(
            user_test_set=self._user_unknown_items(
                users_preferences[users_preferences[Label.USER_ID] == user_id],
                user_id=user_id
            )
        ) for user_id in user_list]
        merged_results_df = pd.concat(result_list)

        # Save all recommendation lists
        logger.info(">>> Saving...")
        SaveAndLoad.save_candidate_items(
            experiment_name=self.experiment_name, based_on=self.based_on,
            data=merged_results_df,
            dataset=self.dataset.system_name, algorithm=self.recommender_name,
            fold=self.fold, trial=self.trial
        )
