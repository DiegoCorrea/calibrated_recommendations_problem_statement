import logging
import scipy.sparse as sparse

import pandas as pd
import implicit
import threadpoolctl

from datasets.registred_datasets import RegisteredDataset
from settings.labels import Label
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class ImplicitRecommenderAlgorithm:
    """
    Class to lead with the surprise recommender algorithms,
        generating the recommendation and saving in the results path
    """

    def __init__(
            self,
            experiment_name: str,
            recommender_name: str, dataset_name: str, fold: int, trial: int,
            list_size: int, based_on: str, metric: str = "map"
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
        self.based_on = based_on
        self.experiment_name = experiment_name
        global OPENBLAS_NUM_THREADS
        OPENBLAS_NUM_THREADS = 1
        threadpoolctl.threadpool_limits(1, "blas")

        # Load the surprise recommender algorithm
        full_params = SaveAndLoad.load_hyperparameters_recommender(
            experiment_name=self.experiment_name, based_on=self.based_on,
            dataset=self.dataset.system_name, algorithm=self.recommender_name
        )
        if self.recommender_name == Label.ALS:
            self.recommender = implicit.als.AlternatingLeastSquares(
                factors=int(full_params["params"]["factors"]),
                regularization=float(full_params["params"]["regularization"]),
                alpha=float(full_params["params"]["alpha"]),
                iterations=int(full_params["params"]["iterations"]),
                random_state=int(full_params["params"]["random_state"]), num_threads=1
            )
        elif self.recommender_name == Label.BPR:
            self.recommender = implicit.bpr.BayesianPersonalizedRanking(
                factors=full_params["params"]["factors"],
                regularization=full_params["params"]["regularization"],
                learning_rate=full_params["params"]["learning_rate"],
                iterations=full_params["params"]["iterations"],
                random_state=full_params["params"]["random_state"], num_threads=1
            )
        elif self.recommender_name == Label.ITEMKNN:
            self.recommender = implicit.nearest_neighbours.CosineRecommender(
                K=full_params["params"]["K"]
            )
        elif self.recommender_name == Label.LMF:
            pass
        else:
            pass

    def __predict(self, user_preferences: pd.DataFrame, user_id) -> pd.DataFrame:
        """
        Method to predict the rating to a user.

        :param user_preferences: A Pandas Dataframe with the user_id and item_id.
        :return: A Pandas Dataframe with the user_id, item_id and predicted_rating.
        """

        ids, scores = self.recommender.recommend(
            user_id, user_preferences, N=self.list_size, filter_already_liked_items=True
        )
        df = pd.DataFrame([], columns=[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE])
        df[Label.ITEM_ID] = ids.tolist()
        df[Label.TRANSACTION_VALUE] = scores.tolist()
        df[Label.USER_ID] = user_id
        return df

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

        sparse_customer_item = sparse.csr_matrix(
            (
                users_preferences[Label.TRANSACTION_VALUE].astype(float),
                (users_preferences[Label.USER_ID], users_preferences[Label.ITEM_ID]),
            )
        )
        self.recommender.fit(sparse_customer_item)

        user_list = users_preferences[Label.USER_ID].unique()

        # Predict the recommendation list
        logger.info(">>> Predicting...")
        result_list = [self.__predict(
            user_preferences=sparse_customer_item[user_id],
            user_id=user_id
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
