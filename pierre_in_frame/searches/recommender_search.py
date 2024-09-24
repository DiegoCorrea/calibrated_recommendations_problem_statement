import multiprocessing

import itertools
import logging
import random
from copy import deepcopy

import pandas as pd
import threadpoolctl
from joblib import Parallel, delayed
from numpy import mean, ceil
from scikit_pierre.metrics.evaluation import MeanAveragePrecision, MeanReciprocalRank
from surprise import SVD, KNNBasic
from surprise.model_selection import RandomizedSearchCV
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.prediction_algorithms.matrix_factorization import SVDpp, NMF
from tqdm import tqdm

from datasets.registred_datasets import RegisteredDataset
from processing.conversions.pandas_surprise import PandasSurprise
from searches.base_search import BaseSearch
from searches.parameters import SurpriseParams
from settings.constants import Constants
from settings.labels import Label
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class RecommenderSearch:
    """
    Class used to lead with the Random Search
    """

    def __init__(
            self, experiment_name: str, based_on: str, recommender: str, dataset: str, trial: int = None, fold: int = None,
            n_inter: int = Constants.N_INTER, n_jobs: int = Constants.N_CORES,
            n_cv: int = Constants.K_FOLDS_VALUE
    ):
        self.measures = ['rmse', 'mae', 'fcp', 'mse']
        self.experiment_name = experiment_name
        self.based_on = based_on
        self.trial = trial
        self.fold = fold
        self.n_inter = n_inter
        self.n_jobs = n_jobs
        self.n_cv = n_cv
        self.dataset = RegisteredDataset.load_dataset(dataset)
        self.recommender_name = recommender
        self.recommender = None
        self.params = None
        if recommender == Label.SVD:
            self.recommender = SVD
            self.params = SurpriseParams.SVD_SEARCH_PARAMS
        elif recommender == Label.NMF:
            self.recommender = NMF
            self.params = SurpriseParams.NMF_SEARCH_PARAMS
        elif recommender == Label.CO_CLUSTERING:
            self.recommender = CoClustering
            self.params = SurpriseParams.CLUSTERING_SEARCH_PARAMS
        elif recommender == Label.ITEM_KNN_BASIC:
            self.recommender = KNNBasic
            self.params = SurpriseParams.ITEM_KNN_SEARCH_PARAMS
        elif recommender == Label.USER_KNN_BASIC:
            self.recommender = KNNBasic
            self.params = SurpriseParams.USER_KNN_SEARCH_PARAMS
        else:
            self.recommender = SVDpp
            self.params = SurpriseParams.SVDpp_SEARCH_PARAMS

    def __search(self):
        """
        Randomized Search Cross Validation to get the best params in the recommender algorithm
        :return: A Random Search instance
        """
        gs = RandomizedSearchCV(
            algo_class=self.recommender, param_distributions=self.params, measures=self.measures,
            n_iter=self.n_inter, cv=self.n_cv,
            n_jobs=self.n_jobs, joblib_verbose=100, random_state=42
        )
        gs.fit(
            PandasSurprise.pandas_transform_all_dataset_to_surprise(
                self.dataset.get_full_train_transactions(trial=self.trial, fold=self.fold)
            )
        )
        return gs

    def fit(self) -> None:
        """
        Search and save the best param values
        """
        gs = self.__search()
        # Saving
        SaveAndLoad.save_hyperparameters_recommender(
            best_params=gs.best_params,
            dataset=self.dataset.system_name, algorithm=self.recommender_name,
            experiment_name=self.experiment_name, based_on=self.based_on
        )


class SurpriseRandomSearch(BaseSearch):

    def __init__(
            self,
            experiment_name: str, algorithm: str,
            dataset_name: str, trial: int = 1, fold: int = 1,
            n_jobs: int = 1, list_size: int = 10, n_inter: int = 50,
            based_on: str = "RANDOM", multiprocessing_lib: str = Label.JOBLIB
    ):
        super().__init__(
            algorithm=algorithm, dataset_name=dataset_name, trial=trial, fold=fold,
            n_jobs=n_jobs, list_size=list_size, n_inter=n_inter, based_on=based_on,
            experiment_name=experiment_name, multiprocessing_lib=multiprocessing_lib
        )

    @staticmethod
    def _user_unknown_items(
            all_items_ids: list,
            users_preferences: pd.DataFrame, user_id: str
    ) -> pd.DataFrame:
        """
        TODO: Docstring
        """
        user_unknown_items_ids = set(
            all_items_ids) - set(users_preferences['ITEM_ID'].unique().tolist())
        unk_df = pd.DataFrame()
        unk_df[Label.ITEM_ID] = list(user_unknown_items_ids)
        unk_df[Label.USER_ID] = user_id
        unk_df[Label.TRANSACTION_VALUE] = 0
        return unk_df

    @staticmethod
    def __predict(recommender, user_test_set: pd.DataFrame, list_size: int) -> pd.DataFrame:
        """
        Method to predict the rating to a user.

        :param user_test_set: A Pandas Dataframe with the user_id and item_id.
        :return: A Pandas Dataframe with the user_id, item_id and predicted_rating.
        """
        # Transform the pandas dataframe in a surprise dataset structure
        testset = PandasSurprise.pandas_transform_testset_to_surprise(testset_df=user_test_set)
        # Predict and transform surprise dataset structure in a pandas dataframe
        return PandasSurprise.surprise_to_pandas_get_candidates_items(
            predictions=recommender.test(testset=testset),
            n=list_size
        )

    @staticmethod
    def __make_recommendation__(
            user_list, all_items_ids, recommender, users_preferences, list_size, progress
    ):
        # Predict the recommendation list
        result_list = [SurpriseRandomSearch.__predict(
            user_test_set=SurpriseRandomSearch._user_unknown_items(
                users_preferences=users_preferences[users_preferences[Label.USER_ID] == user_id],
                user_id=user_id,
                all_items_ids=all_items_ids
            ),
            list_size=list_size,
            recommender=recommender
        ) for user_id in user_list]
        progress.update(len(user_list))
        progress.set_description("Recommendation: ")

        return result_list

    @staticmethod
    def __run__(recommender, users_preferences, list_size, item_df):
        """
        Method to run the recommender algorithm, made and save the recommendation list
        """
        # fit the recommender algorithm
        print("Training")
        recommender.fit(
            PandasSurprise.pandas_transform_trainset_to_surprise(users_preferences)
        )

        # Load test data
        all_items_ids = item_df['ITEM_ID'].unique().tolist()
        user_list = users_preferences[Label.USER_ID].unique()

        progress = tqdm(total=len(user_list))
        loops = int(ceil(len(user_list)/100))

        user_preds = [pd.concat(
            SurpriseRandomSearch.__make_recommendation__(
                user_list=user_list[i * 100: (i + 1) * 100],
                all_items_ids=all_items_ids,
                recommender=recommender,
                users_preferences=users_preferences,
                list_size=list_size,
                progress=progress
            )
        ) for i in range(0, loops)]

        progress.close()
        return pd.concat(user_preds)

    @staticmethod
    def fit_nmf(
            n_factors, n_epochs, reg_pu, reg_qi, reg_bu, reg_bi, lr_bu, lr_bi,
            random_state,
            train_list, valid_list, item_df, list_size
    ):
        map_value = []
        mrr_value = []

        for train, validation in zip(train_list, valid_list):
            recommender = NMF(
                n_factors=n_factors, n_epochs=n_epochs, reg_pu=reg_pu, reg_qi=reg_qi,
                reg_bu=reg_bu, reg_bi=reg_bi, lr_bu=lr_bu, lr_bi=lr_bi,
                random_state=random_state
            )
            rec_lists_df = SurpriseRandomSearch.__run__(
                recommender=recommender, users_preferences=train, list_size=list_size,
                item_df=item_df
            )
            metric_instance = MeanAveragePrecision(
                users_rec_list_df=rec_lists_df,
                users_test_set_df=validation
            )
            mrr_metric_instance = MeanReciprocalRank(
                users_rec_list_df=rec_lists_df,
                users_test_set_df=validation
            )
            map_value.append(metric_instance.compute())
            mrr_value.append(mrr_metric_instance.compute())

        return {
            "map": mean(map_value),
            "mrr": mean(mrr_value),
            "params": {
                "n_factors": n_factors,
                "n_epochs": n_epochs,
                "reg_pu": reg_pu,
                "reg_qi": reg_qi,
                "reg_bu": reg_bu,
                "reg_bi": reg_bi,
                "lr_bu": lr_bu,
                "lr_bi": lr_bi,
                "random_state": random_state
            }
        }

    @staticmethod
    def fit_svd(
            n_factors, n_epochs, lr_all, reg_all, random_state,
            train_list, valid_list, item_df, list_size
    ):
        map_value = []
        mrr_value = []

        for train, validation in zip(train_list, valid_list):
            recommender = SVD(
                n_factors=n_factors, n_epochs=n_epochs, reg_all=reg_all,
                lr_all=lr_all, random_state=random_state
            )
            rec_lists_df = SurpriseRandomSearch.__run__(
                recommender=recommender, users_preferences=train, list_size=list_size,
                item_df=item_df
            )
            metric_instance = MeanAveragePrecision(
                users_rec_list_df=rec_lists_df,
                users_test_set_df=validation
            )
            mrr_metric_instance = MeanReciprocalRank(
                users_rec_list_df=rec_lists_df,
                users_test_set_df=validation
            )
            map_value.append(metric_instance.compute())
            mrr_value.append(mrr_metric_instance.compute())

        return {
            "map": mean(map_value),
            "mrr": mean(mrr_value),
            "params": {
                "n_factors": n_factors,
                "n_epochs": n_epochs,
                "lr_all": lr_all,
                "reg_all": reg_all,
                "random_state": random_state
            }
        }

    def get_nmf_params(self):
        param_distributions = SurpriseParams.NMF_SEARCH_PARAMS

        combination = list(itertools.product(*[
            param_distributions['n_factors'], param_distributions['n_epochs'],
            param_distributions['reg_pu'], param_distributions['reg_qi'],
            param_distributions['reg_bu'], param_distributions['reg_bi'],
            param_distributions['lr_bu'], param_distributions['lr_bi'],
            param_distributions['random_state']
        ]))
        if self.n_inter < len(combination):
            params_to_use = random.sample(combination, self.n_inter)
        else:
            params_to_use = combination
        return params_to_use

    def get_svd_params(self):
        param_distributions = SurpriseParams.SVD_SEARCH_PARAMS

        combination = list(itertools.product(*[
            param_distributions['n_factors'], param_distributions['n_epochs'],
            param_distributions['lr_all'], param_distributions['reg_all'],
            param_distributions['random_state']
        ]))
        if self.n_inter < int(len(combination)):
            params_to_use = random.sample(combination, self.n_inter)
        else:
            params_to_use = combination
        return params_to_use

    def preparing_recommenders(self):
        if self.algorithm == Label.NMF:
            params_to_use = self.get_nmf_params()
            print("Total of combinations: ", str(len(params_to_use)))
            if self.multiprocessing_lib == Label.JOBLIB:
                # Starting the recommender algorithm
                self.output = list(Parallel(n_jobs=self.n_jobs, verbose=100)(
                    delayed(SurpriseRandomSearch.fit_nmf)(
                        n_factors=n_factors, n_epochs=n_epochs, reg_pu=reg_pu,
                        reg_qi=reg_qi, reg_bu=reg_bu, reg_bi=reg_bi, lr_bu=lr_bu, lr_bi=lr_bi,
                        random_state=random_state,
                        train_list=deepcopy(self.train_list),
                        valid_list=deepcopy(self.valid_list),
                        item_df=deepcopy(self.item_df),
                        list_size=deepcopy(self.list_size)
                    ) for n_factors, n_epochs, reg_pu, reg_qi, reg_bu, reg_bi, lr_bu, lr_bi, random_state in
                    params_to_use
                ))
            else:
                process_args = []
                for n_factors, n_epochs, reg_pu, reg_qi, reg_bu, reg_bi, lr_bu, lr_bi, random_state in params_to_use:
                    process_args.append((
                        n_factors, n_epochs, reg_pu, reg_qi, reg_bu, reg_bi, lr_bu, lr_bi, random_state,
                        deepcopy(self.train_list), deepcopy(self.valid_list),
                        deepcopy(self.item_df), deepcopy(self.list_size)
                    ))
                pool = multiprocessing.Pool(processes=self.n_jobs)
                self.output = list(pool.starmap(SurpriseRandomSearch.fit_nmf, process_args))
                pool.close()
                pool.join()

        elif self.algorithm == Label.SVD:
            params_to_use = self.get_svd_params()
            print("Total of combinations: ", str(len(params_to_use)))

            if self.multiprocessing_lib == Label.JOBLIB:
                # Starting the recommender algorithm
                self.output = list(Parallel(n_jobs=self.n_jobs, verbose=100, batch_size=128)(
                    delayed(SurpriseRandomSearch.fit_svd)(
                        n_factors=n_factors, n_epochs=n_epochs,
                        lr_all=lr_all, reg_all=reg_all,
                        random_state=random_state,
                        train_list=deepcopy(self.train_list),
                        valid_list=deepcopy(self.valid_list),
                        item_df=deepcopy(self.item_df),
                        list_size=deepcopy(self.list_size)
                    ) for n_factors, n_epochs, lr_all, reg_all, random_state
                    in params_to_use
                ))
            else:
                process_args = []
                for n_factors, n_epochs, lr_all, reg_all, random_state in params_to_use:
                    process_args.append((
                        n_factors, n_epochs, lr_all, reg_all, random_state,
                        deepcopy(self.train_list), deepcopy(self.valid_list),
                        deepcopy(self.item_df), deepcopy(self.list_size)
                    ))
                pool = multiprocessing.Pool(processes=self.n_jobs)
                self.output = list(pool.starmap(SurpriseRandomSearch.fit_svd, process_args))
                pool.close()
                pool.join()
        else:
            pass
