import itertools
import multiprocessing
import random
from copy import deepcopy
from statistics import mean

import implicit
import pandas as pd
import threadpoolctl
from joblib import Parallel, delayed
from scikit_pierre.metrics.evaluation import MeanAveragePrecision, MeanReciprocalRank
from scipy import sparse

from searches.base_search import BaseSearch
from searches.parameters import ImplicitParams
from settings.labels import Label


class ImplicitGridSearch(BaseSearch):

    def __init__(
            self, experiment_name: str,
            algorithm: str,
            dataset_name: str, trial: int = 1, fold: int = 1,
            n_jobs: int = 1, n_threads: int = 1, list_size: int = 10, n_inter: int = 50,
            split_methodology: str = "RANDOM", multiprocessing_lib: str = Label.JOBLIB
    ):
        global OPENBLAS_NUM_THREADS
        OPENBLAS_NUM_THREADS = n_threads
        threadpoolctl.threadpool_limits(n_threads, "blas")
        super().__init__(
            algorithm=algorithm, dataset_name=dataset_name, trial=trial, fold=fold,
            n_jobs=n_jobs, list_size=list_size, n_inter=n_inter,
            split_methodology=split_methodology,
            experiment_name=experiment_name
        )

    @staticmethod
    def __predict(user_preferences: pd.DataFrame, user_id, recommender, list_size) -> pd.DataFrame:
        """
        Method to predict the rating to a user.

        :param user_preferences: A Pandas Dataframe with the user_id and item_id.
        :return: A Pandas Dataframe with the user_id, item_id and predicted_rating.
        """

        ids, scores = recommender.recommend(
            user_id, user_preferences, N=list_size, filter_already_liked_items=True
        )
        df = pd.DataFrame([], columns=[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE])
        df[Label.ITEM_ID] = ids.tolist()
        df[Label.TRANSACTION_VALUE] = scores.tolist()
        df[Label.USER_ID] = user_id
        return df

    @staticmethod
    def __run__(recommender, users_preferences, list_size):
        """
        Method to run the recommender algorithm, made and save the recommendation list
        """
        # fit the recommender algorithm
        sparse_customer_item = sparse.csr_matrix(
            (
                users_preferences[Label.TRANSACTION_VALUE].astype(float),
                (users_preferences[Label.USER_ID], users_preferences[Label.ITEM_ID]),
            )
        )

        recommender.fit(sparse_customer_item)

        user_list = users_preferences[Label.USER_ID].unique()

        # Predict the recommendation list
        result_list = [ImplicitGridSearch.__predict(
            user_preferences=sparse_customer_item[user_id],
            user_id=user_id,
            recommender=recommender,
            list_size=list_size
        ) for user_id in user_list]
        return pd.concat(result_list)

    @staticmethod
    def fit_als(
            factors, regularization, alpha, iterations, random_state,
            train_list, valid_list, list_size,
            split_methodology, experiment_name, algorithm, dataset_name
    ):
        map_value = []
        mrr_value = []

        for train, validation in zip(train_list, valid_list):
            recommender = implicit.als.AlternatingLeastSquares(
                factors=factors, regularization=regularization, alpha=alpha, iterations=iterations,
                random_state=random_state, num_threads=1
            )
            rec_lists_df = ImplicitGridSearch.__run__(
                recommender=recommender, users_preferences=train, list_size=list_size
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
        print(f"map list: {map_value}\n"
              f"mrr list: {mrr_value}")

        params = {
            "map": mean(map_value),
            "mrr": mean(mrr_value),
            "params": {
                "factors": factors,
                "regularization": regularization,
                "alpha": alpha,
                "iterations": iterations,
                "random_state": random_state
            }
        }
        ImplicitGridSearch.defining_metric_and_save_during_run(
            dataset_name=dataset_name, algorithm=algorithm, params=params,
            split_methodology=split_methodology, experiment_name=experiment_name
        )

    @staticmethod
    def fit_bpr(
            factors, regularization, learning_rate, iterations, random_state,
            train_list, valid_list, list_size,
            split_methodology, experiment_name, algorithm, dataset_name
    ):
        map_value = []
        mrr_value = []

        for train, validation in zip(train_list, valid_list):
            recommender = implicit.bpr.BayesianPersonalizedRanking(
                factors=factors, regularization=regularization, learning_rate=learning_rate,
                iterations=iterations, random_state=random_state, num_threads=1
            )
            rec_lists_df = ImplicitGridSearch.__run__(
                recommender=recommender, users_preferences=train, list_size=list_size
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
        print(f"map list: {map_value}\n"
              f"mrr list: {mrr_value}")

        params = {
            "map": mean(map_value),
            "mrr": mean(mrr_value),
            "params": {
                "factors": factors,
                "regularization": regularization,
                "learning_rate": learning_rate,
                "iterations": iterations,
                "random_state": random_state
            }
        }
        ImplicitGridSearch.defining_metric_and_save_during_run(
            dataset_name=dataset_name, algorithm=algorithm, params=params,
            split_methodology=split_methodology, experiment_name=experiment_name
        )

    @staticmethod
    def fit_item_knn(
            k, train_list, valid_list, list_size,
            split_methodology, experiment_name, algorithm, dataset_name
    ):
        map_value = []
        mrr_value = []

        for train, validation in zip(train_list, valid_list):
            recommender = implicit.nearest_neighbours.ItemItemRecommender(
                K=k
            )
            rec_lists_df = ImplicitGridSearch.__run__(
                recommender=recommender, users_preferences=train, list_size=list_size
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
        print(f"map list: {map_value}\n"
              f"mrr list: {mrr_value}")

        params = {
            "map": mean(map_value),
            "mrr": mean(mrr_value),
            "params": {
                "K": k
            }
        }
        ImplicitGridSearch.defining_metric_and_save_during_run(
            dataset_name=dataset_name, algorithm=algorithm, params=params,
            split_methodology=split_methodology, experiment_name=experiment_name
        )

    @staticmethod
    def fit_bm25(
            k, k1, b, train_list, valid_list, list_size,
            split_methodology, experiment_name, algorithm, dataset_name
    ):
        map_value = []
        mrr_value = []

        for train, validation in zip(train_list, valid_list):
            recommender = implicit.nearest_neighbours.BM25Recommender(
                K=k, K1=k1, B=b
            )
            rec_lists_df = ImplicitGridSearch.__run__(
                recommender=recommender, users_preferences=train, list_size=list_size
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
        print(f"map list: {map_value}\n"
              f"mrr list: {mrr_value}")

        params = {
            "map": mean(map_value),
            "mrr": mean(mrr_value),
            "params": {
                "K": k,
                "K1": k1,
                "B": b
            }
        }
        ImplicitGridSearch.defining_metric_and_save_during_run(
            dataset_name=dataset_name, algorithm=algorithm, params=params,
            split_methodology=split_methodology, experiment_name=experiment_name
        )

    def get_als_params(self):
        param_distributions = ImplicitParams.ALS_PARAMS

        combination = list(itertools.product(*[
            param_distributions['factors'], param_distributions['regularization'],
            param_distributions['alpha'], param_distributions['iterations'],
            param_distributions['random_state'], param_distributions['num_threads'],
        ]))
        if self.n_inter < len(combination):
            params_to_use = random.sample(combination, self.n_inter)
        else:
            params_to_use = combination
        return params_to_use

    def get_bpr_params(self):
        param_distributions = ImplicitParams.BPR_PARAMS

        combination = list(itertools.product(*[
            param_distributions['factors'], param_distributions['regularization'],
            param_distributions['learning_rate'], param_distributions['iterations'],
            param_distributions['random_state'], param_distributions['num_threads'],
        ]))
        if self.n_inter < int(len(combination)):
            params_to_use = random.sample(combination, self.n_inter)
        else:
            params_to_use = combination
        return params_to_use

    def get_item_knn_params(self):
        param_distributions = ImplicitParams.ITEM_KNN_SEARCH_PARAMS

        combination = list(param_distributions['k'])
        if self.n_inter < int(len(combination)):
            params_to_use = random.sample(combination, self.n_inter)
        else:
            params_to_use = combination
        return params_to_use

    def get_bm25_params(self):
        param_distributions = ImplicitParams.BM25_SEARCH_PARAMS

        combination = list(itertools.product(*[
            param_distributions['k'],
            param_distributions['k1'],
            param_distributions['b']
        ]))
        if self.n_inter < int(len(combination)):
            params_to_use = random.sample(combination, self.n_inter)
        else:
            params_to_use = combination
        return params_to_use

    def preparing_recommenders(self):
        if self.algorithm == Label.ALS:
            params_to_use = self.get_als_params()
            print("Total of combinations: ", str(len(params_to_use)))
            if self.multiprocessing_lib == Label.JOBLIB:
                # Starting the recommender algorithm
                Parallel(n_jobs=self.n_jobs)(
                    delayed(ImplicitGridSearch.fit_als)(
                        factors=factors, regularization=regularization, alpha=alpha,
                        iterations=iterations,
                        random_state=random_state,
                        train_list=deepcopy(self.train_list),
                        valid_list=deepcopy(self.valid_list),
                        list_size=self.list_size,
                        experiment_name=deepcopy(self.experiment_name),
                        algorithm=deepcopy(self.algorithm),
                        dataset_name=deepcopy(self.dataset.system_name)
                    ) for factors, regularization, alpha, iterations, random_state, num_threads in
                    params_to_use
                )
            else:
                process_args = []
                for factors, regularization, alpha, iterations, random_state, num_threads in params_to_use:
                    process_args.append((
                        factors, regularization, alpha, iterations, random_state,
                        deepcopy(self.train_list), deepcopy(self.valid_list), self.list_size,
                        deepcopy(self.split_methodology), deepcopy(self.experiment_name),
                        deepcopy(self.algorithm), deepcopy(self.dataset.system_name)
                    ))
                pool = multiprocessing.Pool(processes=self.n_jobs)
                pool.starmap(ImplicitGridSearch.fit_als, process_args)
                pool.close()
                pool.join()

        elif self.algorithm == Label.BPR:
            params_to_use = self.get_bpr_params()
            print("Total of combinations: ", str(len(params_to_use)))
            if self.multiprocessing_lib == Label.JOBLIB:

                # Starting the recommender algorithm
                Parallel(n_jobs=self.n_jobs)(
                    delayed(ImplicitGridSearch.fit_bpr)(
                        factors=factors, regularization=regularization,
                        learning_rate=learning_rate, iterations=iterations,
                        random_state=random_state,
                        train_list=deepcopy(self.train_list),
                        valid_list=deepcopy(self.valid_list),
                        list_size=self.list_size,
                        experiment_name=deepcopy(self.experiment_name),
                        algorithm=deepcopy(self.algorithm),
                        dataset_name=deepcopy(self.dataset.system_name)
                    ) for
                    factors, regularization, learning_rate, iterations, random_state, num_threads
                    in params_to_use
                )
            else:
                process_args = []
                for factors, regularization, learning_rate, iterations, random_state, num_threads in params_to_use:
                    process_args.append((
                        factors, regularization, learning_rate, iterations, random_state,
                        deepcopy(self.train_list), deepcopy(self.valid_list), self.list_size,
                        deepcopy(self.split_methodology), deepcopy(self.experiment_name),
                        deepcopy(self.algorithm), deepcopy(self.dataset.system_name)
                    ))
                pool = multiprocessing.Pool(processes=self.n_jobs)
                pool.starmap(ImplicitGridSearch.fit_bpr, process_args)
                pool.close()
                pool.join()
        elif self.algorithm == Label.ITEMKNN:
            params_to_use = self.get_item_knn_params()
            print("Total of combinations: ", str(len(params_to_use)))
            if self.multiprocessing_lib == Label.JOBLIB:

                # Starting the recommender algorithm
                Parallel(n_jobs=self.n_jobs)(
                    delayed(ImplicitGridSearch.fit_item_knn)(
                        k=k,
                        train_list=deepcopy(self.train_list),
                        valid_list=deepcopy(self.valid_list),
                        list_size=self.list_size,
                        experiment_name=deepcopy(self.experiment_name),
                        algorithm=deepcopy(self.algorithm),
                        dataset_name=deepcopy(self.dataset.system_name)
                    ) for k in params_to_use
                )
            else:
                process_args = []
                for k in params_to_use:
                    process_args.append((
                        k,
                        deepcopy(self.train_list), deepcopy(self.valid_list), self.list_size,
                        deepcopy(self.split_methodology), deepcopy(self.experiment_name),
                        deepcopy(self.algorithm), deepcopy(self.dataset.system_name)
                    ))
                pool = multiprocessing.Pool(processes=self.n_jobs)
                pool.starmap(ImplicitGridSearch.fit_item_knn, process_args)
                pool.close()
                pool.join()
        elif self.algorithm == Label.BM25:
            params_to_use = self.get_bm25_params()
            print("Total of combinations: ", str(len(params_to_use)))
            if self.multiprocessing_lib == Label.JOBLIB:

                # Starting the recommender algorithm
                Parallel(n_jobs=self.n_jobs)(
                    delayed(ImplicitGridSearch.fit_bm25)(
                        k=k, k1=k1, b=b,
                        train_list=deepcopy(self.train_list),
                        valid_list=deepcopy(self.valid_list),
                        list_size=self.list_size,
                        experiment_name=deepcopy(self.experiment_name),
                        algorithm=deepcopy(self.algorithm),
                        dataset_name=deepcopy(self.dataset.system_name)
                    ) for k, k1, b in params_to_use
                )
            else:
                process_args = []
                for k, k1, b in params_to_use:
                    process_args.append((
                        k, k1, b,
                        deepcopy(self.train_list), deepcopy(self.valid_list), self.list_size,
                        deepcopy(self.split_methodology), deepcopy(self.experiment_name),
                        deepcopy(self.algorithm), deepcopy(self.dataset.system_name)
                    ))
                pool = multiprocessing.Pool(processes=self.n_jobs)
                pool.starmap(ImplicitGridSearch.fit_bm25, process_args)
                pool.close()
                pool.join()
        else:
            pass
