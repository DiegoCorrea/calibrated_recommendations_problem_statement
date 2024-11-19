"""
Pierre in frame searches
"""
import itertools
import random
from copy import deepcopy
from statistics import mean

import threadpoolctl
from joblib import Parallel, delayed
from recommender_pierre.autoencoders.CDAEModel import CDAEModel
from recommender_pierre.autoencoders.DeppAutoEncModel import DeppAutoEncModel
from recommender_pierre.autoencoders.EASEModel import EASEModel
# from recommender_pierre.bpr.BPRGRAPH import BPRGRAPH
# from recommender_pierre.bpr.BPRKNN import BPRKNN
from scikit_pierre.metrics.evaluation import MeanAveragePrecision, MeanReciprocalRank

from searches.base_search import BaseSearch
from searches.parameters import ImplicitParams
from searches.parameters import PierreParams
from settings.labels import Label


class PierreGridSearch(BaseSearch):
    """
    Class for performing pierre grid search
    """

    def __init__(
            self, experiment_name: str,
            algorithm: str,
            dataset_name: str, trial: int = 1, fold: int = 3,
            n_jobs: int = 1, n_threads: int = 1, list_size: int = 10, n_inter: int = 50,
            split_methodology: str = "CVTT", multiprocessing_lib: str = Label.JOBLIB
    ):
        """
        Parameters
        """
        global OPENBLAS_NUM_THREADS
        OPENBLAS_NUM_THREADS = n_threads
        threadpoolctl.threadpool_limits(n_threads, "blas")
        super().__init__(
            algorithm=algorithm, dataset_name=dataset_name, trial=trial, fold=fold,
            n_jobs=n_jobs, list_size=list_size, n_inter=n_inter,
            split_methodology=split_methodology,
            experiment_name=experiment_name, multiprocessing_lib=multiprocessing_lib
        )
        self.count = 0

    @staticmethod
    def fit_ease(
            lambda_: float, implicit: bool, train_list: list, valid_list: list,
            split_methodology, experiment_name, algorithm, dataset_name
    ):
        """
        Fits the pierre grid search algorithm to the training set and testing set.
        """
        map_value = []
        mrr_value = []

        for train, test in zip(train_list, valid_list):
            recommender = EASEModel(
                lambda_=lambda_, implicit=implicit
            )

            mapv, mrrv = PierreGridSearch.__fit_and_metric(recommender, train, test)
            map_value.append(mapv)
            mrr_value.append(mrrv)

        params = {
            "map": mean(map_value),
            "mrr": mean(mrr_value),
            "params": {
                "lambda_": lambda_,
                "implicit": implicit
            }
        }
        PierreGridSearch.defining_metric_and_save_during_run(
            dataset_name=dataset_name, algorithm=algorithm, params=params,
            split_methodology=split_methodology, experiment_name=experiment_name
        )

    @staticmethod
    def fit_bpr(
            factors, regularization, learning_rate, iterations, random_state,
            train_list, valid_list, list_size,
            split_methodology, experiment_name, algorithm, dataset_name
    ):
        """
        Fits the pierre grid search algorithm to the training set and testing set.
        """
        pass
        # map_value = []
        # mrr_value = []
        #
        # for train, test in zip(train_list, valid_list):
        #     recommender = BPRKNN(
        #         factors=factors, regularization=regularization, learning_rate=learning_rate,
        #         iterations=iterations, seed=random_state, list_size=list_size
        #     )
        #
        #     mapv, mrrv = PierreGridSearch.__fit_and_metric(recommender, train, test)
        #     map_value.append(mapv)
        #     mrr_value.append(mrrv)
        #
        # params = {
        #     "map": mean(map_value),
        #     "mrr": mean(mrr_value),
        #     "params": {
        #         "factors": factors,
        #         "regularization": regularization,
        #         "learning_rate": learning_rate,
        #         "iterations": iterations,
        #         "random_state": random_state
        #     }
        # }
        # PierreGridSearch.defining_metric_and_save_during_run(
        #     dataset_name=dataset_name, algorithm=algorithm, params=params,
        #     split_methodology=split_methodology, experiment_name=experiment_name
        # )

    @staticmethod
    def fit_bpr_graph(
            factors, learning_rate, iterations, lambda_bias, lambda_user, lambda_item,
            train_list, valid_list, list_size,
            split_methodology, experiment_name, algorithm, dataset_name
    ):
        """
        Fits the pierre grid search algorithm to the training set and testing set.
        """
        pass
        # map_value = []
        # mrr_value = []
        #
        # for train, test in zip(train_list, valid_list):
        #     recommender = BPRGRAPH(
        #         factors=factors, learning_rate=learning_rate,
        #         iterations=iterations, list_size=list_size,
        #         lambda_bias=lambda_bias, lambda_user=lambda_user, lambda_item=lambda_item
        #     )
        #     mapv, mrrv = PierreGridSearch.__fit_and_metric(recommender, train, test)
        #     map_value.append(mapv)
        #     mrr_value.append(mrrv)
        #
        # params = {
        #     "map": mean(map_value),
        #     "mrr": mean(mrr_value),
        #     "params": {
        #         "factors": factors,
        #         "lambda_user": lambda_user,
        #         "lambda_item": lambda_item,
        #         "lambda_bias": lambda_bias,
        #         "learning_rate": learning_rate,
        #         "iterations": iterations
        #     }
        # }
        # PierreGridSearch.defining_metric_and_save_during_run(
        #     dataset_name=dataset_name, algorithm=algorithm, params=params,
        #     split_methodology=split_methodology, experiment_name=experiment_name
        # )

    def print_run(self):
        self.count += 1
        print("*" * 50)
        print(self.count)
        print("*" * 50)

    @staticmethod
    def fit_autoencoders(
            algorithm, factors, epochs, dropout, lr, reg, train_list, valid_list,
            split_methodology, experiment_name, dataset_name
    ):
        """
        Fits the pierre grid search algorithm to the training set and testing set.
        """
        map_value = []
        mrr_value = []

        for train, test in zip(train_list, valid_list):
            if algorithm == Label.DEEP_AE:
                recommender = DeppAutoEncModel(
                    factors=int(factors), epochs=int(epochs), dropout=int(dropout), lr=int(lr),
                    reg=int(reg),
                    batch=64
                )
            else:
                recommender = CDAEModel(
                    factors=int(factors), epochs=int(epochs), dropout=int(dropout), lr=int(lr),
                    reg=int(reg),
                    batch=64
                )
            mapv, mrrv = PierreGridSearch.__fit_and_metric(recommender, train, test)
            map_value.append(mapv)
            mrr_value.append(mrrv)

        params = {
            "map": mean(map_value),
            "mrr": mean(mrr_value),
            "params": {
                "factors": factors,
                "epochs": epochs,
                "dropout": dropout,
                "lr": lr,
                "reg": reg
            }
        }
        PierreGridSearch.defining_metric_and_save_during_run(
            dataset_name=dataset_name, algorithm=algorithm, params=params,
            split_methodology=split_methodology, experiment_name=experiment_name
        )

    @staticmethod
    def __fit_and_metric(recommender, train, test):
        """
        Fits the pierre grid search algorithm to the training set and testing set.
        """
        rec_lists_df = recommender.train_and_produce_rec_list(
            user_transactions_df=deepcopy(train)
        )
        map_metric_instance = MeanAveragePrecision(
            users_rec_list_df=rec_lists_df,
            users_test_set_df=test
        )
        mrr_metric_instance = MeanReciprocalRank(
            users_rec_list_df=rec_lists_df,
            users_test_set_df=test
        )
        return map_metric_instance.compute(), mrr_metric_instance.compute()

    def get_params_dae(self):
        """
        Returns the parameters of the pierre grid search algorithm.
        """
        param_distributions = PierreParams.DAE_PARAMS
        combination = list(itertools.product(*[
            param_distributions['factors'], param_distributions['epochs'],
            param_distributions['dropout'], param_distributions['lr'],
            param_distributions['reg']
        ]))

        if self.n_inter < len(combination):
            params_to_use = random.sample(combination, self.n_inter)
        else:
            params_to_use = combination

        return params_to_use

    def get_params_ease(self):
        """
        Returns the parameters of the pierre grid search algorithm.
        """
        param_distributions = PierreParams.EASE_PARAMS
        combination = list(itertools.product(*[
            param_distributions['lambda_'], param_distributions['implicit'],
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

    def get_bpr_graph_params(self):
        param_distributions = PierreParams.BPR_PARAMS

        combination = list(itertools.product(*[
            param_distributions['factors'], param_distributions['learning_rate'],
            param_distributions['lambda_item'], param_distributions['lambda_user'],
            param_distributions['lambda_bias'],
            param_distributions['iterations']
        ]))
        if self.n_inter < int(len(combination)):
            params_to_use = random.sample(combination, self.n_inter)
        else:
            params_to_use = combination
        return params_to_use

    def preparing_recommenders(self):
        if self.algorithm in Label.ENCODERS_RECOMMENDERS:
            params_to_use = self.get_params_dae()
            print("Total of combinations: ", str(len(params_to_use)))

            Parallel(n_jobs=self.n_jobs, verbose=100)(
                delayed(PierreGridSearch.fit_autoencoders)(
                    algorithm=self.algorithm, factors=factors, epochs=epochs,
                    dropout=dropout, lr=lr, reg=reg,
                    train_list=deepcopy(self.train_list),
                    valid_list=deepcopy(self.valid_list),
                    experiment_name=deepcopy(self.experiment_name),
                    dataset_name=deepcopy(self.dataset.system_name)
                ) for factors, epochs, dropout, lr, reg in params_to_use
            )
        elif self.algorithm in Label.EASE_RECOMMENDERS:
            params_to_use = self.get_params_ease()
            print("Total of combinations: ", str(len(params_to_use)))

            Parallel(n_jobs=self.n_jobs, verbose=100)(
                delayed(PierreGridSearch.fit_ease)(
                    lambda_=lambda_, implicit=implicit,
                    train_list=deepcopy(self.train_list),
                    valid_list=deepcopy(self.valid_list),
                    experiment_name=deepcopy(self.experiment_name),
                    algorithm=deepcopy(self.algorithm),
                    dataset_name=deepcopy(self.dataset.system_name)
                ) for lambda_, implicit in params_to_use
            )
        elif self.algorithm in Label.BPRGRAPH:
            params_to_use = self.get_bpr_graph_params()
            print("Total of combinations: ", str(len(params_to_use)))

            Parallel(n_jobs=self.n_jobs, verbose=100)(
                delayed(PierreGridSearch.fit_bpr_graph)(
                    factors=int(factors),
                    lambda_item=float(lambda_item),
                    lambda_user=float(lambda_user),
                    lambda_bias=float(lambda_bias),
                    learning_rate=int(learning_rate), iterations=int(iterations),
                    train_list=deepcopy(self.train_list),
                    valid_list=deepcopy(self.valid_list),
                    list_size=self.list_size,
                    experiment_name=deepcopy(self.experiment_name),
                    algorithm=deepcopy(self.algorithm),
                    dataset_name=deepcopy(self.dataset.system_name)
                ) for factors, learning_rate, lambda_item, lambda_user, lambda_bias, iterations
                in params_to_use
            )
        else:
            params_to_use = self.get_bpr_params()
            print("Total of combinations: ", str(len(params_to_use)))

            Parallel(n_jobs=self.n_jobs, verbose=100)(
                delayed(PierreGridSearch.fit_bpr)(
                    factors=factors, regularization=regularization,
                    learning_rate=learning_rate, iterations=iterations,
                    random_state=random_state,
                    train_list=deepcopy(self.train_list),
                    valid_list=deepcopy(self.valid_list),
                    list_size=self.list_size,
                    experiment_name=deepcopy(self.experiment_name),
                    algorithm=deepcopy(self.algorithm),
                    dataset_name=deepcopy(self.dataset.system_name)
                ) for factors, regularization, learning_rate, iterations, random_state, num_threads
                in params_to_use
            )
