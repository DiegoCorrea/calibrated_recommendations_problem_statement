import json
import os

from pandas import DataFrame, read_csv

from settings.labels import Label
from settings.path_dir_file import PathDirFile
from utils.utils import NpEncoder


class SaveAndLoad:
    """
    TODO: Docstring
    """

    @staticmethod
    def load_step_file(step: str, file_name: str):
        """
        TODO: Docstring
        """
        path_to_open = PathDirFile.get_step_file(step=step, file_name=file_name)
        with open(path_to_open) as json_file:
            params = json.load(json_file)

        return params

    # ########################################################################################### #
    # ########################################################################################### #
    # ########################################################################################### #
    # [STEP 1] Pre Processing step methods
    # ########################################################################################### #
    # ########################################################################################### #
    # ########################################################################################### #
    @staticmethod
    def save_clean_transactions(experiment_name: str, dataset: str, based_on: str):
        pass

    @staticmethod
    def load_clean_transactions(experiment_name: str, dataset: str, based_on: str):
        clean_dataset_dir = "/".join(
            [PathDirFile.DATA_DIR, experiment_name,
             "datasets", dataset, based_on]
        )
        transactions = read_csv(
            os.path.join(clean_dataset_dir, PathDirFile.TRANSACTIONS_FILE)
        )
        return transactions.astype({
            Label.USER_ID: 'str',
            Label.ITEM_ID: 'str'
        })

    @staticmethod
    def load_train_transactions(
            experiment_name: str, dataset: str, based_on: str, trial: int, fold: int):
        clean_dataset_dir = "/".join([
            PathDirFile.DATA_DIR, experiment_name, "datasets", dataset, based_on,
            "trial-" + str(trial), "fold-" + str(fold)
        ])
        return read_csv(os.path.join(clean_dataset_dir, PathDirFile.TRAIN_FILE))

    @staticmethod
    def load_validation_transactions(
            experiment_name: str, dataset: str, based_on: str, trial: int, fold: int):
        clean_dataset_dir = "/".join([
            PathDirFile.DATA_DIR, experiment_name, "datasets", dataset, based_on,
            "trial-" + str(trial), "fold-" + str(fold)
        ])
        return read_csv(os.path.join(clean_dataset_dir, PathDirFile.VALIDATION_FILE))

    @staticmethod
    def load_test_transactions(
            experiment_name: str, dataset: str, based_on: str, trial: int, fold: int):
        clean_dataset_dir = "/".join([
            PathDirFile.DATA_DIR, experiment_name, "datasets", dataset, based_on,
            "trial-" + str(trial), "fold-" + str(fold)
        ])
        return read_csv(os.path.join(clean_dataset_dir, PathDirFile.TEST_FILE))

    @staticmethod
    def save_clean_items(
            experiment_name: str, dataset: str, based_on: str):
        clean_dataset_dir = "/".join(
            [PathDirFile.DATA_DIR, experiment_name,
             "datasets", dataset, based_on]
        )
        items = read_csv(
            os.path.join(clean_dataset_dir, PathDirFile.ITEMS_FILE)
        )
        return items.astype({
            Label.ITEM_ID: 'str'
        })

    # ########################################################################################### #

    @staticmethod
    def save_user_preference_distribution(
            data: DataFrame, experiment_name: str, dataset: str, based_on: str,
            trial: int, fold: int, distribution: str, ext: str = 'csv'
    ):
        """
        This method is to save the distribution file.
        """
        data.to_csv(
            PathDirFile.preference_distribution_file(
                dataset=dataset, experiment_name=experiment_name, based_on=based_on,
                fold=fold, trial=trial, filename=distribution + '.' + ext
            )
        )

    @staticmethod
    def load_user_preference_distribution(
            experiment_name: str, dataset: str, based_on: str,
            trial: int, fold: int, distribution: str, ext: str = 'csv'
    ) -> DataFrame:
        """
        This method is to load the distribution file.
        """
        preference_distribution_path = PathDirFile.preference_distribution_file(
            dataset=dataset, experiment_name=experiment_name, based_on=based_on,
            fold=fold, trial=trial, filename=distribution + '.' + ext
        )
        return read_csv(preference_distribution_path, index_col=0).fillna(0)

    # ########################################################################################### #

    @staticmethod
    def save_item_class_one_hot_encode(
            data: DataFrame, experiment_name: str, dataset: str, based_on: str, ext: str = 'csv'
    ):
        """
        This method is to save the item one hot encode file.
        """
        data.to_csv(
            PathDirFile.item_class_one_hot_encode_file(
                dataset=dataset, experiment_name=experiment_name, based_on=based_on,
                filename="item_one_hot_encode" + '.' + ext
            ), mode='w+'
        )

    @staticmethod
    def load_item_class_one_hot_encode(
            experiment_name: str, dataset: str, based_on: str, ext: str = 'csv'
    ) -> DataFrame:
        """
        This method is to load the one hot encode file.
        """
        preference_distribution_path = PathDirFile.item_class_one_hot_encode_file(
            dataset=dataset, experiment_name=experiment_name, based_on=based_on,
            filename="item_one_hot_encode" + '.' + ext
        )
        return read_csv(preference_distribution_path, index_col=0)

    # ########################################################################################### #

    @staticmethod
    def save_dataset_analyze(
            data: DataFrame, experiment_name: str, dataset: str, based_on: str, ext: str = 'csv'
    ):
        """
        This method is to save the distribution file.
        """
        data.to_csv(
            PathDirFile.dataset_analyze_file(
                dataset=dataset, experiment_name=experiment_name, based_on=based_on,
                filename="general" + '.' + ext
            ), index=False, mode='w+'
        )

    # ########################################################################################### #
    # ########################################################################################### #
    # ########################################################################################### #
    # [STEP 2] Search step methods - Best Parameters
    # ########################################################################################### #
    # ########################################################################################### #
    # ########################################################################################### #

    @staticmethod
    def save_hyperparameters_recommender(
            best_params: dict, experiment_name: str, dataset: str, based_on: str, algorithm: str):
        """
        TODO: Docstring
        """
        with open(PathDirFile.set_recommender_hyperparameter_file(
                opt=Label.RECOMMENDER, dataset=dataset, algorithm=algorithm,
                experiment_name=experiment_name, based_on=based_on
        ), 'w+') as fp:
            json.dump(best_params, fp, cls=NpEncoder)

    @staticmethod
    def load_hyperparameters_recommender(
            experiment_name: str, dataset: str, based_on: str, algorithm: str):
        """
        TODO: Docstring
        """
        path_to_open = PathDirFile.get_recommender_hyperparameter_file(
            opt=Label.RECOMMENDER, dataset=dataset, algorithm=algorithm,
            experiment_name=experiment_name, based_on=based_on
        )
        with open(path_to_open) as json_file:
            params = json.load(json_file)

        return params

    @staticmethod
    def save_hyperparameters_conformity(
            best_params: dict, experiment_name: str, dataset: str, based_on: str,
            cluster: str, distribution: str):
        """
        TODO: Docstring
        """
        with open(PathDirFile.set_conformity_hyperparameter_file(
                opt=Label.CONFORMITY, dataset=dataset, cluster=cluster,
                distribution=distribution,
                experiment_name=experiment_name, based_on=based_on
        ), 'w') as fp:
            json.dump(best_params, fp)

    @staticmethod
    def load_hyperparameters_conformity(
            experiment_name: str, dataset: str, based_on: str, cluster: str, distribution: str
    ):
        """
        TODO: Docstring
        """
        path_to_open = PathDirFile.get_conformity_hyperparameter_file(
            opt=Label.CONFORMITY, dataset=dataset,
            cluster=cluster, distribution=distribution,
            experiment_name=experiment_name, based_on=based_on
        )
        with open(path_to_open) as json_file:
            params = json.load(json_file)

        return params

    # ########################################################################################### #
    # [STEP 3] Processing step methods - Candidate Items
    # ########################################################################################### #
    @staticmethod
    def save_candidate_items(
            experiment_name: str, based_on: str,
            data: DataFrame, dataset: str, algorithm: str, trial: int, fold: int):
        """
        TODO: Docstring
        """
        data.to_csv(
            PathDirFile.set_candidate_items_file(
                experiment_name=experiment_name, based_on=based_on,
                dataset=dataset, algorithm=algorithm, fold=fold, trial=trial
            ),
            index=False, mode='w+'
        )

    @staticmethod
    def load_candidate_items(
            experiment_name: str, based_on: str,
            dataset: str, algorithm: str, trial: int, fold: int):
        """
        TODO: Docstring
        """
        candidate_items_path = PathDirFile.get_candidate_items_file(
            experiment_name=experiment_name, based_on=based_on,
            dataset=dataset, algorithm=algorithm, fold=fold, trial=trial
        )
        return read_csv(candidate_items_path)

    # ########################################################################################### #
    # [STEP 4] Post-Processing step methods - Recommendation Lists
    # ########################################################################################### #
    @staticmethod
    def load_recommendation_lists(
        experiment_name: str, based_on: str,
        dataset: str, recommender: str, trial: int, fold: int,
        tradeoff: str, distribution: str, fairness: str, relevance: str,
        tradeoff_weight: str, select_item: str
    ):
        """
        TODO: Docstring
        """
        recommendation_list_path = PathDirFile.get_recommendation_list_file(
            experiment_name=experiment_name, based_on=based_on,
            dataset=dataset, recommender=recommender, trial=trial, fold=fold,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, tradeoff_weight=tradeoff_weight, select_item=select_item
        )
        return read_csv(recommendation_list_path)

    @staticmethod
    def save_recommendation_lists(
        experiment_name: str, based_on: str,
        data: DataFrame,
        dataset: str, recommender: str, trial: int, fold: int,
        tradeoff: str, distribution: str, fairness: str, relevance: str,
        tradeoff_weight: str, select_item: str
    ):
        """
        TODO: Docstring
        """
        path = PathDirFile.set_recommendation_list_file(
            experiment_name=experiment_name, based_on=based_on,
            recommender=recommender, dataset=dataset,
            trial=trial, fold=fold,
            tradeoff=tradeoff,
            distribution=distribution,
            fairness=fairness,
            relevance=relevance,
            tradeoff_weight=tradeoff_weight,
            select_item=select_item
        )
        data.to_csv(path, index=False, mode='w+')

    # ########################################################################################### #
    # [STEP 5] Metrics step methods - Time
    # ########################################################################################### #
    @staticmethod
    def save_conformity_metric_time(
            data: DataFrame,
            cluster: str, recommender: str, dataset: str, trial: int, fold: int,
            distribution: str, fairness: str, relevance: str, weight: str,
            tradeoff: str, selector: str
    ):
        """
        TODO: Docstring
        """
        data.to_csv(
            PathDirFile.set_conformity_metrics_time_file(
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector, cluster=cluster
            ),
            index=False
        )

    # ########################################################################################### #
    # [STEP 5] Metrics step methods - Conformity Evaluation
    # ########################################################################################### #
    @staticmethod
    def save_conformity_metric(
        data: DataFrame,
        cluster: str, metric: str, recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ):
        """
        TODO: Docstring
        """
        data.to_csv(
            PathDirFile.set_conformity_metric_fold_file_by_name(
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
                cluster=cluster, filename=metric + '.csv'
            ),
            index=False, mode='w+'
        )

    @staticmethod
    def load_conformity_metric(
        cluster: str, metric: str, recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
        ext: str = 'csv'
    ) -> DataFrame:
        """
        TODO: Docstring
        """
        path = PathDirFile.get_conformity_metric_fold_file_by_name(
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
            cluster=cluster, filename=metric + '.' + ext
        )
        return read_csv(path)

    # ########################################################################################### #
    # [STEP 5] Metrics step methods - Recommender Evaluation
    # ########################################################################################### #
    @staticmethod
    def save_recommender_metric(
        data: DataFrame,
        metric: str, recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ):
        """
        TODO: Docstring
        """
        data.to_csv(
            PathDirFile.set_recommender_metric_fold_file(
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
                filename=metric + '.csv'
            ),
            index=False, mode='w+'
        )

    @staticmethod
    def load_recommender_metric(
        metric: str, recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
        ext: str = 'csv'
    ) -> DataFrame:
        """
        TODO: Docstring
        """
        path = PathDirFile.get_recommender_metric_fold_file(
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
            filename=metric + '.' + ext
        )
        return read_csv(path)

    # ########################################################################################### #
    # [STEP 6] Compile Metrics step methods - Compiled Evaluation Metric
    # ########################################################################################### #
    @staticmethod
    def save_compiled_metric(data: DataFrame, dataset: str, metric: str, ext: str = 'csv'):
        """
        TODO: Docstring
        """
        path = PathDirFile.set_compiled_metric_file(
            dataset=dataset, filename=metric, ext=ext
        )
        data.to_csv(path, index=False, mode='w+')

    @staticmethod
    def load_compiled_metric(dataset: str, metric: str, ext: str = 'csv') -> DataFrame:
        """
        TODO: Docstring
        """
        path = PathDirFile.get_compiled_metric_file(
            dataset=dataset, filename=metric, ext=ext
        )
        return read_csv(path)
