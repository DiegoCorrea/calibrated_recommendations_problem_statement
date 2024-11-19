import json

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
    def save_clean_transactions(experiment_name: str, dataset: str, split_methodology: str):
        pass

    @staticmethod
    def load_clean_transactions(experiment_name: str, dataset: str, split_methodology: str):
        directory_name = PathDirFile.dataset_path(
            dataset=dataset, experiment_name=experiment_name, split_methodology=split_methodology,
            filename=PathDirFile.TRANSACTIONS_FILE
        )
        data = read_csv(directory_name)
        print(directory_name)
        return data

    @staticmethod
    def load_train_transactions(experiment_name: str, dataset: str, split_methodology: str, trial: int, fold: int):
        directory_name = PathDirFile.dataset_fold_path(
            dataset=dataset, experiment_name=experiment_name, split_methodology=split_methodology,
            trial=trial, fold=fold,
            filename=PathDirFile.TRAIN_FILE
        )
        print(directory_name)
        data = read_csv(directory_name)
        return data

    @staticmethod
    def load_validation_transactions(experiment_name: str, dataset: str, split_methodology: str, trial: int, fold: int):
        directory_name = PathDirFile.dataset_fold_path(
            dataset=dataset, experiment_name=experiment_name, split_methodology=split_methodology,
            trial=trial, fold=fold,
            filename=PathDirFile.VALIDATION_FILE
        )
        print(directory_name)
        data = read_csv(directory_name)
        return data

    @staticmethod
    def load_test_transactions(experiment_name: str, dataset: str, split_methodology: str, trial: int, fold: int):
        directory_name = PathDirFile.dataset_fold_path(
            dataset=dataset, experiment_name=experiment_name, split_methodology=split_methodology,
            trial=trial, fold=fold,
            filename=PathDirFile.TEST_FILE
        )
        print(directory_name)
        data = read_csv(directory_name)
        return data

    @staticmethod
    def load_clean_items(experiment_name: str, dataset: str, split_methodology: str):
        directory_name = PathDirFile.dataset_path(
            dataset=dataset, experiment_name=experiment_name, split_methodology=split_methodology,
            filename=PathDirFile.ITEMS_FILE
        )
        print(directory_name)
        data = read_csv(directory_name)
        return data

    @staticmethod
    def save_clean_items(experiment_name: str, dataset: str, split_methodology: str, data: DataFrame):
        directory_name = PathDirFile.dataset_path(
            dataset=dataset, experiment_name=experiment_name, split_methodology=split_methodology,
            filename=PathDirFile.ITEMS_FILE
        )
        data.to_csv(
            directory_name,
            index=False,
            mode='w+'
        )

    # ########################################################################################### #

    @staticmethod
    def save_user_preference_distribution(
            data: DataFrame, experiment_name: str, dataset: str, split_methodology: str,
            trial: int, fold: int, distribution: str, distribution_class: str,
            ext: str = 'csv'
    ):
        """
        This method is to save the distribution file.
        """
        data.to_csv(
            PathDirFile.dataset_distribution_path(
                dataset=dataset, experiment_name=experiment_name, split_methodology=split_methodology,
                fold=fold, trial=trial, distribution_class=distribution_class,
                filename=distribution + '.' + ext
            )
        )

    @staticmethod
    def load_user_preference_distribution(
            experiment_name: str, dataset: str, split_methodology: str,
            trial: int, fold: int, distribution: str, distribution_class: str,
            ext: str = 'csv'
    ) -> DataFrame:
        """
        This method is to load the distribution file.
        """
        preference_distribution_path = PathDirFile.dataset_distribution_path(
            dataset=dataset, experiment_name=experiment_name, split_methodology=split_methodology,
            fold=fold, trial=trial, distribution_class=distribution_class,
            filename=distribution + '.' + ext
        )
        return read_csv(preference_distribution_path, index_col=0).fillna(0)

    @staticmethod
    def save_distribution_time(
            data: DataFrame, experiment_name: str, dataset: str, split_methodology: str,
            trial: int, fold: int, distribution: str, distribution_class: str,
            ext: str = 'csv'
    ):
        """
        This method is to save the distribution file.
        """
        data.to_csv(
            PathDirFile.dataset_distribution_path(
                dataset=dataset, experiment_name=experiment_name, split_methodology=split_methodology,
                fold=fold, trial=trial, distribution_class=distribution_class,
                filename=distribution + "_" + "TIME"  + '.' + ext
            )
        )

    # ########################################################################################### #

    @staticmethod
    def save_item_class_one_hot_encode(
            data: DataFrame, experiment_name: str, dataset: str, split_methodology: str, ext: str = 'csv'
    ):
        """
        This method is to save the item one hot encode file.
        """
        data.to_csv(
            PathDirFile.dataset_path(
                dataset=dataset, experiment_name=experiment_name, split_methodology=split_methodology,
                filename="item_one_hot_encode" + '.' + ext
            ), mode='w+'
        )

    @staticmethod
    def load_item_class_one_hot_encode(
            experiment_name: str, dataset: str, split_methodology: str, ext: str = 'csv'
    ) -> DataFrame:
        """
        This method is to load the one hot encode file.
        """
        preference_distribution_path = PathDirFile.dataset_path(
            dataset=dataset, experiment_name=experiment_name, split_methodology=split_methodology,
            filename="item_one_hot_encode" + '.' + ext
        )
        return read_csv(preference_distribution_path, index_col=0)

    # ########################################################################################### #

    @staticmethod
    def save_dataset_analyze(
            data: DataFrame, experiment_name: str, dataset: str, split_methodology: str, ext: str = 'csv'
    ):
        """
        This method is to save the distribution file.
        """
        data.to_csv(
            PathDirFile.dataset_path(
                dataset=dataset, experiment_name=experiment_name, split_methodology=split_methodology,
                filename="GENERAL_ANALYZE" + '.' + ext
            ), index=False, mode='w+'
        )

    @staticmethod
    def save_fold_analyze(
            data: DataFrame, experiment_name: str, dataset: str, split_methodology: str, ext: str = 'csv'
    ):
        """
        This method is to save the folds analyze file.
        """
        data.to_csv(
            PathDirFile.dataset_path(
                dataset=dataset, experiment_name=experiment_name, split_methodology=split_methodology,
                filename="FOLDS_ANALYZE" + '.' + ext
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
            best_params: dict, experiment_name: str, dataset: str, split_methodology: str, algorithm: str):
        """
        TODO: Docstring
        """
        with open(PathDirFile.set_recommender_hyperparameter_file(
                opt=Label.RECOMMENDER, dataset=dataset, algorithm=algorithm,
                experiment_name=experiment_name, split_methodology=split_methodology
        ), 'w+') as fp:
            json.dump(best_params, fp, cls=NpEncoder)

    @staticmethod
    def load_hyperparameters_recommender(
            experiment_name: str, dataset: str, split_methodology: str, algorithm: str):
        """
        TODO: Docstring
        """
        path_to_open = PathDirFile.get_recommender_hyperparameter_file(
            opt=Label.RECOMMENDER, dataset=dataset, algorithm=algorithm,
            experiment_name=experiment_name, split_methodology=split_methodology
        )
        with open(path_to_open) as json_file:
            params = json.load(json_file)

        return params

    @staticmethod
    def save_hyperparameters_conformity(
            best_params: dict, experiment_name: str, dataset: str, split_methodology: str,
            cluster: str, distribution: str):
        """
        TODO: Docstring
        """
        with open(PathDirFile.set_conformity_hyperparameter_file(
                opt=Label.CONFORMITY, dataset=dataset, cluster=cluster,
                distribution=distribution,
                experiment_name=experiment_name, split_methodology=split_methodology
        ), 'w') as fp:
            json.dump(best_params, fp)

    @staticmethod
    def load_hyperparameters_conformity(
            experiment_name: str, dataset: str, split_methodology: str, cluster: str, distribution: str
    ):
        """
        TODO: Docstring
        """
        path_to_open = PathDirFile.get_conformity_hyperparameter_file(
            opt=Label.CONFORMITY, dataset=dataset,
            cluster=cluster, distribution=distribution,
            experiment_name=experiment_name, split_methodology=split_methodology
        )
        with open(path_to_open) as json_file:
            params = json.load(json_file)

        return params

    # ########################################################################################### #
    # [STEP 3] Processing step methods - Candidate Items
    # ########################################################################################### #
    @staticmethod
    def save_candidate_items(
            experiment_name: str, split_methodology: str,
            data: DataFrame, dataset: str, algorithm: str, trial: int, fold: int):
        """
        TODO: Docstring
        """
        data.to_csv(
            PathDirFile.set_candidate_items_file(
                experiment_name=experiment_name, split_methodology=split_methodology,
                dataset=dataset, algorithm=algorithm, fold=fold, trial=trial
            ),
            index=False, mode='w+'
        )

    @staticmethod
    def load_candidate_items(
            experiment_name: str, split_methodology: str,
            dataset: str, algorithm: str, trial: int, fold: int):
        """
        TODO: Docstring
        """
        candidate_items_path = PathDirFile.get_candidate_items_file(
            experiment_name=experiment_name, split_methodology=split_methodology,
            dataset=dataset, algorithm=algorithm, fold=fold, trial=trial
        )
        return read_csv(candidate_items_path)

    # ########################################################################################### #
    # [STEP 4] Post-Processing step methods - Recommendation Lists
    # ########################################################################################### #
    @staticmethod
    def load_recommendation_lists(
        experiment_name: str, split_methodology: str,
        dataset: str, recommender: str, trial: int, fold: int,
        tradeoff: str, distribution: str, fairness: str, relevance: str,
        tradeoff_weight: str, select_item: str, distribution_class: str
    ):
        """
        TODO: Docstring
        """
        recommendation_list_path = PathDirFile.get_recommendation_list_file(
            experiment_name=experiment_name, split_methodology=split_methodology,
            dataset=dataset, recommender=recommender, trial=trial, fold=fold,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, tradeoff_weight=tradeoff_weight, select_item=select_item,
            distribution_class=distribution_class
        )
        return read_csv(recommendation_list_path)

    @staticmethod
    def save_recommendation_lists(
        experiment_name: str, split_methodology: str,
        data: DataFrame,
        dataset: str, recommender: str, trial: int, fold: int,
        tradeoff: str, distribution: str, fairness: str, relevance: str,
        tradeoff_weight: str, select_item: str, distribution_class: str
    ):
        """
        TODO: Docstring
        """
        path = PathDirFile.set_recommendation_list_file(
            experiment_name=experiment_name, split_methodology=split_methodology,
            recommender=recommender, dataset=dataset,
            trial=trial, fold=fold,
            tradeoff=tradeoff,
            distribution=distribution,
            fairness=fairness,
            relevance=relevance,
            tradeoff_weight=tradeoff_weight,
            select_item=select_item,
            distribution_class=distribution_class
        )
        data.to_csv(path, index=False, mode='w+')

    # ########################################################################################### #
    # [STEP 5] Metrics step methods - Conformity Evaluation
    # ########################################################################################### #
    @staticmethod
    def save_conformity_metric(
        data: DataFrame,
        experiment_name: str, split_methodology: str,
        cluster: str, metric: str, recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
        distribution_class: str
    ):
        """
        TODO: Docstring
        """
        data.to_csv(
            PathDirFile.set_conformity_metric_fold_file_by_name(
                experiment_name=experiment_name, split_methodology=split_methodology,
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
                cluster=cluster, filename=metric + '.csv', distribution_class=distribution_class
            ),
            index=False, mode='w+'
        )

    @staticmethod
    def load_conformity_metric(
        experiment_name: str, split_methodology: str,
        cluster: str, metric: str, recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
        distribution_class: str,
        ext: str = 'csv'
    ) -> DataFrame:
        """
        TODO: Docstring
        """
        path = PathDirFile.get_conformity_metric_fold_file_by_name(
            experiment_name=experiment_name, split_methodology=split_methodology,
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
            cluster=cluster, filename=metric + '.' + ext, distribution_class=distribution_class
        )
        return read_csv(path)

    # ########################################################################################### #
    # [STEP 5] Metrics step methods - Recommender Evaluation
    # ########################################################################################### #
    @staticmethod
    def save_recommender_metric(
        data: DataFrame, experiment_name: str, split_methodology: str,
        metric: str, recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
        distribution_class: str
    ):
        """
        TODO: Docstring
        """
        data.to_csv(
            PathDirFile.set_recommender_metric_fold_file(
                experiment_name=experiment_name, split_methodology=split_methodology,
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
                filename=metric + '.csv', distribution_class=distribution_class
            ),
            index=False, mode='w+'
        )

    @staticmethod
    def load_recommender_metric(
        experiment_name: str, split_methodology: str,
        metric: str, recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
        distribution_class: str,
        ext: str = 'csv'
    ) -> DataFrame:
        """
        TODO: Docstring
        """
        path = PathDirFile.get_recommender_metric_fold_file(
            experiment_name=experiment_name, split_methodology=split_methodology,
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
            filename=metric + '.' + ext, distribution_class=distribution_class
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
