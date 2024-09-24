import itertools
import logging
from statistics import mean, mode

import pandas as pd
from joblib import Parallel, delayed
from pandas import DataFrame

from evaluations.metrics_comprises import MetricComprises
from settings.constants import Constants
from settings.labels import Label
from settings.save_and_load import SaveAndLoad
from utils.input import Input
from utils.logging_settings import setup_logging
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep6(Step):
    """
    TODO: Docstring
    """

    def read_the_entries(self):
        """
        TODO: Docstring
        """
        self.experimental_settings = Input.step6()

    @staticmethod
    def set_the_logfile_by_instance(
            dataset: str, recommender: str, trial: int, fold: int, tradeoff: str, distribution: str,
            fairness: str, relevance: str, tradeoff_weight: str, select_item: str
    ):
        """
        TODO: Docstring
        """
        # Setup Log configuration
        # setup_logging(
        #     log_error="error.log", log_info="info.log",
        #     save_path=PathDirFile.set_decision_file(
        #         dataset=dataset, recommender=recommender, trial=trial, fold=fold, tradeoff=tradeoff,
        #         distribution=distribution, fairness=fairness, relevance=relevance, tradeoff_weight=tradeoff_weight,
        #         select_item=select_item
        #     )
        # )
        pass

    def print_basic_info_by_instance(self, **kwargs):
        """
        TODO: Docstring
        """

        logger.info("$" * 50)
        logger.info("$" * 50)
        # Logging machine data
        self.machine_information()
        logger.info("-" * 50)

        # Logging the experiment setup
        logger.info("[METRIC STEP]")
        logger.info(kwargs)
        logger.info("$" * 50)
        logger.info("$" * 50)

    def main(self):
        """
        TODO: Docstring
        """
        if self.experimental_settings['opt'] == Label.CONFORMITY:
            self.conformity_parallelization()
        elif self.experimental_settings['opt'] == Label.EVALUATION_METRICS:
            self.recommender_metrics_parallelization()
        else:
            print(f"Option {self.experimental_settings['opt']} is not registered!")

    # Conformity parallelization
    def load_conformity_metric_jaccard(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_cand_items_list = []
        users_rec_lists_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_conformity_metric(
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=conformity, metric=Label.JACCARD_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_cand_items_list.append(metric_df.iloc[0][Label.JACCARD_SCORE])
                users_rec_lists_list.append(metric_df.iloc[1][Label.JACCARD_SCORE])

        merged_metrics_df = DataFrame([
            [mean(users_cand_items_list), Label.USERS_CAND_ITEMS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mean(users_rec_lists_list), Label.USERS_REC_LISTS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight]
        ],
            columns=[
                Label.EVALUATION_METRICS, Label.CONFORMITY_DIST_MEANING,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df

    def conformity_jaccard_metric(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        jaccard_df = self.load_conformity_metric_jaccard(
            dataset=dataset, recommender=recommender, conformity=conformity,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, weight=weight, selector=selector
        )
        jaccard_df['COMBINATION'] = "-".join([
            recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight
        ])

        return jaccard_df

    def load_conformity_metric_silhouette(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_pref_list = []
        users_cand_items_list = []
        users_rec_lists_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_conformity_metric(
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=conformity, metric=Label.SILHOUETTE_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_pref_list.append(metric_df.iloc[0][Label.SILHOUETTE_SCORE])
                users_cand_items_list.append(metric_df.iloc[1][Label.SILHOUETTE_SCORE])
                users_rec_lists_list.append(metric_df.iloc[2][Label.SILHOUETTE_SCORE])

        merged_metrics_df = DataFrame([
            [mean(users_pref_list), Label.USERS_PREF,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mean(users_cand_items_list), Label.USERS_CAND_ITEMS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mean(users_rec_lists_list), Label.USERS_REC_LISTS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight]],
            columns=[
                Label.EVALUATION_METRICS, Label.CONFORMITY_DIST_MEANING,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df

    def conformity_silhouette_metric(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        silhouette_df = self.load_conformity_metric_silhouette(
            dataset=dataset, recommender=recommender, conformity=conformity,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, weight=weight, selector=selector
        )
        silhouette_df['COMBINATION'] = "-".join([
            recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight
        ])

        return silhouette_df

    def load_conformity_metric_label(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_pref_list = []
        users_cand_items_list = []
        users_rec_lists_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_conformity_metric(
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=conformity, metric=Label.LABEL_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_pref_list.append(metric_df.iloc[0][Label.LABEL_SCORE])
                users_cand_items_list.append(metric_df.iloc[1][Label.LABEL_SCORE])
                users_rec_lists_list.append(metric_df.iloc[2][Label.LABEL_SCORE])

        merged_metrics_df = DataFrame([
            [mode(users_pref_list), Label.USERS_PREF,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_cand_items_list), Label.USERS_CAND_ITEMS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_rec_lists_list), Label.USERS_REC_LISTS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight]],
            columns=[
                Label.EVALUATION_METRICS, Label.CONFORMITY_DIST_MEANING,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df

    def conformity_labels_metric(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        label_df = self.load_conformity_metric_label(
            dataset=dataset, recommender=recommender, conformity=conformity,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, weight=weight, selector=selector
        )
        label_df['COMBINATION'] = "-".join([
            recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight
        ])

        return label_df

    def load_conformity_metric_davies(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_pref_list = []
        users_cand_items_list = []
        users_rec_lists_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_conformity_metric(
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=conformity, metric=Label.DAVIES_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_pref_list.append(metric_df.iloc[0][Label.DAVIES_SCORE])
                users_cand_items_list.append(metric_df.iloc[1][Label.DAVIES_SCORE])
                users_rec_lists_list.append(metric_df.iloc[2][Label.DAVIES_SCORE])

        merged_metrics_df = DataFrame([
            [mode(users_pref_list), Label.USERS_PREF,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_cand_items_list), Label.USERS_CAND_ITEMS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_rec_lists_list), Label.USERS_REC_LISTS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight]],
            columns=[
                Label.EVALUATION_METRICS, Label.CONFORMITY_DIST_MEANING,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df

    def conformity_davies_metric(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        label_df = self.load_conformity_metric_davies(
            dataset=dataset, recommender=recommender, conformity=conformity,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, weight=weight, selector=selector
        )
        label_df['COMBINATION'] = "-".join([
            recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight
        ])

        return label_df

    def load_conformity_metric_calinski(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_pref_list = []
        users_cand_items_list = []
        users_rec_lists_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_conformity_metric(
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=conformity, metric=Label.CALINSKI_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_pref_list.append(metric_df.iloc[0][Label.CALINSKI_SCORE])
                users_cand_items_list.append(metric_df.iloc[1][Label.CALINSKI_SCORE])
                users_rec_lists_list.append(metric_df.iloc[2][Label.CALINSKI_SCORE])

        merged_metrics_df = DataFrame([
            [mode(users_pref_list), Label.USERS_PREF,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_cand_items_list), Label.USERS_CAND_ITEMS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_rec_lists_list), Label.USERS_REC_LISTS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight]],
            columns=[
                Label.EVALUATION_METRICS, Label.CONFORMITY_DIST_MEANING,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df

    def conformity_calinski_metric(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        label_df = self.load_conformity_metric_calinski(
            dataset=dataset, recommender=recommender, conformity=conformity,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, weight=weight, selector=selector
        )
        label_df['COMBINATION'] = "-".join([
            recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight
        ])

        return label_df

    def conformity_parallelization(self):
        """
        TODO: Docstring
        """
        for dataset in self.experimental_settings['dataset']:
            combination = [
                self.experimental_settings['recommender'], self.experimental_settings['conformity'],
                self.experimental_settings['distribution'], self.experimental_settings['fairness'],
                self.experimental_settings['relevance'], self.experimental_settings['weight'],
                self.experimental_settings['tradeoff'], self.experimental_settings['selector']
            ]

            # Jaccard
            jaccard_output = Parallel(n_jobs=Constants.N_CORES)(
                delayed(self.conformity_jaccard_metric)(
                    dataset=dataset, recommender=recommender, conformity=conformity,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                ) for recommender, conformity, distribution, fairness, relevance, weight, tradeoff, selector
                in list(itertools.product(*combination))
            )
            jaccard_results = pd.concat(jaccard_output)
            print(jaccard_results)
            SaveAndLoad.save_compiled_metric(
                data=jaccard_results, dataset=dataset, metric=Label.JACCARD_SCORE
            )

            # Silhouette
            silhouette_output = Parallel(n_jobs=Constants.N_CORES)(
                delayed(self.conformity_silhouette_metric)(
                    dataset=dataset, recommender=recommender, conformity=conformity,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                ) for recommender, conformity, distribution, fairness, relevance, weight, tradeoff, selector
                in list(itertools.product(*combination))
            )
            silhouette_results = pd.concat(silhouette_output)
            print(silhouette_results)
            SaveAndLoad.save_compiled_metric(
                data=silhouette_results, dataset=dataset, metric=Label.SILHOUETTE_SCORE
            )

            # Labels
            label_output = Parallel(n_jobs=Constants.N_CORES)(
                delayed(self.conformity_labels_metric)(
                    dataset=dataset, recommender=recommender, conformity=conformity,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                ) for recommender, conformity, distribution, fairness, relevance, weight, tradeoff, selector
                in list(itertools.product(*combination))
            )
            label_results = pd.concat(label_output)
            print(label_results)
            SaveAndLoad.save_compiled_metric(
                data=label_results, dataset=dataset, metric=Label.LABEL_SCORE
            )

            # Davies
            davies_output = Parallel(n_jobs=Constants.N_CORES)(
                delayed(self.conformity_davies_metric)(
                    dataset=dataset, recommender=recommender, conformity=conformity,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                ) for recommender, conformity, distribution, fairness, relevance, weight, tradeoff, selector
                in list(itertools.product(*combination))
            )
            davies_results = pd.concat(davies_output)
            print(davies_results)
            SaveAndLoad.save_compiled_metric(
                data=davies_results, dataset=dataset, metric=Label.DAVIES_SCORE
            )

            # Calinski
            calinski_output = Parallel(n_jobs=Constants.N_CORES)(
                delayed(self.conformity_calinski_metric)(
                    dataset=dataset, recommender=recommender, conformity=conformity,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                ) for recommender, conformity, distribution, fairness, relevance, weight, tradeoff, selector
                in list(itertools.product(*combination))
            )
            calinski_results = pd.concat(calinski_output)
            print(calinski_results)
            SaveAndLoad.save_compiled_metric(
                data=calinski_results, dataset=dataset, metric=Label.CALINSKI_SCORE
            )

    # Metrics Parallelization

    def recommender_metrics_parallelization(self):
        """
        TODO: Docstring
        """
        for dataset in self.experimental_settings['dataset']:
            for metric in self.experimental_settings['metric']:
                combination = [
                    self.experimental_settings['recommender'],
                    self.experimental_settings['distribution'], self.experimental_settings['fairness'],
                    self.experimental_settings['relevance'], self.experimental_settings['weight'],
                    self.experimental_settings['tradeoff'], self.experimental_settings['selector']
                ]
                process_combination = list(itertools.product(*combination))
                print(f"Dataset {dataset} with metric {metric} starts {len(process_combination)} processes.")
                output = Parallel(
                    n_jobs=self.experimental_settings['n_jobs'], verbose=10, batch_size=1,
                    backend="multiprocessing", prefer="processes"
                )(
                    delayed(MetricComprises.it_comprises_recommender_metric)(
                        recommender=recommender, dataset=dataset, metric=metric,
                        trial=self.experimental_settings['trial'], fold=self.experimental_settings['fold'],
                        distribution=distribution, fairness=fairness, relevance=relevance,
                        weight=weight, tradeoff=tradeoff, selector=selector
                    ) for recommender, distribution, fairness, relevance, weight, tradeoff, selector
                    in process_combination
                )

                results = pd.concat(output)

                SaveAndLoad.save_compiled_metric(
                    data=results, dataset=dataset, metric=metric
                )


if __name__ == '__main__':
    """
    Starting the decision protocol
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep6()
    step.read_the_entries()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
