import logging
import pandas as pd

from checkpoint_verification import CheckpointVerification
from datasets.registred_datasets import RegisteredDataset
from scikit_pierre.metrics.explaining import ExplainingMiscalibration
from scikit_pierre.metrics.evaluation import (
    Miscalibration, MeanAbsoluteCalibrationError,
    MeanAveragePrecision, MeanReciprocalRank, MeanAverageMiscalibration,
    AverageNumberOfOItemsChanges, AverageNumberOfGenreChanges, Unexpectedness, Serendipity,
    NumberOfUserIncreaseAndDecreaseMiscalibration, UserIDMiscalibration,
    IntraListSimilarity, Personalization, Coverage, Novelty
)

from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class ApplyingMetric:
    """

    """

    def __init__(
            self,
            experiment_name: str, based_on: str,
            recommender: str, dataset: str, trial: int, fold: int,
            distribution: str, fairness: str, relevance: str,
            weight: str, tradeoff: str, selector: str,
            checkpoint: str
    ):
        """

        """
        self.recommender = recommender
        self.dataset = dataset
        self.experiment_name = experiment_name
        self.based_on = based_on
        self.trial = trial
        self.fold = fold
        self.distribution = distribution
        self.fairness = fairness
        self.relevance = relevance
        self.weight = weight
        self.tradeoff = tradeoff
        self.selector = selector
        self.checkpoint = checkpoint
        self.metric = None
        self.users_prof_df = None
        self.target_dist = None
        self.users_rec_list_df = None
        self.users_test_set_df = None
        self.users_cand_items_df = None
        self.users_baseline_df = None
        self.dataset_instance = None
        self.metric_instance = None
        self.items_set = None
        self.items_one_hot_encoded = None

    def set_metric(self, metric: str):
        """

        """
        self.metric = metric

    def load_user_profile(self):
        self.users_prof_df = self.dataset_instance.get_full_train_transactions(
            trial=self.trial, fold=self.fold
        )

    def load_rec_list(self):
        """

        """
        self.users_rec_list_df = SaveAndLoad.load_recommendation_lists(
            recommender=self.recommender, dataset=self.dataset, trial=self.trial, fold=self.fold,
            distribution=self.distribution, fairness=self.fairness, relevance=self.relevance,
            tradeoff_weight=self.weight, tradeoff=self.tradeoff, select_item=self.selector,
            experiment_name=self.experiment_name, based_on=self.based_on
        )

    def load_rec_baseline(self):
        """

        """
        if self.users_baseline_df is None:
            self.users_baseline_df = SaveAndLoad.load_recommendation_lists(
                recommender=self.recommender, dataset=self.dataset, trial=self.trial, fold=self.fold,
                distribution=self.distribution, fairness=self.fairness, relevance=self.relevance,
                tradeoff_weight="C@0.0", tradeoff=self.tradeoff, select_item=self.selector,
            experiment_name=self.experiment_name, based_on=self.based_on
            )

    def load_items_set(self):
        self.items_set = self.dataset_instance.get_items()

    def load_item_class_one_hot_encode(self):
        """

        """
        if self.items_one_hot_encoded is None:
            try:
                self.items_one_hot_encoded = SaveAndLoad.load_item_class_one_hot_encode(
                    dataset=self.dataset,
                    experiment_name=self.experiment_name,
                    based_on=self.based_on
                )
            except FileNotFoundError or IOError:
                self.items_one_hot_encoded = None

    def load_candidate_items(self):
        """

        """
        self.users_cand_items_df = SaveAndLoad.load_candidate_items(
            algorithm=self.recommender, dataset=self.dataset, trial=self.trial, fold=self.fold,
            experiment_name=self.experiment_name, based_on=self.based_on
        )

    def load_test_set(self):
        self.users_test_set_df = self.dataset_instance.get_test_transactions(
            trial=self.trial, fold=self.fold
        )

    def load_dataset(self):
        """

        """
        self.dataset_instance = RegisteredDataset.load_dataset(self.dataset)
        self.dataset_instance.set_environment(
            experiment_name=self.experiment_name,
            based_on=self.based_on
        )


    def load_user_prof_distribution(self):
        try:
            if self.target_dist is None:
                target_dist = SaveAndLoad.load_user_preference_distribution(
                    dataset=self.dataset, fold=self.fold, trial=self.trial,
                    distribution=self.distribution,
                    experiment_name=self.experiment_name, based_on=self.based_on
                )
                self.target_dist = target_dist.to_dict('index')
        except IOError or FileNotFoundError as e:
            self.target_dist = None

    def load(self):
        self.load_dataset()
        self.load_rec_list()
        self.load_test_set()
        self.load_candidate_items()
        self.load_items_set()
        self.load_user_profile()

    def load_anc(self):
        self.metric_instance = AverageNumberOfOItemsChanges(
            users_rec_list_df=self.users_rec_list_df,
            users_baseline_df=self.users_baseline_df
        )

    def load_angc(self):
        self.metric_instance = AverageNumberOfGenreChanges(
            users_baseline_df=self.users_baseline_df,
            users_rec_list_df=self.users_rec_list_df,
            items_df=self.items_set
        )

    def load_map(self):
        self.metric_instance = MeanAveragePrecision(
            users_rec_list_df=self.users_rec_list_df,
            users_test_set_df=self.users_test_set_df
        )

    def load_mrr(self):
        self.metric_instance = MeanReciprocalRank(
            users_rec_list_df=self.users_rec_list_df,
            users_test_set_df=self.users_test_set_df
        )

    def load_personalization(self):
        self.metric_instance = Personalization(
            users_rec_list_df=self.users_rec_list_df
        )

    def load_unexpectedness(self):
        self.metric_instance = Unexpectedness(
            users_rec_list_df=self.users_rec_list_df,
            users_test_df=self.users_test_set_df
        )

    def load_coverage(self):
        self.metric_instance = Coverage(
            users_rec_list_df=self.users_rec_list_df,
            items_df=self.items_set
        )

    def load_novelty(self):
        self.metric_instance = Novelty(
            users_profile_df=self.users_prof_df,
            users_rec_list_df=self.users_rec_list_df,
            items_df=self.items_set
        )

    def load_ils(self):
        self.load_item_class_one_hot_encode()
        self.metric_instance = IntraListSimilarity(
            users_rec_list_df=self.users_rec_list_df,
            items_df=self.items_set,
            encoded_df=self.items_one_hot_encoded
        )

    def load_serendipity(self):
        self.metric_instance = Serendipity(
            users_rec_list_df=self.users_rec_list_df,
            users_test_df=self.users_test_set_df,
            users_baseline_df=self.users_baseline_df
        )

    def load_mace(self):
        self.load_user_prof_distribution()
        self.metric_instance = MeanAbsoluteCalibrationError(
            users_profile_df=self.users_prof_df,
            users_rec_list_df=self.users_rec_list_df,
            items_set_df=self.items_set,
            distribution_name=self.distribution
        )

    def load_mc(self):
        self.load_user_prof_distribution()
        self.metric_instance = Miscalibration(
            users_profile_df=self.users_prof_df,
            users_rec_list_df=self.users_rec_list_df,
            items_set_df=self.items_set,
            distribution_name=self.distribution,
            distance_func_name=self.fairness,
            target_dist=self.target_dist
        )

    def load_mamc(self):
        self.load_user_prof_distribution()
        self.metric_instance = MeanAverageMiscalibration(
            users_profile_df=self.users_prof_df,
            users_rec_list_df=self.users_rec_list_df,
            items_set_df=self.items_set,
            distribution_name=self.distribution,
            distance_func_name=self.fairness,
            target_dist=self.target_dist
        )

    def load_exp_mc(self):
        self.metric_instance = ExplainingMiscalibration(
            users_profile_df=self.users_prof_df,
            users_rec_list_df=self.users_rec_list_df,
            users_baseline_df=self.users_baseline_df,
            items_df=self.items_set,
            distribution_name=self.distribution,
            distance_func_name=self.fairness
        )

    def load_inc_dec_mc(self, increase: bool):
        self.metric_instance = NumberOfUserIncreaseAndDecreaseMiscalibration(
            users_profile_df=self.users_prof_df,
            users_rec_list_df=self.users_rec_list_df,
            users_baseline_df=self.users_baseline_df,
            items_df=self.items_set,
            distribution_name=self.distribution,
            distance_func_name=self.fairness
        )
        self.metric_instance.set_choice(choice=increase)

    def load_user_inc_dec_mc(self, increase: bool):
        self.metric_instance = UserIDMiscalibration(
            users_profile_df=self.users_prof_df,
            users_rec_list_df=self.users_rec_list_df,
            users_baseline_df=self.users_baseline_df,
            items_df=self.items_set,
            distribution_name=self.distribution,
            distance_func_name=self.fairness
        )
        self.metric_instance.set_choice(choice=increase)

    def compute(self):
        _value = self.metric_instance.compute()

        results = pd.DataFrame([[
            _value
        ]], columns=[self.metric])

        self.saving_results(results)

    def saving_results(self, results):
        SaveAndLoad.save_recommender_metric(
            data=results, metric=self.metric,
            recommender=self.recommender, dataset=self.dataset, trial=self.trial, fold=self.fold,
            distribution=self.distribution, fairness=self.fairness, relevance=self.relevance,
            weight=self.weight, tradeoff=self.tradeoff, selector=self.selector,
            experiment_name=self.experiment_name, based_on=self.based_on
        )

    def verifying_checkpoint(self):
        if (
            self.checkpoint == "YES" and
            CheckpointVerification.unit_step5_recommendation_verification(
                dataset=self.dataset, trial=self.trial, fold=self.fold,
                metric=self.metric, recommender=self.recommender,
                distribution=self.distribution, fairness=self.fairness,
                relevance=self.relevance,
                weight=self.weight, tradeoff=self.tradeoff, selector=self.selector,
                experiment_name=self.experiment_name, based_on=self.based_on
            )
        ):
            return True
        return False
