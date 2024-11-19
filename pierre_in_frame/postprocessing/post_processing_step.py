import logging

from scikit_pierre.tradeoff.calibration import LinearCalibration, LogarithmBias

from datasets.candidate_items import CandidateItems
from datasets.registred_datasets import RegisteredDataset
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class PostProcessingStep:
    """
    Class to lead with post-processing step
    """

    def __init__(
            self,
            experiment_name: str, split_methodology: str,
            recommender: str, dataset_name: str, fold: int, trial: int,
            tradeoff_component: str, distribution_component: str, fairness_component: str,
            relevance_component: str, tradeoff_weight_component: str, selector_component: str,
            list_size: int, alpha: int, d: int, distribution_class: str
    ):
        self.experiment_name = experiment_name
        self.split_methodology = split_methodology
        self.recommender = recommender
        self.fold = fold
        self.trial = trial
        self.tradeoff_component = tradeoff_component
        self.distribution_component = distribution_component
        self.fairness_component = fairness_component
        self.relevance_component = relevance_component
        self.tradeoff_weight_component = tradeoff_weight_component
        self.selector_component = selector_component
        self.list_size = list_size
        self.alpha = alpha
        self.d = d
        self.distribution_class = distribution_class
        # Load dataset
        self.dataset = RegisteredDataset.load_dataset(dataset_name)
        self.dataset.set_environment(
            experiment_name=self.experiment_name,
            split_methodology=self.split_methodology
        )
        # Load candidate items set
        self.users_distribution = None
        self.candidate_items = CandidateItems(
            experiment_name=experiment_name, split_methodology=split_methodology,
            recommender=recommender, dataset=dataset_name, trial=trial, fold=fold)
        try:
            self.users_distribution = SaveAndLoad.load_user_preference_distribution(
                experiment_name=experiment_name, split_methodology=split_methodology,
                dataset=dataset_name, fold=fold, trial=trial, distribution=distribution_component,
                distribution_class=distribution_class
            )
        except IOError:
            self.users_distribution = None
            print("We do not find the preference distribution precomputed."
                  "It will take more time to compute")

        # Choice the tradeoff
        if self.tradeoff_component == 'LIN':
            self.tradeoff_instance = LinearCalibration(
                users_preferences=self.dataset.get_full_train_transactions(fold=fold, trial=trial),
                candidate_items=self.candidate_items.get_candidate_items(),
                item_set=self.dataset.get_items(),
                users_distribution=self.users_distribution
            )
        elif self.tradeoff_component == 'LOG':
            self.tradeoff_instance = LogarithmBias(
                users_preferences=self.dataset.get_full_train_transactions(fold=fold, trial=trial),
                candidate_items=self.candidate_items.get_candidate_items(),
                item_set=self.dataset.get_items(),
                users_distribution=self.users_distribution
            )
        # elif self.tradeoff_component == 'POP_LIN':
        #     items_dataframe = self.dataset.get_items()
        #     items_dataframe.drop(["GENRES"])
        #     items_dataframe.rename(columns={"POPULARITY": "GENRES"}, inplace=True)
        #     self.tradeoff_instance = PopularityCalibration(
        #         users_preferences=self.dataset.get_full_train_transactions(fold=fold, trial=trial),
        #         candidate_items=self.candidate_items.get_candidate_items(),
        #         item_set=items_dataframe,
        #         users_distribution=self.users_distribution
        #     )
        # elif self.tradeoff_component == 'POP':
        #     items_dataframe = self.dataset.get_items()
        #     items_dataframe.drop(["GENRES"])
        #     items_dataframe.rename(columns={"POPULARITY": "GENRES"}, inplace=True)
        #     self.tradeoff_instance = PopularityCalibration(
        #         users_preferences=self.dataset.get_full_train_transactions(fold=fold, trial=trial),
        #         candidate_items=self.candidate_items.get_candidate_items(),
        #         item_set=items_dataframe,
        #         users_distribution=self.users_distribution
        #     )
        else:
            exit(0)

        # Configuring the experimentation
        self.tradeoff_instance.config(
            distribution_component=distribution_component,
            fairness_component=fairness_component,
            relevance_component=relevance_component,
            tradeoff_weight_component=tradeoff_weight_component,
            select_item_component=selector_component,
            list_size=list_size, d=d, alpha=alpha
        )

    def run(self):
        """
        TODO: Docstring
        """

        # Execute the instance and get the recommendation list to all users.
        merged_results_df = self.tradeoff_instance.fit()

        # Save all recommendation lists
        SaveAndLoad.save_recommendation_lists(
            experiment_name=self.experiment_name, split_methodology=self.split_methodology,
            data=merged_results_df,
            recommender=self.recommender, dataset=self.dataset.system_name,
            trial=self.trial, fold=self.fold,
            tradeoff=self.tradeoff_component,
            distribution=self.distribution_component,
            fairness=self.fairness_component,
            relevance=self.relevance_component,
            tradeoff_weight=self.tradeoff_weight_component,
            select_item=self.selector_component,
            distribution_class=self.distribution_class
        )
