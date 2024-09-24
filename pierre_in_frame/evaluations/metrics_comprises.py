from pandas import DataFrame
from statistics import mean

from settings.labels import Label
from settings.save_and_load import SaveAndLoad


class MetricComprises:
    @staticmethod
    def it_comprises_recommender_metric(
            dataset: str, recommender: str, metric: str, trial: list, fold: list,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_pref_list = []

        for t in trial:
            for f in fold:
                metric_df = SaveAndLoad.load_recommender_metric(
                    dataset=dataset, trial=t, fold=f,
                    metric=metric, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_pref_list.append(mean(metric_df[metric].tolist()))

        merged_metrics_df = DataFrame(
            data=[[
                mean(users_pref_list), metric,
                recommender, tradeoff, distribution, fairness, relevance, selector, weight
            ]],
            columns=[
                Label.EVALUATION_VALUE, Label.RECOMMENDER_METRIC,
                Label.RECOMMENDER, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df


