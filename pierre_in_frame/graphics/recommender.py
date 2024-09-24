import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from evaluations.hypothesis import WelchHypothesisTest
from settings.charts import ChartsConfig
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class SingleRecommenderGraphics:
    """
    TODO
    """

    @staticmethod
    def weight_by_metric_histogram(data: DataFrame, dataset_name: str, metric: str):
        """
        TODO
        """

        X_axis = np.arange(len(Label.CONST_WEIGHT))

        plt.bar(X_axis, data[Label.EVALUATION_METRICS].tolist(), 0.4)

        plt.xticks(X_axis, Label.CONST_WEIGHT)
        plt.xlabel("Trafeoff Weight")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.legend()

        # Pasta para salvar a figura
        filename = "_".join([metric, "histogram", "CONST", ".png"])
        file_dir = PathDirFile.set_graphics_file(dataset_name, filename)
        # Salvar figura no disco
        plt.savefig(
            file_dir,
            format='png',
            dpi=400,
            bbox_inches='tight'
        )
        # Figura fechada
        plt.close('all')

    @staticmethod
    def weight_by_metric_all_datasets(data: dict, y_label: str, metric: str):
        """
        TODO
        """
        plt.grid(True)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        plt.xlabel("Tradeoff Weight", fontsize=18)
        plt.ylabel(y_label, fontsize=24)

        n = len(Label.CONST_WEIGHT)
        datasets = ['ml-1m', 'yahoo-movies', 'twitter_movies']
        datasets_name = ["Movielens", "Yahoo Movies", "Twitter Movies"]
        for dataset, name, m, l in zip(datasets, datasets_name, ChartsConfig.markers_list[:n], ChartsConfig.line_style_list[:n]):
            df = data[dataset]
            plt.plot(
                [str(x) for x in df[Label.TRADEOFF_WEIGHT_LABEL].tolist()],
                df[Label.EVALUATION_METRICS].abs().tolist(), alpha=0.5, linestyle=l, marker=m,
                label=str(name), linewidth=4
            )

        lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.20), ncol=3, prop={'size': 18})
        plt.xticks(rotation=30)

        # Pasta para salvar a figura
        filename = "_".join([metric, "all_datasets", ".png"])
        file_dir = PathDirFile.set_graphics_file("all", filename)
        # Salvar figura no disco
        plt.savefig(
            file_dir,
            format='png',
            dpi=400,
            bbox_inches='tight'
        )
        # Figura fechada
        plt.close('all')

    @staticmethod
    def line_generic_weight_by_metric(
            data: DataFrame, y_label: str, metric: str, dataset: str,
            goal_label: str, goal_list: list, graphic_name: str) -> None:
        """
        TODO: Docstring
        """
        plt.grid(True)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        plt.xlabel("Tradeoff Weight", fontsize=18)
        plt.ylabel(y_label, fontsize=24)

        n = len(goal_list)
        for goal, m, l in zip(goal_list, ChartsConfig.markers_list[:n], ChartsConfig.line_style_list[:n]):
            graphic_data = data[data[goal_label] == goal]
            plt.plot(
                [str(x.split("@")[1]) for x in graphic_data[Label.TRADEOFF_WEIGHT_LABEL].tolist()],
                graphic_data[Label.EVALUATION_VALUE].abs().tolist(), alpha=0.5, linestyle=l, marker=m,
                label=str(goal), linewidth=4
            )

        lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.20), ncol=3, prop={'size': 18})
        plt.xticks(rotation=30)

        # Pasta para salvar a figura
        filename = graphic_name + ".png"
        file_dir = PathDirFile.set_graphics_dataset_metric_file(dataset=dataset, metric=metric, filename=filename)
        # Salvar figura no disco
        plt.savefig(
            file_dir,
            format='png',
            dpi=400,
            bbox_inches='tight'
        )
        # Figura fechada
        plt.close('all')


class WelchHypothesisTestGraphics:
    """
    TODO
    """

    @staticmethod
    def heatmap(
            data: DataFrame, metric: str, dataset: str,
            goal_label: str, goal_list: list, graphic_name: str) -> None:
        """
        TODO: Docstring
        """
        plt.grid(True)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)

        harvest = np.array(WelchHypothesisTest.compute_matrix_comparison(data=data, goal_list=goal_list, goal_label=goal_label))

        fig, ax = plt.subplots()
        im = ax.imshow(harvest)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(goal_list)), labels=goal_list)
        ax.set_yticks(np.arange(len(goal_list)), labels=goal_list)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(goal_list)):
            for j in range(len(goal_list)):
                text = ax.text(j, i, harvest[i, j], ha="center", va="center", color="w")

        # ax.set_title("Harvest of local farmers (in tons/year)")
        fig.tight_layout()
        # plt.show()

        # Pasta para salvar a figura
        filename = graphic_name + ".png"
        file_dir = PathDirFile.set_graphics_dataset_metric_file(dataset=dataset, metric=metric, filename=filename)
        # Salvar figura no disco
        plt.savefig(
            file_dir,
            format='png',
            dpi=400,
            bbox_inches='tight'
        )
        # Figura fechada
        plt.close('all')
