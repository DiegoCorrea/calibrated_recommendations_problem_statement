import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from settings.charts import ChartsConfig
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class ConformityGraphics:
    """
    TODO
    """

    @staticmethod
    def pref_cand_silhouette_boxplot(data: DataFrame, dataset_name: str, conformity_algos: list):
        """
        TODO
        """

        metric = Label.SILHOUETTE_SCORE
        X = conformity_algos

        data_pref = data[data[Label.CONFORMITY_DIST_MEANING] == Label.USERS_PREF]
        data_cand = data[data[Label.CONFORMITY_DIST_MEANING] == Label.USERS_CAND_ITEMS]

        user_pref_values = [
            np.mean(data_pref[data_pref[Label.CONFORMITY] == algo][Label.EVALUATION_METRICS].abs().tolist()) for algo in
            X
        ]
        cand_items_values = [
            np.mean(data_cand[data_cand[Label.CONFORMITY] == algo][Label.EVALUATION_METRICS].abs().tolist()) for algo in
            X
        ]

        X_axis = np.arange(len(X))

        plt.bar(X_axis - 0.2, user_pref_values, 0.4, label="Preferences")
        plt.bar(X_axis + 0.2, cand_items_values, 0.4, label="Candidates")

        plt.xticks(X_axis, X)
        plt.xlabel("Algorithms")
        plt.ylabel("Silhouette Value")
        plt.xticks(rotation=45)
        plt.legend()

        # Pasta para salvar a figura
        filename = "_".join([metric, Label.USERS_PREF, Label.USERS_CAND_ITEMS, ".png"])
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
    def weight_by_metric_line(
            data: DataFrame, dataset_name: str, conformity_algos: list, y_label: str, metric_name: str
    ):
        """
        TODO
        """
        data_pref = data[data[Label.CONFORMITY_DIST_MEANING] == Label.USERS_PREF]
        user_pref_values = [
            np.mean(data_pref[data_pref[Label.CONFORMITY] == algo][Label.EVALUATION_METRICS].abs().tolist()) for algo in
            conformity_algos
        ]
        used_data = data[data[Label.TRADEOFF_WEIGHT_LABEL].isin(Label.CONST_WEIGHT)]
        rec_list_df = used_data[used_data[Label.CONFORMITY_DIST_MEANING] == Label.USERS_REC_LISTS]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        plt.grid(True)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.xlabel("Tradeoff Weight", fontsize=18)
        plt.ylabel(y_label, fontsize=24)

        n = len(conformity_algos)
        for algo, m, l, pref in zip(
                conformity_algos, ChartsConfig.markers_list[:n], ChartsConfig.line_style_list[:n], user_pref_values
        ):
            df = rec_list_df[rec_list_df[Label.CONFORMITY] == algo]
            plt.plot(
                ["PREF"] + [str(x) for x in df[Label.TRADEOFF_WEIGHT_LABEL].tolist()],
                [pref] + df[Label.EVALUATION_METRICS].abs().tolist(),
                alpha=0.5, linestyle=l, marker=m, label=str(algo), linewidth=4
            )

        lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.20), ncol=3, prop={'size': 18})
        plt.xticks(rotation=30)

        # Pasta para salvar a figura
        filename = "_".join([metric_name, Label.TRADEOFF_WEIGHT_LABEL, ".png"])
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
    def weight_by_silhouette_line(data: DataFrame, dataset_name: str, conformity_algos: list):
        """
        TODO
        """
        data_pref = data[data[Label.CONFORMITY_DIST_MEANING] == Label.USERS_PREF]
        user_pref_values = [
            np.mean(data_pref[data_pref[Label.CONFORMITY] == algo][Label.EVALUATION_METRICS].abs().tolist()) for algo in
            conformity_algos
        ]
        used_data = data[data[Label.TRADEOFF_WEIGHT_LABEL].isin(Label.CONST_WEIGHT)]
        rec_list_df = used_data[used_data[Label.CONFORMITY_DIST_MEANING] == Label.USERS_REC_LISTS]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        plt.grid(True)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.xlabel("Tradeoff Weight", fontsize=18)
        plt.ylabel("Silhouette Value", fontsize=24)

        n = len(conformity_algos)
        for algo, m, l, pref in zip(
                conformity_algos, ChartsConfig.markers_list[:n], ChartsConfig.line_style_list[:n], user_pref_values
        ):
            df = rec_list_df[rec_list_df[Label.CONFORMITY] == algo]
            plt.plot(
                ["PREF"] + [str(x) for x in df[Label.TRADEOFF_WEIGHT_LABEL].tolist()],
                [pref] + df[Label.EVALUATION_METRICS].abs().tolist(),
                alpha=0.5, linestyle=l, marker=m, label=str(algo), linewidth=4
            )

        lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.20), ncol=3, prop={'size': 18})
        plt.xticks(rotation=30)

        # Pasta para salvar a figura
        filename = "_".join([Label.SILHOUETTE_SCORE, Label.TRADEOFF_WEIGHT_LABEL, ".png"])
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
    def weight_by_jaccard_line(data: DataFrame, dataset_name: str, conformity_algos: list, rule: str):
        """
        TODO
        """

        used_data = data[data[Label.TRADEOFF_WEIGHT_LABEL].isin(Label.CONST_WEIGHT)]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        plt.grid(True)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.xlabel("Tradeoff Weight", fontsize=18)
        plt.ylabel("Jaccard Value", fontsize=24)

        n = len(conformity_algos)
        for algo, m, l in zip(conformity_algos, ChartsConfig.markers_list[:n], ChartsConfig.line_style_list[:n]):
            df = used_data[used_data[Label.CONFORMITY] == algo]
            plt.plot(
                [str(x) for x in df[Label.TRADEOFF_WEIGHT_LABEL].tolist()],
                df[Label.EVALUATION_METRICS].abs().tolist(), alpha=0.5, linestyle=l, marker=m,
                label=str(algo), linewidth=4
            )

        lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.20), ncol=3, prop={'size': 18})
        plt.xticks(rotation=30)

        # Pasta para salvar a figura
        filename = "_".join([Label.JACCARD_SCORE, rule, ".png"])
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
