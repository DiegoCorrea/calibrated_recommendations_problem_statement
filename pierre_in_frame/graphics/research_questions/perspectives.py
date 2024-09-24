import itertools

import matplotlib.pyplot as plt
import numpy as np

from analyses.results import results_by_component, results_by_fairness
from settings.labels import Label
from settings.path_dir_file import PathDirFile


def components_box_graphic(data, dataset_name, order_by_metric):
    """

    """
    def plot(list_a, list_b, label_a, label_b):
        numbers_a = list(range(1, len(list_a) + len(list_b), 2))
        numbers_b = list(range(2, len(list_a) + len(list_b) + 2, 2))

        plt.figure(figsize=(8, 6))
        plt.grid(True)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.xlabel('Perspectives', fontsize=16)
        plt.ylabel(order_by_metric, fontsize=16)

        color_1 = "red"
        color_2 = "purple"

        # option 1, specify props dictionaries
        plt.boxplot(
            list_a, positions=numbers_a, notch=True, patch_artist=True,
            boxprops=dict(facecolor=color_1, color=color_1, alpha=0.6),
            capprops=dict(color=color_1),
            whiskerprops=dict(color=color_1, alpha=0.6),
            flierprops=dict(color=color_1, markeredgecolor=color_1, alpha=0.6),
            medianprops=dict(color=color_2),
            widths=.75
        )

        # option 2, set all colors individually
        box1 = plt.boxplot(
            list_b, positions=numbers_b, notch=True, patch_artist=True, widths=.75,
            meanline=True, medianprops=dict(color=color_1)
        )
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color=color_2, alpha=0.6)
        plt.setp(box1["boxes"], facecolor=color_2, color=color_2, alpha=0.6)
        plt.setp(box1["fliers"], markeredgecolor=color_2, alpha=0.6)
        plt.setp(box1["whiskers"], markeredgecolor=color_2, alpha=0.6)
        plt.setp(box1["medians"], color=color_1)

        plt.xlim(0, len(list_a) + len(list_b) + 1)
        plt.xticks(numbers_a + numbers_b, label_a + label_b, rotation=30)
        cut_value = np.mean(list_a[0] + list_b[1])
        plt.axhline(y=cut_value, color="green", linestyle="--", linewidth=1, alpha=0.7)

        # Pasta para salvar a figura
        file_name = "".join(["Perspective-", order_by_metric])
        file_dir = PathDirFile.set_graphics_file(dataset_name, file_name + '.png')

        # Salvar figura no disco
        plt.savefig(
            file_dir,
            format='png',
            dpi=400,
            quality=100,
            bbox_inches='tight'
        )

        # Figura fechada
        plt.close('all')

    results_data = results_by_component(data=data, order_by_metric=order_by_metric)
    plot(
        list_a=[
            results_data["N-Calib"], results_data["DIST"], results_data["SUM"],
            results_data["CWS"], results_data["LIN"], results_data["CONST"]
        ],
        list_b=[
            results_data["Calib"], results_data["SIM"], results_data["NDCG"],
            results_data["WPS"], results_data["LOG"], results_data["PERSON"]
        ],
        label_a=["N-Calib", "DIST", "SUM", "CWS", "LIN", "CONST"],
        label_b=["Calib", "SIM", "NDCG", "WPS", "LOG", "PERSON"]
    )


def fairness_box_graphic(data, dataset_name, order_by_metric):
    """

    """
    def plot(list_a, list_b, label_a, label_b):
        numbers_a = list(range(1, len(list_a) + 1))
        numbers_b = list(range(len(list_a) + 1, len(list_a) + len(list_b) + 1))

        plt.figure(figsize=(21, 9))
        plt.grid(True)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        plt.xlabel('Fairness Measures', fontsize=16)
        plt.ylabel(order_by_metric, fontsize=16)

        color_1 = "red"
        color_2 = "purple"

        # option 1, specify props dictionaries
        plt.boxplot(
            list_a, positions=numbers_a, notch=True, patch_artist=True,
            boxprops=dict(facecolor=color_1, color=color_1, alpha=0.6),
            capprops=dict(color=color_1),
            whiskerprops=dict(color=color_1, alpha=0.6),
            flierprops=dict(color=color_1, markeredgecolor=color_1, alpha=0.6),
            medianprops=dict(color=color_2),
            widths=.75
        )

        # option 2, set all colors individually
        box1 = plt.boxplot(
            list_b, positions=numbers_b, notch=True, patch_artist=True, widths=.75,
            meanline=True, medianprops=dict(color=color_1)
        )
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color=color_2, alpha=0.6)
        plt.setp(box1["boxes"], facecolor=color_2, color=color_2, alpha=0.6)
        plt.setp(box1["fliers"], markeredgecolor=color_2, alpha=0.6)
        plt.setp(box1["whiskers"], markeredgecolor=color_2, alpha=0.6)
        plt.setp(box1["medians"], color=color_1)

        plt.xlim(0, len(list_a) + len(list_b) + 1)
        plt.xticks(numbers_a + numbers_b, label_a + label_b, rotation=90)
        a_mean = list(itertools.chain(*list_a))
        b_mean = list(itertools.chain(*list_b))
        cut_value = np.mean(a_mean + b_mean)
        plt.axhline(y=cut_value, color="green", linestyle="--", linewidth=1, alpha=0.7)

        # Pasta para salvar a figura
        file_name = "".join(["All-Fairness-Measures-", order_by_metric])
        file_dir = PathDirFile.set_graphics_file(dataset_name, file_name + '.png')

        # Salvar figura no disco
        plt.savefig(
            file_dir,
            format='png',
            dpi=400,
            quality=100,
            bbox_inches='tight'
        )

        # Figura fechada
        plt.close('all')

    results_data = results_by_fairness(data=data, order_by_metric=order_by_metric)

    plot(
        list_a=[
            results_data[index] for index in Label.DIVERGENCE_LIST
        ],
        list_b=[
            results_data[index] for index in Label.SIMILARITY_LIST
        ],
        label_a=Label.DIVERGENCE_LIST,
        label_b=Label.SIMILARITY_LIST
    )
