
import matplotlib
import numpy as np
import pandas as pd

from settings.charts import ChartsConfig
from settings.labels import Label
from settings.path_dir_file import PathDirFile

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
# matplotlib.style.use('default')


class GenreChats:
    # ######################### #
    #     Data set graphics     #
    # ######################### #
    @staticmethod
    def user_model_size_by_number_of_genres(user_profile_df: pd.DataFrame, dataset: str) -> None:
        x_data = user_profile_df[Label.USER_MODEL_SIZE_LABEL].tolist()
        y_data = user_profile_df[Label.NUMBER_OF_CLASSES].tolist()
        plt.figure()
        plt.grid(True)
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        plt.xlabel("User Profile Size", fontsize=18)
        plt.ylabel("Number of Genres", fontsize=18)
        plt.scatter(x_data,
                    y_data,
                    alpha=0.5, c=ChartsConfig.SCATTER_BUBBLE_COLOR)
        list_number = list(set(y_data))
        plt.yticks(range(min(list_number), max(list_number) + 1))
        plt.xticks(rotation=30)
        # plt.legend()

        saving_file = PathDirFile.preprocessing_graphics_file(dataset, 'user_model_size_by_number_of_genres.eps')
        # Salvar figura no disco
        plt.savefig(
            saving_file,
            format='eps',
            dpi=ChartsConfig.DPI_VALUE,
            bbox_inches='tight'
        )

        saving_file = PathDirFile.preprocessing_graphics_file(dataset, 'user_model_size_by_number_of_genres.png')
        # Salvar figura no disco
        plt.savefig(
            saving_file,
            format='png',
            dpi=ChartsConfig.DPI_VALUE,
            bbox_inches='tight'
        )
        plt.close('all')

    @staticmethod
    def compare_genre_distribution_bar(users_genre_distr_df, items_genre_distr_df, dataset: str):
        users_genre_distr_df = users_genre_distr_df.reindex(sorted(users_genre_distr_df.columns), axis=1)
        items_genre_distr_df = items_genre_distr_df.reindex(sorted(items_genre_distr_df.columns), axis=1)
        women_means = users_genre_distr_df.mean(axis=0).tolist()
        men_means = items_genre_distr_df.mean(axis=0).tolist()
        users_std = users_genre_distr_df.sem(axis=0).tolist()
        items_std = items_genre_distr_df.sem(axis=0).tolist()
        labels = users_genre_distr_df.columns.tolist()

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        rects1 = ax.bar(x - width / 2, men_means, width, label='Items', yerr=items_std)
        rects2 = ax.bar(x + width / 2, women_means, width, label='Users', yerr=users_std)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Probability', fontsize=ChartsConfig.FONT_SIZE_VALUE)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()

        plt.xticks(rotation=90)

        saving_file = PathDirFile.preprocessing_graphics_file(dataset, 'compare_genre_distribution_bar.eps')
        # Salvar figura no disco
        plt.savefig(
            saving_file,
            format='eps',
            dpi=ChartsConfig.DPI_VALUE,
            bbox_inches='tight'
        )

        saving_file = PathDirFile.preprocessing_graphics_file(dataset, 'compare_genre_distribution_bar.png')
        # Salvar figura no disco
        plt.savefig(
            saving_file,
            format='png',
            dpi=ChartsConfig.DPI_VALUE,
            bbox_inches='tight'
        )
        plt.close('all')


# def genre_distribution_bar(genre_distr_df, db, distr_type):
#     # genre_distr_df = genre_distr_df.reindex(sorted(genre_distr_df.columns), axis=1)
#     x = genre_distr_df.columns.tolist()
#     y = genre_distr_df.mean(axis=0).tolist()
#     std = genre_distr_df.sem(axis=0).tolist()
#     plt.figure()
#     plt.rc('xtick', labelsize=16)
#     plt.rc('ytick', labelsize=16)
#     plt.bar(x, y, color=scatter_bubble_color, yerr=std)
#     # Turn on the grid
#     plt.xticks(rotation=90)
#     path_to_save = pre_processing_to_use_path(db)
#     if not os.path.exists(path_to_save):
#         os.makedirs(path_to_save)
#     plt.savefig(
#         path_to_save
#         + 'genre_distribution_bar_' + str(distr_type)
#         + '.png',
#         format='png',
#         dpi=DPI_VALUE,
#         quality=QUALITY_VALUE,
#         bbox_inches='tight'
#     )
#     plt.close('all')
#
#

#
#
# # ############################### #
# #     Postprocessing graphics     #
# # ############################### #
# def compare_distributions(distr_list, file_name, title, db=0):
#     concated_distr = pd.DataFrame()
#     for df in distr_list:
#         genre_distr_df = df.reindex(sorted(df.columns), axis=1)
#         x = genre_distr_df.columns.tolist()
#         y = genre_distr_df.mean(axis=0).tolist()
#         std = genre_distr_df.std(axis=0).tolist()
#     distribution_path = pos_processing_results_path(db) + "genres/"
#     plt.figure()
#     plt.rc('xtick', labelsize=16)
#     plt.rc('ytick', labelsize=16)
#     concated_distr.plot(kind='bar')
#     # Turn on the grid
#     plt.minorticks_on()
#     plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
#     plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#     plt.title(title)
#     lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
#                      ncol=3, fancybox=True, shadow=True)
#     plt.xticks(rotation=30)
#     if not os.path.exists(distribution_path):
#         os.makedirs(distribution_path)
#     plt.savefig(
#         distribution_path
#         + file_name
#         + '.png',
#         format='png',
#         dpi=DPI_VALUE,
#         quality=QUALITY_VALUE,
#         bbox_inches='tight'
#     )
#     plt.close('all')
