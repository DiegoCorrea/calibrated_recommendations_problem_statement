import numpy as np
import pandas as pd

from analyses.genres import user_genres_analysis, genre_probability_distribution
from analyses.popularities import count_item_popularity
from datasets.registred_datasets import RegisteredDataset
from graphics.genres import GenreChats
from graphics.popularity import long_tail_graphic, popularity_group_graphic
from settings.labels import Label


class DatasetChart:
    """
    This class administrates the dataset chart generation
    """

    def __init__(self, dataset_name):
        self.dataset = RegisteredDataset.load_dataset(dataset_name)

    def item_long_tail(self):
        """
        This method produce the long tail figure.
        """
        item_popularity = count_item_popularity(self.dataset.get_transactions())
        item_popularity.sort_values(Label.TOTAL_TIMES, inplace=True, ascending=False)
        long_tail_graphic(item_popularity, self.dataset.system_name)

        df_split = np.array_split(item_popularity, 10)
        total = item_popularity['TOTAL_TIMES'].sum()
        values = []
        ix = []
        for i, split in enumerate(df_split):
            values.append(split['TOTAL_TIMES'].sum() / total)
            ix.append('G-' + str(i + 1))
        df_data = pd.DataFrame()
        df_data['group'] = ix
        df_data['values'] = values
        popularity_group_graphic(df_data, self.dataset.system_name)

    def genres(self):
        complete_transactions_df = self.dataset.get_transactions().merge(self.dataset.get_items(), on=Label.ITEM_ID)
        #
        analysis_of_users_df = user_genres_analysis(complete_transactions_df)
        analysis_of_users_df.sort_values(by=[Label.USER_MODEL_SIZE_LABEL], ascending=[False], inplace=True)
        GenreChats.user_model_size_by_number_of_genres(analysis_of_users_df, self.dataset.system_name)
        #
        users_genre_distr_df = genre_probability_distribution(complete_transactions_df, label=Label.USER_ID)

        items_genre_distr_df = genre_probability_distribution(self.dataset.get_items(), label=Label.ITEM_ID)

        GenreChats.compare_genre_distribution_bar(users_genre_distr_df, items_genre_distr_df, self.dataset.dir_name)
