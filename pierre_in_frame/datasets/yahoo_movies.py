import os

import numpy as np
import pandas as pd

from datasets.utils.base_preprocess import Dataset
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class YahooMovies(Dataset):
    """
    Yahoo Movies dataset.
    This class organize the work with the dataset.
    """
    dir_name = "yahoo-movies"
    verbose_name = "Yahoo Movies"
    system_name = "yahoo-movies"
    # Raw
    dataset_raw_path = "/".join([PathDirFile.RAW_DATASETS_DIR, dir_name])
    raw_transaction_file = "ydata-ymovies-user-movie-ratings-train-v1_0.txt"
    raw_transaction_test_file = "ydata-ymovies-user-movie-ratings-test-v1_0.txt"
    raw_items_file = "movies.csv"
    # Clan
    dataset_clean_path = "/".join([PathDirFile.CLEAN_DATASETS_DIR, dir_name])

    # ######################################### #
    # ############## Constructor ############## #
    # ######################################### #

    def __init__(self):
        """
        Class constructor. Firstly call the super constructor and after start personalized things.
        """
        super().__init__()

    # ######################################### #
    # ############# Transactions ############## #
    # ######################################### #

    def load_raw_transactions(self):
        """
        Load Raw Transactions into self variable
        """
        raw_training_df = pd.read_csv(os.path.join(self.dataset_raw_path, self.raw_transaction_file),
                                      names=[Label.USER_ID, Label.ITEM_ID, 'original_value', Label.TRANSACTION_VALUE],
                                      sep='\t')
        raw_testing_df = pd.read_csv(os.path.join(self.dataset_raw_path, self.raw_transaction_test_file),
                                     names=[Label.USER_ID, Label.ITEM_ID, 'original_value', Label.TRANSACTION_VALUE],
                                     sep='\t')
        full_raw_rating = pd.concat([raw_training_df, raw_testing_df])
        full_raw_rating.drop(columns=['original_value'], inplace=True)
        self.raw_transactions = full_raw_rating

    def clean_transactions(self):
        """
        Cleaning the raw transactions and save as clean transactions
        """
        super().clean_transactions()
        raw_transactions = self.get_raw_transactions()

        raw_transactions[Label.ITEM_ID] = raw_transactions[Label.ITEM_ID].astype(str)
        self.items[Label.ITEM_ID] = self.items[Label.ITEM_ID].astype(str)

        self.set_transactions(
            new_transactions=YahooMovies.cut_users(
                raw_transactions[raw_transactions[Label.ITEM_ID].isin(self.items[Label.ITEM_ID].tolist())], 4))
        self.set_items(
            new_items=self.items[self.items[Label.ITEM_ID].isin(self.transactions[Label.ITEM_ID].unique().tolist())])

        if Constants.NORMALIZED_SCORE:
            self.transactions[Label.TRANSACTION_VALUE] = np.where(self.transactions[Label.TRANSACTION_VALUE] >= 4, 1, 0)

        self.transactions.to_csv(os.path.join(self.clean_dataset_dir, PathDirFile.TRANSACTIONS_FILE),
                                 index=False)
        self.items.to_csv(os.path.join(self.clean_dataset_dir, PathDirFile.ITEMS_FILE), index=False)

    # ######################################### #
    # ################# Items ################# #
    # ######################################### #

    def load_raw_items(self):
        """
        Load Raw Items
        """
        self.raw_items = pd.read_csv(os.path.join(self.dataset_raw_path, self.raw_items_file), engine='python',
                                     dtype=str,
                                     sep=',', names=[Label.ITEM_ID, Label.TITLE, Label.GENRES], header=None, skiprows=1)

    def clean_items(self):
        """
        Cleaning the raw items and save as clean items
        """
        # Load transactions set from csv
        raw_items_df = self.get_raw_items()
        #
        yahoo_items_mapping = pd.read_csv(
            os.path.join(self.dataset_raw_path, "ydata-ymovies-mapping-to-movielens-v1_0.txt"),
            names=[Label.ITEM_ID, Label.TITLE, "movieId"], sep='\t', encoding='ISO-8859-1',
            dtype=str)
        yahoo_items_mapping.drop([Label.TITLE], axis=1, inplace=True)
        yahoo_items_mapping.dropna(inplace=True)

        items_df = pd.merge(yahoo_items_mapping, raw_items_df, how='left', left_on="movieId", right_on=Label.ITEM_ID)
        items_df.drop(columns=["movieId"], inplace=True)
        items_df.rename(columns={Label.ITEM_ID + "_x": Label.ITEM_ID}, inplace=True)
        items_df.drop(columns=[Label.ITEM_ID + "_y"], inplace=True)
        #
        items_df.dropna(inplace=True)
        items_df.drop_duplicates(subset=[Label.ITEM_ID], inplace=True)

        self.set_items(new_items=items_df[items_df[Label.GENRES] != '(no genres listed)'])
        self.items.to_csv(os.path.join(self.clean_dataset_dir, PathDirFile.ITEMS_FILE), index=False)
