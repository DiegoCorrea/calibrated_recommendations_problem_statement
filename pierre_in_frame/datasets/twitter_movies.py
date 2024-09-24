import itertools
import os

import numpy as np
import pandas as pd

from datasets.utils.base_preprocess import Dataset
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class TwitterMovies(Dataset):
    """
    Twitter Movies dataset.
    This class organize the work with the dataset.
    """
    # Class information.
    dir_name = "twitter_movies"
    verbose_name = "Twitter Movies"
    system_name = "twitter_movies"

    # Raw paths.
    dataset_raw_path = "/".join([PathDirFile.RAW_DATASETS_DIR, dir_name])
    raw_transaction_file = "ratings.dat"
    raw_items_file = "movies.dat"

    # Clean paths.
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
        Load raw transactions into the instance variable.
        """
        self.raw_transactions = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_transaction_file),
            names=[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE, Label.TIME],
            sep='::', engine='python'
        )

    def clean_transactions(self):
        """
        Cleaning the raw transactions and save as clean transactions.
        """
        super().clean_transactions()

        # Load the raw transactions.
        raw_transactions = self.get_raw_transactions()

        # Filter transactions based on the items id list.
        filtered_raw_transactions = raw_transactions[
            raw_transactions[Label.ITEM_ID].isin(self.items[Label.ITEM_ID].tolist())]

        # Cut users and set the new data into the instance.
        self.set_transactions(
            new_transactions=TwitterMovies.cut_users(filtered_raw_transactions, 8))

        if Constants.NORMALIZED_SCORE:
            self.transactions[Label.TRANSACTION_VALUE] = np.where(self.transactions[Label.TRANSACTION_VALUE] >= 8, 1, 0)

        # Save the clean transactions as CSV.
        self.transactions.to_csv(
            os.path.join(self.clean_dataset_dir, PathDirFile.TRANSACTIONS_FILE),
            index=False
        )

    # ######################################### #
    # ################# Items ################# #
    # ######################################### #

    def load_raw_items(self):
        """
        Load Raw Items into the instance variable.
        """
        self.raw_items = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_items_file),
            sep='::', encoding='ISO-8859-1', engine='python',
            names=[Label.ITEM_ID, Label.TITLE, Label.GENRES]
        )

    def clean_items(self):
        """
        Cleaning the raw items and save as clean items.
        """
        # Load the raw items.
        raw_items_df = self.get_raw_items()

        # Clean the items without information and with the label indicating no genre in the item.
        raw_items_df.dropna(inplace=True)
        raw_items_df[Label.GENRES] = raw_items_df[Label.GENRES].astype(str)
        genre_clean_items = raw_items_df[raw_items_df[Label.GENRES] != '(no genres listed)']
        genre_clean_items = genre_clean_items[genre_clean_items[Label.GENRES] != '']

        # Set the new data into the instance.
        self.set_items(new_items=genre_clean_items)
        self.items.drop_duplicates(subset=[Label.ITEM_ID], inplace=True)

        # Save the clean transactions as CSV.
        self.items.to_csv(os.path.join(self.clean_dataset_dir, PathDirFile.ITEMS_FILE), index=False)

    def raw_data_basic_info(self):
        self.load_raw_items()
        self.load_raw_transactions()

        def classes(item):
            if str(item) == '':
                return ''
            splitted = item.split('|')
            return [c for c in splitted]

        total_of_users = len(self.raw_transactions[Label.USER_ID].unique())
        total_of_items = len(self.raw_items)
        total_of_transactions = len(self.raw_transactions)
        self.raw_items[Label.GENRES] = self.raw_items[Label.GENRES].astype(str)
        total_of_classes = len(
            set(list(itertools.chain.from_iterable(list(map(classes, self.raw_items[Label.GENRES].tolist()))))))
        print("RAW DATASET INFORMATION")
        print("Total of Users: ", total_of_users)
        print("Total of Items: ", total_of_items)
        print("Total of Transactions: ", total_of_transactions)
        print("Total of Classes: ", total_of_classes)