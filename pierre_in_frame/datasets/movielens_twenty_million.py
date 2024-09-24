from collections import Counter

import numpy as np
import os
import pandas as pd

from datasets.utils.base_preprocess import Dataset
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class MovielensTwentyMillion(Dataset):
    """
    Movielens 20 Million dataset.
    This class organizes the work with the dataset.
    """
    # Class information.
    dir_name = "ml-20m"
    verbose_name = "Movielens Twenty Million"
    system_name = "ml-20m"

    # Raw paths.
    dataset_raw_path = "/".join([PathDirFile.RAW_DATASETS_DIR, dir_name])
    raw_transaction_file = "ratings.csv"
    raw_items_file = "movies.csv"

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
        # Constant Values
        self.cut_value = 4
        self.item_cut_value = 10
        self.profile_len_cut_value = 200

    # ######################################### #
    # ############# Transactions ############## #
    # ######################################### #

    def load_raw_transactions(self):
        """
        Load raw transactions into the instance variable.
        """
        self.raw_transactions = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_transaction_file),
            engine='python', sep=','
        )
        self.raw_transactions.rename(
            columns={
                'userId': Label.USER_ID,
                'movieId': Label.ITEM_ID,
                'rating': Label.TRANSACTION_VALUE,
                'timestamp': Label.TIME,
            }, inplace=True
        )

    def clean_transactions(self):
        """
        Cleaning the raw transactions and save as clean transactions.
        """
        super().clean_transactions()

        # Load the raw transactions.
        raw_transactions = self.get_raw_transactions().astype({
            Label.USER_ID: 'int32',
            Label.ITEM_ID: 'int32',
            Label.TRANSACTION_VALUE: 'int32'
        })

        # Filter transactions based on the items id list.
        filtered_raw_transactions = raw_transactions[
            raw_transactions[Label.ITEM_ID].isin(
                self.items[Label.ITEM_ID].tolist()
            )
        ]

        # Cut users and set the new data into the instance.
        self.set_transactions(
            new_transactions=MovielensTwentyMillion.cut_users(
                transactions=filtered_raw_transactions, item_cut_value=self.cut_value,
                profile_len_cut_value=self.profile_len_cut_value
            )
        )
        self.set_transactions(
            new_transactions=MovielensTwentyMillion.cut_item(
                transactions=self.transactions, item_cut_value=self.item_cut_value
            )
        )
        self.set_transactions(
            new_transactions=MovielensTwentyMillion.cut_users(
                transactions=filtered_raw_transactions, item_cut_value=self.cut_value,
                profile_len_cut_value=self.profile_len_cut_value
            )
        )
        self.set_items(
            new_items=self.items[
                self.items[Label.ITEM_ID].isin(self.transactions[Label.ITEM_ID].unique().tolist())
            ]
        )

        if Constants.NORMALIZED_SCORE:
            self.transactions[Label.TRANSACTION_VALUE] = np.where(
                self.transactions[Label.TRANSACTION_VALUE] >= self.cut_value, 1, 0
            )

        self.reset_indexes()

        # Save the clean transactions as CSV.
        count_user_trans = Counter(self.transactions[Label.USER_ID].tolist())
        min_c = min(list(count_user_trans.values()))
        max_c = max(list(count_user_trans.values()))
        print(f"Maximum: {max_c}")
        print(f"Minimum: {min_c}")
        self.transactions.to_csv(
            os.path.join(self.clean_dataset_dir, PathDirFile.TRANSACTIONS_FILE),
            index=False
        )
        self.items.to_csv(os.path.join(
            self.clean_dataset_dir, PathDirFile.ITEMS_FILE),
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
            os.path.join(self.dataset_raw_path, self.raw_items_file), engine='python',
            sep=','
        )
        self.raw_items.rename(
            columns={
                'movieId': Label.ITEM_ID,
                'title': Label.TITLE,
                'genres': Label.GENRES,
            }, inplace=True
        )

    def clean_items(self):
        """
        Cleaning the raw items and save as clean items.
        """
        # Load the raw items.
        raw_items_df = self.get_raw_items()

        # Clean the items without information and with the label indicating no genre in the item.
        raw_items_df.dropna(inplace=True)
        genre_clean_items = raw_items_df[raw_items_df[Label.GENRES] != '(no genres listed)'].copy()

        # Set the new data into the instance.
        self.set_items(new_items=genre_clean_items)
        self.items.drop_duplicates(subset=[Label.ITEM_ID], inplace=True)

        self.items = self.items.astype({
            Label.ITEM_ID: 'int32'
        })

        # Save the clean transactions as CSV.
        self.items.to_csv(os.path.join(self.clean_dataset_dir, PathDirFile.ITEMS_FILE), index=False)
