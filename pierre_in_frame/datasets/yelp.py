from collections import Counter
import json

import os
from datetime import datetime

import pandas as pd
import numpy as np

from datasets.utils.base_preprocess import Dataset
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class Yelp(Dataset):
    """
    Yelp dataset.
    This class organizes the work with the dataset.
    """
    # Class information.
    dir_name = "yelp"
    verbose_name = "Yelp Dataset"
    system_name = "yelp"

    # Raw paths.
    dataset_raw_path = "/".join([PathDirFile.RAW_DATASETS_DIR, dir_name])
    raw_transaction_file = "yelp_academic_dataset_review.json"
    raw_items_file = "yelp_academic_dataset_business.json"

    # Clean paths.
    dataset_clean_path = "/".join([PathDirFile.CLEAN_DATASETS_DIR, dir_name])

    # Constant Values

    # ######################################### #
    # ############## Constructor ############## #
    # ######################################### #

    def __init__(self):
        """
        Class constructor. Firstly call the super constructor and after start personalized things.
        """
        super().__init__()
        self.cut_value = 4
        self.item_cut_value = 5
        self.profile_len_cut_value = 100

    # ######################################### #
    # ############# Transactions ############## #
    # ######################################### #

    def load_raw_transactions(self):
        """
        Load raw transactions into the instance variable.
        """
        print("Loading Raw Transactions")
        def make_dict(line_str: str):
            data = json.loads(line_str)
            cleaned_line = {
                Label.USER_ID: data["user_id"],
                Label.ITEM_ID: data["business_id"],
                Label.TRANSACTION_VALUE: data["stars"],
                Label.TIME: data["date"]
            }
            return cleaned_line
        data_file = open("/".join([Yelp.dataset_raw_path, Yelp.raw_transaction_file]))
        items_list = list(map(make_dict, data_file))
        data_file.close()
        self.raw_transactions = pd.DataFrame.from_dict(items_list)


    def clean_transactions(self):
        """
        Cleaning the raw transactions and save as clean transactions.
        """
        super().clean_transactions()

        # Load the raw transactions.
        raw_transactions = self.get_raw_transactions()

        print("Cleaning Dataset")
        # Filter transactions based on the items id list.
        filtered_raw_transactions = raw_transactions[
            raw_transactions[Label.ITEM_ID].isin(self.items[Label.ITEM_ID].tolist())]

        # Cut users and set the new data into the instance.
        self.set_transactions(
            new_transactions=Yelp.cut_users(
                transactions=filtered_raw_transactions, item_cut_value=self.cut_value,
                profile_len_cut_value=self.profile_len_cut_value
            )
        )
        self.set_transactions(
            new_transactions=Yelp.cut_item(
                transactions=self.transactions, item_cut_value=self.item_cut_value
            )
        )
        self.set_transactions(
            new_transactions=Yelp.cut_users(
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

        print("Re-Indexing Dataset")
        self.reset_indexes()
        self.transactions[Label.TIME] = self.transactions[Label.TIME].apply(lambda dtimestamp: int(round(datetime.strptime(dtimestamp, '%Y-%m-%d %H:%M:%S').timestamp())))

        # Save the clean transactions as CSV.
        count_user_trans = Counter(self.transactions[Label.USER_ID].tolist())
        min_c = min(list(count_user_trans.values()))
        max_c = max(list(count_user_trans.values()))
        print(f"Maximum: {max_c}")
        print(f"Minimum: {min_c}")
        # self.transactions = self.transactions.astype({
        #     Label.USER_ID: 'int32',
        #     Label.ITEM_ID: 'int32'
        # })
        self.transactions.to_csv(
            os.path.join(self.clean_dataset_dir, PathDirFile.TRANSACTIONS_FILE),
            index=False,
            mode='w+'
        )
        self.items.to_csv(
            os.path.join(self.clean_dataset_dir, PathDirFile.ITEMS_FILE),
            index=False,
            mode='w+'
        )

    # ######################################### #
    # ################# Items ################# #
    # ######################################### #

    def load_raw_items(self):
        """
        Load Raw Items into the instance variable.
        """

        def make_dict(line_str):
            data = json.loads(line_str)
            if data["categories"] is None:
                data["categories"] = "(no genres listed)"
            cleaned_line = {
                Label.ITEM_ID: data["business_id"],
                # Label.GENRES: data.categories,
                Label.GENRES: "|".join(list(data["categories"].split(", ")))
            }
            return cleaned_line

        print("Loading Raw Items")
        data_file = open("/".join([Yelp.dataset_raw_path, Yelp.raw_items_file]))
        items_list = list(map(make_dict, data_file))
        data_file.close()

        self.raw_items = pd.DataFrame.from_dict(items_list)

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
        #
        # self.items = self.items.astype({
        #     Label.ITEM_ID: 'int32'
        # })

        # Save the clean transactions as CSV.
        self.items.to_csv(
            os.path.join(self.clean_dataset_dir, PathDirFile.ITEMS_FILE),
            index=False,
            mode='w+'
        )
