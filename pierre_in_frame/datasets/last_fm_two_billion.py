from collections import Counter

from datetime import datetime

import json
import numpy as np
import os
import pandas as pd

from datasets.utils.base_preprocess import Dataset
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class LastFMTwoBillion(Dataset):
    """
    Last FM 2 Billion dataset.
    This class organizes the work with the dataset.
    """
    # Class information.
    dir_name = "lfm-2b-subset"
    verbose_name = "Last FM Two Billion Subset"
    system_name = "lfm-2b-subset"

    # Raw paths.
    dataset_raw_path = "/".join([PathDirFile.RAW_DATASETS_DIR, dir_name])
    raw_transaction_file = "listening_events.tsv"
    raw_combined_transaction_file = "listening_play_count.csv"
    raw_items_file = "tags-micro-genres.json"

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
        self.raw_transactions_events = None
        # Constant Values
        self.cut_value = 3
        self.item_cut_value = 10
        self.profile_len_cut_value = 100

    # ######################################### #
    # ############# Transactions ############## #
    # ######################################### #

    def load_raw_transactions(self):
        """
        Load raw transactions into the instance variable.
        """
        self.raw_transactions = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_transaction_file),
            engine='python', sep='\t'
        )
        self.raw_transactions.rename(
            columns={
                'user_id': Label.USER_ID,
                'track_id': Label.ITEM_ID,
                'album_id': Label.ALBUM,
                'timestamp': Label.TIME,
            }, inplace=True
        )

    def filtering_transations(self, raw_transactions):
        combined_raw_transactions = raw_transactions.groupby(
            [Label.USER_ID, Label.ITEM_ID]
        ).size().reset_index().rename(
            columns={0: Label.TRANSACTION_VALUE}
        ).sort_values(
            by=[Label.USER_ID, Label.ITEM_ID], ascending=False
        )

        df_f = raw_transactions.drop_duplicates(subset=[Label.USER_ID, Label.ITEM_ID], keep='last').copy()

        df_ordered = df_f.sort_values(by=[Label.USER_ID, Label.ITEM_ID], ascending=False).copy()
        combined_raw_transactions[Label.TIME] = df_ordered[Label.TIME].to_list()
        return combined_raw_transactions

    def clean_transactions(self):
        """
        Cleaning the raw transactions and save as clean transactions.
        """
        super().clean_transactions()

        # Load the raw transactions.
        raw_transactions = self.get_raw_transactions()
        combined_raw_transactions = self.filtering_transations(raw_transactions)

        combined_raw_transactions = combined_raw_transactions.astype({
            Label.USER_ID: 'int32',
            Label.ITEM_ID: 'int32'
        })

        # Filter transactions based on the items id list.
        filtered_raw_transactions = combined_raw_transactions[
            combined_raw_transactions[Label.ITEM_ID].isin(self.items[Label.ITEM_ID].tolist())]
        filtered_raw_transactions.to_csv(
            os.path.join(self.dataset_raw_path, self.raw_combined_transaction_file),
            index=False
        )
        # Cut users and set the new data into the instance.
        self.set_transactions(
            new_transactions=LastFMTwoBillion.cut_users(
                transactions=filtered_raw_transactions, item_cut_value=self.cut_value,
                profile_len_cut_value=self.profile_len_cut_value
            )
        )
        self.set_transactions(
            new_transactions=LastFMTwoBillion.cut_item(
                self.transactions, self.item_cut_value
            )
        )
        self.set_transactions(
            new_transactions=LastFMTwoBillion.cut_users(
                transactions=filtered_raw_transactions, item_cut_value=self.cut_value,
                profile_len_cut_value=self.profile_len_cut_value
            )
        )
        self.set_items(
            new_items=self.items[self.items[Label.ITEM_ID].isin(self.transactions[Label.ITEM_ID].unique().tolist())]
        )

        if Constants.NORMALIZED_SCORE:
            self.transactions[Label.TRANSACTION_VALUE] = np.where(
                self.transactions[Label.TRANSACTION_VALUE] >= self.cut_value, 1, 0)

        self.reset_indexes()
        self.transactions[Label.TIME] = self.transactions[Label.TIME].apply(
            lambda dtimestamp: int(round(datetime.strptime(dtimestamp, '%Y-%m-%d %H:%M:%S').timestamp()))
        )

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
        self.items.to_csv(os.path.join(self.clean_dataset_dir, PathDirFile.ITEMS_FILE), index=False)

    # ######################################### #
    # ################# Items ################# #
    # ######################################### #

    def load_raw_items(self):
        """
        Load Raw Items into the instance variable.
        """

        def make_dict(line_str: str):
            line = json.loads(line_str)
            result = {
                Label.ITEM_ID: int(line["i"]),
                Label.ARTIST: line["_id"]["artist"],
                Label.TRACK_ID: line["_id"]["track"],
                Label.GENRES: "|".join(list(line["tags"].keys())),
            }
            return result

        path_to_open = "/".join([LastFMTwoBillion.dataset_raw_path, LastFMTwoBillion.raw_items_file])

        f = open(path_to_open, 'rt', encoding='utf-8')
        items_list = list(map(make_dict, f))
        f.close()

        self.raw_items = pd.DataFrame.from_dict(items_list)

    def clean_items(self):
        """
        Cleaning the raw items and save as clean items.
        """
        # Load the raw items.
        raw_items_df = self.get_raw_items()

        # Clean the items without information and with the label indicating no genre in the item.
        raw_items_df.dropna(inplace=True)
        genre_clean_items = raw_items_df[raw_items_df[Label.GENRES] != '']

        # Set the new data into the instance.
        self.set_items(new_items=genre_clean_items)
        self.items.drop_duplicates(subset=[Label.ITEM_ID], inplace=True)

        self.items = self.items.astype({
            Label.ITEM_ID: 'int32'
        })

        # Save the clean transactions as CSV.
        self.items.to_csv(os.path.join(self.clean_dataset_dir, PathDirFile.ITEMS_FILE), index=False)
