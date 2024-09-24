import os

import numpy as np
import pandas as pd

from datasets.utils.base_preprocess import Dataset
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class TasteProfile(Dataset):
    """
    Taste Profile dataset.
    This class organize the work with the dataset.
    """
    dir_name = "taste-profile"
    verbose_name = "Taste Profile"
    system_name = "taste-profile"
    # Raw
    dataset_raw_path = "/".join([PathDirFile.RAW_DATASETS_DIR, dir_name])
    raw_transaction_file = "train_triplets.txt"
    raw_tracks_file = "unique_tracks.txt"
    raw_items_file = "songs.csv"
    raw_genres_file = "msd_tagtraum_cd2.cls"
    raw_genres = None
    raw_track_items = None
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
        self.raw_transactions = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_transaction_file),
            names=[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE],
            sep='\t'
        )

    def clean_transactions(self):
        """
        Cleaning the raw transactions and save as clean transactions
        """
        super().clean_transactions()
        raw_transactions_df = self.get_raw_transactions()
        self.items = self.get_items()

        self.set_transactions(new_transactions=TasteProfile.cut_users(
            raw_transactions_df[
                raw_transactions_df[Label.ITEM_ID].isin(self.items[Label.ITEM_ID].tolist())
            ], 3
        ))
        self.set_items(
            new_items=self.items[self.items[Label.ITEM_ID].isin(self.transactions[Label.ITEM_ID].unique().tolist())]
        )

        if Constants.NORMALIZED_SCORE:
            self.transactions[Label.TRANSACTION_VALUE] = np.where(self.transactions[Label.TRANSACTION_VALUE] >= 3, 1, 0)

        self.transactions.to_csv(os.path.join(self.clean_dataset_dir, PathDirFile.TRANSACTIONS_FILE), index=False)
        self.items.to_csv(os.path.join(self.clean_dataset_dir, PathDirFile.ITEMS_FILE), index=False)

    # ######################################### #
    # ################# Items ################# #
    # ######################################### #

    def load_raw_items(self):
        """
        Load Raw Items
        """
        self.raw_items = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_items_file), engine='python',
            dtype=str, sep=',', header=None,
            names=[Label.ITEM_ID, Label.TITLE, Label.ALBUM, Label.ARTIST, Label.YEAR]
        )
        self.raw_items.drop([Label.ALBUM, Label.ARTIST, Label.YEAR], axis=1, inplace=True)

        self.raw_track_items = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_tracks_file), engine='python',
            dtype=str, sep='<SEP>', header=None,
            names=[Label.TRACK_ID, Label.ITEM_ID, Label.TITLE, Label.ARTIST]
        )
        self.raw_track_items.drop([Label.TITLE, Label.ARTIST], axis=1, inplace=True)

        self.raw_genres = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_genres_file),
            sep='\t', names=[Label.TRACK_ID, Label.MAJORITY_GENRE, Label.MINORITY_GENRE], na_values=' '
        )

    def clean_items(self):
        """
        Cleaning the raw items and save as clean items
        """
        # Load transactions set from csv
        raw_items_df = self.get_raw_items()
        raw_genres_df = self.raw_genres
        raw_track_df = self.raw_track_items
        # Genres
        raw_genres_df = raw_genres_df.replace(np.nan, '', regex=True)
        raw_genres_df[Label.GENRES] = raw_genres_df.apply(
            lambda r: r[Label.MAJORITY_GENRE] + '|' + r[Label.MINORITY_GENRE] if r[Label.MINORITY_GENRE] != '' else r[
                Label.MAJORITY_GENRE], axis=1)
        raw_genres_df.drop([Label.MAJORITY_GENRE, Label.MINORITY_GENRE], inplace=True, axis=1)
        #
        self.items = pd.merge(
            pd.merge(
                raw_items_df, raw_track_df,
                how='left', left_on=Label.ITEM_ID, right_on=Label.ITEM_ID
            ),
            raw_genres_df, how='inner', left_on=Label.TRACK_ID, right_on=Label.TRACK_ID
        )
        self.items.drop_duplicates([Label.ITEM_ID], inplace=True)
        self.items.set_index(Label.TRACK_ID, inplace=True, drop=True)
        #

        self.items.to_csv(os.path.join(self.clean_dataset_dir, PathDirFile.ITEMS_FILE), index=False)
