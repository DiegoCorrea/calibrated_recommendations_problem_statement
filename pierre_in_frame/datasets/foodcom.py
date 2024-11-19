from datetime import datetime

import itertools
from collections import Counter

import ast

import numpy as np
import os
import pandas as pd
from numpy import mean
from numpy.ma import median

from datasets.utils.base_preprocess import Dataset
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class FoodComRecipe(Dataset):
    """
    Food.com Recipes dataset.
    This class organizes the work with the dataset.
    """
    # Class information.
    dir_name = "food"
    verbose_name = "Food.com Recipe"
    system_name = "food"

    # Raw paths.
    dataset_raw_path = "/".join([PathDirFile.RAW_DATASETS_DIR, dir_name])
    raw_transaction_file = "RAW_interactions.csv"
    raw_items_file = "RAW_recipes.csv"

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
        self.translation_index_items = None

        # Constant Values
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
        self.raw_transactions = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_transaction_file),
            usecols=[0, 1, 2, 3],
            engine='python', sep=','
        )
        self.raw_transactions.rename(
            columns={
                'user_id': Label.USER_ID,
                'recipe_id': Label.ITEM_ID,
                'rating': Label.TRANSACTION_VALUE,
                'date': Label.TIME,
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
            new_transactions=FoodComRecipe.cut_users(
                transactions=filtered_raw_transactions, item_cut_value=self.cut_value,
                profile_len_cut_value=self.profile_len_cut_value
            )
        )
        self.set_transactions(
            new_transactions=FoodComRecipe.cut_item(
                transactions=self.transactions, item_cut_value=self.item_cut_value
            )
        )
        self.set_transactions(
            new_transactions=FoodComRecipe.cut_users(
                transactions=self.transactions, item_cut_value=self.cut_value,
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
        self.transactions[Label.TIME] = self.transactions[Label.TIME].apply(lambda dtimestamp: int(round(datetime.strptime(dtimestamp, '%Y-%m-%d').timestamp())))
        # Save the clean transactions as CSV.
        count_user_trans = Counter(self.transactions[Label.USER_ID].tolist())
        min_c = min(list(count_user_trans.values()))
        max_c = max(list(count_user_trans.values()))
        print(f"Maximum: {max_c}")
        print(f"Minimum: {min_c}")
        self.transactions.to_csv(
            os.path.join(self.clean_dataset_dir, PathDirFile.TRANSACTIONS_FILE),
            index=False,
            mode='w+'
        )
        self.items.to_csv(os.path.join(
            self.clean_dataset_dir, PathDirFile.ITEMS_FILE),
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
        self.raw_items = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_items_file), engine='python',
            sep=',', usecols=[1, 5]
        )
        self.raw_items.rename(
            columns={
                'id': Label.ITEM_ID,
                'tags': Label.GENRES,
            }, inplace=True
        )
        self.raw_items[Label.GENRES] = self.raw_items[Label.GENRES].apply(lambda tags: "|".join(ast.literal_eval(tags)))

    def clean_items(self):
        """
        Cleaning the raw items and save as clean items.
        """

        # Load the raw items.
        raw_items_df = self.get_raw_items()

        # Clean the items without information and with the label indicating no genre in the item.
        raw_items_df.dropna(inplace=True)
        # raw_items_df[Label.GENRES] = raw_items_df[Label.GENRES].apply(lambda tags: "|".join(ast.literal_eval(tags)))
        genre_clean_items = raw_items_df[raw_items_df[Label.GENRES] != ''].copy()

        # Set the new data into the instance.
        self.set_items(new_items=genre_clean_items)
        self.items.drop_duplicates(subset=[Label.ITEM_ID], inplace=True)

        self.items = self.items.astype({
            Label.ITEM_ID: 'int32'
        })

        # Save the clean transactions as CSV.
        self.items.to_csv(
            os.path.join(self.clean_dataset_dir, PathDirFile.ITEMS_FILE),
            mode='w+', index=False
        )

    def raw_data_basic_info(self):
        """
        This method is to print the raw basic information
        """
        self.load_raw_items()
        self.load_raw_transactions()

        total_of_users = len(self.raw_transactions[Label.USER_ID].unique())

        count_user_trans = Counter(self.raw_transactions[Label.USER_ID].tolist())
        mean_c = round(mean(list(count_user_trans.values())), 3)
        median_c = round(median(list(count_user_trans.values())), 3)

        total_of_items = len(self.raw_items)
        total_of_transactions = len(self.raw_transactions)
        raw_items_df = self.raw_items.copy()
        raw_items_df[Label.GENRES] = self.raw_items[Label.GENRES].apply(lambda tags: "|".join(ast.literal_eval(tags)))

        total_of_classes = len(
            set(list(itertools.chain.from_iterable(list(map(Dataset.classes, raw_items_df[Label.GENRES].tolist()))))))
        return pd.DataFrame(
            data=[['Raw', total_of_users, total_of_items, total_of_transactions, total_of_classes, mean_c, median_c]],
            columns=['Dataset', 'Users', 'Items', 'Transactions', 'Classes', "Users_trans_mean", "Users_trans_median"]
        )
