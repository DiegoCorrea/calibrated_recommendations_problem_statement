import shutil
from collections import Counter
from copy import deepcopy

from statistics import median

import itertools
import logging
import os

import pandas as pd
from numpy import mean, array_split

from datasets.utils import split
from datasets.utils.kfold import KFoldTrainValidationTest
from datasets.utils.split import SequentialTimeSplit, CrossValidationThroughTime
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class Dataset:
    """
    Dataset class to be inherited.
    To be inheritance by all registered dataset.
    """
    # Class information.
    dir_name = ""
    verbose_name = ""
    system_name = ""

    # Raw paths.
    dataset_raw_path = ""
    raw_transaction_file = ""
    raw_items_file = ""

    # Clean paths.
    dataset_clean_path = PathDirFile.CLEAN_DATASETS_DIR
    PROFILE_LEN_CUT_VALUE = 150

    # ######################################### #
    # ############## Constructor ############## #
    # ######################################### #

    def __init__(self):
        """
        Base Dataset Class Constructor (super).
        """
        # Creating the raw data variables.
        self.raw_transactions = None
        self.raw_items = None

        # Creating the clean data variables.
        self.transactions = None
        self.items = None

        # Fold
        self.full_train_transaction = None
        self.train_transaction = None
        self.validation_transaction = None
        self.test_transaction = None

        # Domain Variable
        self.experiment_name = None
        self.cut_value = None
        self.item_cut_value = None
        self.profile_len_cut_value = None
        self.test_len_cut_value = None
        self.n_trials = None
        self.n_folds = None
        self.based_on = None

        # Creating the directory to lead with the clean data.
        self.clean_dataset_dir = None

    # ######################################### #
    # ############## Get and Sets ############# #
    # ######################################### #

    def set_experiment_variables(
            self,
            experiment_name: str, cut_value: int, item_cut_value: int,
            profile_len_cut_value: int, test_len_cut_value: int,
            n_trials: int, n_folds: int, based_on: str
    ):
        self.experiment_name = experiment_name
        self.cut_value = cut_value
        self.item_cut_value = item_cut_value
        self.profile_len_cut_value = profile_len_cut_value
        self.test_len_cut_value = test_len_cut_value
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.based_on = based_on
        self.create_clean_dir()

    def set_environment(
            self,
            experiment_name: str, based_on: str,
            n_trials: int = None, n_folds: int = None):
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.based_on = based_on
        self.experiment_name = experiment_name

    def get_dataset_name(self) -> str:
        return self.system_name

    # ######################################### #
    # ############# Transactions ############## #
    # ######################################### #

    # # RAW
    def load_raw_transactions(self):
        """
        Load Raw Transactions.
        The specific implementation is created by the children.
        """
        pass

    def get_raw_transactions(self) -> pd.DataFrame():
        """
        Get method to return the raw data transactions.
        :return: A pandas Dataframe with the dataset raw transactions.
        """
        # If it is the first requisition, load from the file
        if self.raw_transactions is None:
            self.load_raw_transactions()
        return self.raw_transactions

    # # CLEAN
    def clean_transactions(self):
        """
        Cleaning the raw transactions and save as clean transactions.
        The specific implementation is created by the children.
        """
        pass

    def load_clean_transactions(self):
        """
        Load clean transactions into the instance.
        """
        self.transactions = SaveAndLoad.load_clean_transactions(
            experiment_name=self.experiment_name,
            dataset=self.get_dataset_name(), based_on=self.based_on
        )

    def get_transactions(self) -> pd.DataFrame():
        """
        Get method to return the clean data transactions.
        :return: A pandas Dataframe with the dataset clean transactions.
        """
        # If it is the first requisition, load from the file
        if self.transactions is None:
            self.load_clean_transactions()
        return self.transactions

    def set_transactions(self, new_transactions: pd.DataFrame):
        """
        Set method to attribute the new transactions data to the instance.
        :param new_transactions: A pandas Dataframe with the new clean dataset transactions.
        """
        self.transactions = new_transactions

    # ############################################################################################ #
    # ################################ Train, Validation and Test Data ########################### #
    # ############################################################################################ #
    def get_full_train_transactions(self, trial: int, fold: int) -> pd.DataFrame:
        """
        Get the train transaction set.
        :param trial: An int that represents a number of an experimental trial.
        :param fold: An int that represents a number of a fold.
        :return: A pandas Dataframe with the train transactions.
        """
        # If it is the first requisition, load from the file
        if self.full_train_transaction is None:
            if self.train_transaction is None:
                self.load_train_transactions(trial=trial, fold=fold)

            if self.validation_transaction is None:
                self.load_validation_transactions(trial=trial, fold=fold)

            self.full_train_transaction = pd.concat(
                [self.train_transaction, self.validation_transaction]
            )
        return self.full_train_transaction

    def load_train_transactions(self, trial: int, fold: int):
        """
        Load a train transaction set.
        :param trial: An int that represents a number of an experimental trial.
        :param fold: An int that represents a number of a fold.
        """
        self.train_transaction = SaveAndLoad.load_train_transactions(
            experiment_name=self.experiment_name,
            dataset=self.get_dataset_name(), based_on=self.based_on,
            trial=trial, fold=fold
        )

    def get_train_transactions(self, trial: int, fold: int) -> pd.DataFrame:
        """
        Get the train transaction set.
        :param trial: An int that represents a number of an experimental trial.
        :param fold: An int that represents a number of a fold.
        :return: A pandas Dataframe with the train transactions.
        """
        # If it is the first requisition, load from the file
        self.load_train_transactions(trial=trial, fold=fold)
        return self.train_transaction

    def load_validation_transactions(self, trial: int, fold: int):
        """
        Load a validation transaction set.
        :param trial: An int that represents a number of an experimental trial.
        :param fold: An int that represents a number of a fold.
        """
        self.validation_transaction = SaveAndLoad.load_validation_transactions(
            experiment_name=self.experiment_name,
            dataset=self.get_dataset_name(), based_on=self.based_on,
            trial=trial, fold=fold
        )

    def get_validation_transactions(self, trial: int, fold: int) -> pd.DataFrame:
        """
        Get the train transaction set.
        :param trial: An int that represents a number of an experimental trial.
        :param fold: An int that represents a number of a fold.
        :return: A pandas Dataframe with the train transactions.
        """
        # If it is the first requisition, load from the file
        self.load_validation_transactions(trial=trial, fold=fold)
        return self.validation_transaction

    def load_test_transactions(self, trial: int, fold: int):
        """
        Load a test transaction set.
        :param trial: An int that represents a number of an experimental trial.
        :param fold: An int that represents a number of a fold.
        """
        self.test_transaction = SaveAndLoad.load_test_transactions(
            experiment_name=self.experiment_name,
            dataset=self.get_dataset_name(), based_on=self.based_on,
            trial=trial, fold=fold
        )

    def get_test_transactions(self, trial: int, fold: int):
        """
        Get the test transaction set.
        :param trial: An int that represents a number of an experimental trial.
        :param fold: An int that represents a number of a fold.
        :return: A pandas Dataframe with the test transactions.
        """
        # If it is the first requisition, load from the file
        self.load_test_transactions(trial, fold)
        return self.test_transaction

    # ######################################### #
    # ################# Items ################# #
    # ######################################### #

    # RAW
    def load_raw_items(self):
        """
        Load Raw Items.
        The specific implementation is created by the children.
        """
        pass

    def get_raw_items(self) -> pd.DataFrame():
        """
        Get method to return the raw items data.
        :return: A pandas Dataframe with the dataset raw items.
        """
        # If it is the first requisition, load from the file
        self.load_raw_items()
        return self.raw_items

    # CLEAN
    def clean_items(self):
        """
        Cleaning the raw items and save as clean items.
        The specific implementation is created by the children.
        """
        pass

    def load_clean_items(self):
        """
        Load clean items into the instance.
        """
        self.items = SaveAndLoad.save_clean_items(
            experiment_name=self.experiment_name,
            dataset=self.get_dataset_name(), based_on=self.based_on
        )

    def load_clean_dataset(self):
        """
        Load the clean transactions and items into the instance.
        """
        self.load_clean_transactions()
        self.load_clean_items()

    def get_items(self) -> pd.DataFrame():
        """
        Get method to return the clean items data.
        :return: A pandas Dataframe with the dataset clean items.
        """
        # If is the first requisition, load from the file
        self.load_clean_items()
        return self.items

    def set_items(self, new_items: pd.DataFrame):
        """
        Set method to attribute the new items data to the instance.
        :param new_items: A pandas Dataframe with the new clean dataset items.
        """
        self.items = new_items

    # ######################################### #
    # ################# Other ################# #
    # ######################################### #
    def dataset_location(self):
        self.clean_dataset_dir = "/".join(
            [PathDirFile.DATA_DIR, self.experiment_name,
             "datasets", self.get_dataset_name(), self.based_on]
        )

    def create_clean_dir(self):
        """
        Create the dataset directory to receive the clean data (pre-processed data).
        """
        self.dataset_location()
        if not os.path.exists(self.clean_dataset_dir):
            os.makedirs(self.clean_dataset_dir)

    def drop_time_column(self):
        """
        Drop the time column, if needed.
        """
        self.transactions.drop([Label.TIME], inplace=True, axis=1)

    # ############################################################################################ #
    # ################################ To choice Pre-Processing ################################## #
    # ############################################################################################ #

    def choosing_preprocessing(self):
        """
        Choosing the pre-processing method
        """
        if self.based_on == Label.TIME_SINGLE_SPLIT:
            self.mining_data_and_create_fold_based_on_time()
        elif self.based_on == Label.CVTT:
            self.mining_data_and_create_fold_based_on_cvtt()
        elif self.based_on == Label.CROSS_VALIDATION:
            self.mining_data_and_create_fold()
        elif self.based_on == Label.CROSS_TRAIN_VALIDATION_TEST:
            self.mining_data_and_create_fold_based_on_tvt()
        else:
            raise f"Invalid based_on value! {self.based_on} does not exists!"

    def clean_data(self):
        """
        Caller to clean the transactions and items
        """
        # Extract the items
        self.clean_items()
        # Extract the transactions
        self.clean_transactions()

    def delete_dataset(self):
        self.dataset_location()
        try:
            shutil.rmtree(self.clean_dataset_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        if not os.path.exists(self.clean_dataset_dir):
            os.makedirs(self.clean_dataset_dir)

    # ############################################################################################ #
    # ######################## K-fold Cross Validation Train-Validation-test ##################### #
    # ############################################################################################ #

    def mining_data_and_create_fold_based_on_tvt(self):
        """
        The raw dataset is preprocessed and the clean dataset produce n_trials with n_folds.
        """
        # Deleting old files from previous dataset pre-process
        self.delete_dataset()
        # Clean and filter the data
        self.clean_data()
        # Creating Folds
        self.create_folds_based_on_tvt()

    def create_folds_based_on_tvt(self) -> None:
        """
        Create all folds to be used by the system.
        The clean dataset produce n_trials with n_folds.
        """
        for trial in range(1, self.n_trials + 1):
            logger.info("+ Preparing trial: " + str(trial))
            instance = KFoldTrainValidationTest(
                transactions_df=self.transactions, n_trial=trial, n_folds=self.n_folds
            )
            train_list, valid_list, test_list = instance.main()
            for k in range(self.n_folds):
                train_df = pd.concat(train_list[k])
                validation_df = pd.concat(valid_list[k])
                test_df = pd.concat(test_list[k])

                logger.info("+ + Preparing fold: " + str(k + 1))
                fold_dir = "/".join([self.clean_dataset_dir, "trial-" + str(trial), "fold-" + str(k + 1)])
                if not os.path.exists(fold_dir):
                    os.makedirs(fold_dir)

                train_path = os.path.join(fold_dir, PathDirFile.TRAIN_FILE)
                if 'index' in train_df.columns.tolist():
                    train_df.drop(columns=['index'], inplace=True)
                train_df.to_csv(train_path, index=False, mode='w+')

                validation_path = os.path.join(fold_dir, PathDirFile.VALIDATION_FILE)
                if 'index' in validation_df.columns.tolist():
                    validation_df.drop(columns=['index'], inplace=True)
                validation_df.to_csv(validation_path, index=False, mode='w+')

                test_path = os.path.join(fold_dir, PathDirFile.TEST_FILE)
                if 'index' in test_df.columns.tolist():
                    test_df.drop(columns=['index'], inplace=True)
                test_df.to_csv(test_path, index=False, mode='w+')

    # ############################################################################################ #
    # ################################# K-fold Cross Validation ################################## #
    # ############################################################################################ #

    def mining_data_and_create_fold(self):
        """
        The raw dataset is preprocessed and the clean dataset produce n_trials with n_folds.
        """
        self.delete_dataset()
        # Clean and filter the data
        self.clean_data()
        # Creating Folds
        self.create_folds()

    def create_folds(self) -> None:
        """
        Create all folds to be used by the system.
        The clean dataset produce n_trials with n_folds.
        """
        for trial in range(1, self.n_trials + 1):
            logger.info("+ Preparing trial: " + str(trial))
            results = split.split_with_joblib(
                transactions_df=self.transactions, trial=trial, n_folds=self.n_folds
            )
            for k in range(self.n_folds):
                train_df, test_df = results[k]

                logger.info("+ + Preparing fold: " + str(k + 1))
                fold_dir = "/".join([self.clean_dataset_dir, "trial-" + str(trial), "fold-" + str(k + 1)])
                if not os.path.exists(fold_dir):
                    os.makedirs(fold_dir)

                train_path = os.path.join(fold_dir, PathDirFile.TRAIN_FILE)
                if 'index' in train_df.columns.tolist():
                    train_df.drop(columns=['index'], inplace=True)
                train_df.to_csv(train_path, index=False, mode='w+')

                test_path = os.path.join(fold_dir, PathDirFile.TEST_FILE)
                if 'index' in test_df.columns.tolist():
                    test_df.drop(columns=['index'], inplace=True)
                test_df.to_csv(test_path, index=False, mode='w+')

    # ############################################################################################ #
    # ################################## Sequential Validation ################################### #
    # ############################################################################################ #

    def mining_data_and_create_fold_based_on_time(self):
        """
        The raw dataset is preprocessed and the clean dataset produce n_trials with n_folds.
        """
        self.delete_dataset()
        # Clean and filter the data
        self.clean_data()
        # Creating Folds
        self.create_dataset_based_on_time()

    def create_dataset_based_on_time(self) -> None:
        """
        Create all folds to be used by the system.
        The clean dataset produce n_trials with n_folds.
        """
        self.transactions.sort_values(by=[Label.TIME], inplace=True)

        instance = SequentialTimeSplit(transactions_df=self.transactions, n_folds=self.n_folds)
        train_df, test_df = instance.main()

        item_cut_value = self.cut_value

        if Constants.NORMALIZED_SCORE:
            item_cut_value = 1

        test_df = self.cut_users(
            transactions=test_df, item_cut_value=item_cut_value,
            profile_len_cut_value=self.test_len_cut_value
        )

        train_df = train_df[
            train_df[Label.USER_ID].isin(test_df[Label.USER_ID].unique().tolist())
        ]

        logger.info("+ + Preparing fold: " + str(self.n_folds))
        fold_dir = "/".join([self.clean_dataset_dir, "trial-" + str(1), "fold-" + str(1)])
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        train_path = os.path.join(fold_dir, PathDirFile.TRAIN_FILE)
        if 'index' in train_df.columns.tolist():
            train_df.drop(columns=['index'], inplace=True)
        train_df.to_csv(train_path, index=False)

        test_path = os.path.join(fold_dir, PathDirFile.TEST_FILE)
        if 'index' in test_df.columns.tolist():
            test_df.drop(columns=['index'], inplace=True)
        test_df.to_csv(test_path, index=False)

        self.transactions = pd.concat([train_df, test_df])

        self.transactions.to_csv(
            str(os.path.join(self.clean_dataset_dir, PathDirFile.TRANSACTIONS_FILE)),
            index=False
        )

        self.items = self.items[self.items[Label.ITEM_ID].isin(self.transactions[Label.ITEM_ID])]

        # Rewriting keys
        self.items.sort_values(by=[Label.ITEM_ID], inplace=True)
        translation_index_items = {old_index: new_index for new_index, old_index in
                                   enumerate(self.items[Label.ITEM_ID].tolist())}
        self.items[Label.ITEM_ID] = [new_index for new_index, _ in enumerate(self.items[Label.ITEM_ID].tolist())]

        self.transactions[Label.ITEM_ID] = [translation_index_items[old_index] for old_index in
                                            self.transactions[Label.ITEM_ID].tolist()]
        train_df[Label.ITEM_ID] = [translation_index_items[old_index] for old_index in
                                   train_df[Label.ITEM_ID].tolist()]
        test_df[Label.ITEM_ID] = [translation_index_items[old_index] for old_index in
                                  test_df[Label.ITEM_ID].tolist()]

        self.transactions.sort_values(by=[Label.USER_ID], inplace=True)
        train_df.sort_values(by=[Label.USER_ID], inplace=True)
        test_df.sort_values(by=[Label.USER_ID], inplace=True)

        translation_index_user = {old_index: new_index for new_index, old_index in
                                  enumerate(self.transactions[Label.USER_ID].unique())}

        self.transactions[Label.USER_ID] = [translation_index_user[old_index] for old_index in
                                            self.transactions[Label.USER_ID].tolist()]
        train_df[Label.USER_ID] = [translation_index_user[old_index] for old_index in
                                   train_df[Label.USER_ID].tolist()]
        test_df[Label.USER_ID] = [translation_index_user[old_index] for old_index in
                                  test_df[Label.USER_ID].tolist()]

        # Preparing to Save
        self.transactions.sort_values(by=[Label.USER_ID], inplace=True)
        self.items.sort_values(by=[Label.ITEM_ID], inplace=True)
        self.transactions.reset_index(drop=True, inplace=True)
        self.items.reset_index(drop=True, inplace=True)
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        self.transactions.to_csv(
            str(os.path.join(self.clean_dataset_dir, PathDirFile.TRANSACTIONS_FILE)),
            index=False, mode='w+'
        )
        self.items.to_csv(
            os.path.join(self.clean_dataset_dir, PathDirFile.ITEMS_FILE),
            index=False, mode='w+'
        )
        train_df.to_csv(train_path, index=False, mode='w+')
        test_df.to_csv(test_path, index=False, mode='w+')
        print(self.transactions)
        print(self.transactions[Label.USER_ID])
        print(self.transactions[Label.USER_ID].max())
        print(max(self.transactions[Label.USER_ID].tolist()))
        print(self.transactions[Label.USER_ID].min())
        print(len(self.transactions[Label.USER_ID].unique()))

        set_1 = set({str(ix) for ix in self.transactions['ITEM_ID'].unique().tolist()})
        set_2 = set({str(ix) for ix in self.items['ITEM_ID'].unique().tolist()})

        if set_1 != set_2:
            raise IndexError(
                'There are a problem with the ITEM IDs. '
            )

    # ############################################################################################ #
    # ################################## CVTT Cross Validation ################################### #
    # ############################################################################################ #

    def mining_data_and_create_fold_based_on_cvtt(self):
        """
        The raw dataset is preprocessed and the clean dataset produce n_trials with n_folds.
        """
        self.delete_dataset()
        # Clean and filter the data
        self.clean_data()
        # Creating Folds
        self.create_fold_based_on_cvtt()

    def create_fold_based_on_cvtt(self) -> None:
        """
        Create all folds to be used by the system.
        The clean dataset produce n_trials with n_folds.
        """
        self.transactions.sort_values(by=[Label.TIME], inplace=True)

        instance = CrossValidationThroughTime(transactions_df=self.transactions, n_folds=self.n_folds)
        train_list, valid_list, test_list = instance.main()
        for k in range(self.n_folds):
            train_df = pd.concat(train_list[k])
            valid_df = pd.concat(valid_list[k])
            test_df = pd.concat(test_list[k])

            logger.info("+ + Preparing fold: " + str(k + 1))
            fold_dir = "/".join([self.clean_dataset_dir, "trial-" + str(1), "fold-" + str(k + 1)])
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)

            train_path = os.path.join(fold_dir, PathDirFile.TRAIN_FILE)
            if 'index' in train_df.columns.tolist():
                train_df.drop(columns=['index'], inplace=True)
            train_df.to_csv(train_path, index=False, mode='w+')

            valid_path = os.path.join(fold_dir, PathDirFile.VALIDATION_FILE)
            if 'index' in valid_df.columns.tolist():
                valid_df.drop(columns=['index'], inplace=True)
            valid_df.to_csv(valid_path, index=False, mode='w+')

            test_path = os.path.join(fold_dir, PathDirFile.TEST_FILE)
            if 'index' in test_df.columns.tolist():
                test_df.drop(columns=['index'], inplace=True)
            test_df.to_csv(test_path, index=False, mode='w+')

    # ############################################################################################ #
    # ####################################### Cut Methods ######################################## #
    # ############################################################################################ #

    @staticmethod
    def cut_users(
            transactions: pd.DataFrame,
            item_cut_value: float = 4, profile_len_cut_value: int = Constants.PROFILE_LEN_CUT_VALUE
    ) -> pd.DataFrame:
        """
        Cleaning the raw transactions and save as clean transactions.
        The specific implementation is created by the children.

        :return: A pandas Dataframe with the users transactions.
        """
        lower_transactions = transactions[transactions[Label.TRANSACTION_VALUE] < item_cut_value].copy()
        higher_transactions = transactions[transactions[Label.TRANSACTION_VALUE] >= item_cut_value]

        user_counts = higher_transactions[Label.USER_ID].value_counts()
        selected_users = [k for k, v in user_counts.items() if v > profile_len_cut_value]
        return pd.concat(
            [higher_transactions[higher_transactions[Label.USER_ID].isin(selected_users)].copy(),
             lower_transactions[lower_transactions[Label.USER_ID].isin(selected_users)].copy()])

    @staticmethod
    def cut_item(
        transactions: pd.DataFrame, item_cut_value: float = 5
    ) -> pd.DataFrame:
        item_counts = transactions[Label.ITEM_ID].value_counts()
        selected_items = [k for k, v in item_counts.items() if v > item_cut_value]
        return transactions[transactions[Label.ITEM_ID].isin(selected_items)].copy()

    def reset_indexes(self):
        self.items.sort_values(by=[Label.ITEM_ID], inplace=True)
        translation_index_items = {old_index: new_index for new_index, old_index in
                                   enumerate(self.items[Label.ITEM_ID].tolist())}
        self.items[Label.ITEM_ID] = [new_index for new_index, _ in enumerate(self.items[Label.ITEM_ID].tolist())]
        self.transactions[Label.ITEM_ID] = [translation_index_items[old_index] for old_index in
                                            self.transactions[Label.ITEM_ID].tolist()]

        self.transactions.sort_values(by=[Label.USER_ID], inplace=True)
        translation_index_user = {old_index: new_index for new_index, old_index in
                                  enumerate(self.transactions[Label.USER_ID].unique())}
        self.transactions[Label.USER_ID] = [translation_index_user[old_index] for old_index in
                                            self.transactions[Label.USER_ID].tolist()]
        self.transactions.reset_index(drop=True, inplace=True)
        self.items.reset_index(drop=True, inplace=True)

    # ######################################### #
    # ############# Data Analyze ############## #
    # ######################################### #
    @staticmethod
    def classes(item):
        """
        This method is to split the item genre.
        """
        splitted = item.split('|')
        return [
            c for c in splitted
        ]

    def raw_data_basic_info(self):
        """
        This method is to print the raw basic information
        """
        self.load_raw_items()
        self.load_raw_transactions()

        total_of_users = len(self.raw_transactions[Label.USER_ID].unique())

        count_user_trans = Counter(self.raw_transactions[Label.USER_ID].tolist())
        mean_c = round(mean(list(count_user_trans.values())), 3)
        median_c = median(list(count_user_trans.values()))
        min_c = min(list(count_user_trans.values()))
        max_c = max(list(count_user_trans.values()))

        total_of_items = len(self.raw_items)
        total_of_transactions = len(self.raw_transactions)
        total_of_classes = len(
            set(list(itertools.chain.from_iterable(list(map(Dataset.classes, self.raw_items[Label.GENRES].tolist()))))))
        return pd.DataFrame(
            data=[['Raw', total_of_users, total_of_items, total_of_transactions, total_of_classes, mean_c, median_c, min_c, max_c]],
            columns=['Dataset', 'Users', 'Items', 'Transactions', 'Classes', "Users_trans_mean", "Users_trans_median", "Minimum", "Maximum"]
        )

    def clean_data_basic_info(self):
        """
        This method is to print the cleaned dataset information
        """
        self.load_clean_dataset()

        total_of_users = len(self.transactions[Label.USER_ID].unique())

        count_user_trans = Counter(self.transactions[Label.USER_ID].tolist())
        mean_c = round(mean(list(count_user_trans.values())), 3)
        median_c = median(list(count_user_trans.values()))
        min_c = min(list(count_user_trans.values()))
        max_c = max(list(count_user_trans.values()))

        count_plays = self.transactions[Label.TRANSACTION_VALUE].tolist()
        mean_count_plays = round(mean(count_plays), 3)
        median_count_plays = round(median(count_plays), 3)

        total_of_items = len(self.items)
        total_of_transactions = len(self.transactions)
        total_of_classes = len(
            set(list(itertools.chain.from_iterable(list(map(Dataset.classes, self.items[Label.GENRES].tolist())))))
        )
        return pd.DataFrame(
            data=[[
                'Clean', total_of_users, total_of_items, total_of_transactions, total_of_classes,
                mean_c, median_c, mean_count_plays, median_count_plays, min_c, max_c
            ]],
            columns=[
                'Dataset', 'Users', 'Items', 'Transactions', 'Classes',
                "Users_trans_mean", "Users_trans_median", "mean_count_plays", "median_count_plays", "Minimum", "Maximum"
            ]
        )
