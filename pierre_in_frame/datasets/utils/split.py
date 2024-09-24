from collections import Counter

import pandas as pd
from joblib import Parallel, delayed
from numpy import array_split
from sklearn.model_selection import KFold

from settings.constants import Constants
from settings.labels import Label

# Shared memory between the processes
global_train = []
global_test = []


def user_split_in_kfolds(
        user_transactions: pd.DataFrame, trial: int = Constants.N_TRIAL_VALUE, n_folds: int = Constants.K_FOLDS_VALUE
) -> None:
    """
    Split the user transaction in K-folds.
    :param user_transactions: A Pandas DataFrame with user transactions.
    :param trial: An int that represents a number of the experimental trial.
    :param n_folds: An int representing a number of the k folds.
    """
    user_transactions.reset_index(inplace=True)
    kf = KFold(n_splits=n_folds, random_state=42 * trial * n_folds, shuffle=True)

    i = 0
    if len(user_transactions) < 10:
        print(user_transactions)
    for train, test in kf.split(user_transactions):
        global_train[i].append(user_transactions.iloc[train])
        global_test[i].append(user_transactions.iloc[test])
        i += 1


def concat_folds(train_results_df, test_results_df):
    """
    To concat the users' folds on the same structure.
    """
    return pd.concat(train_results_df, sort=False), pd.concat(test_results_df, sort=False)


def compute_kfold(
        transactions_df: pd.DataFrame, trial: int = Constants.N_TRIAL_VALUE,
        n_folds: int = Constants.K_FOLDS_VALUE, n_jobs: int = Constants.N_CORES
) -> list:
    """
    Prepare the users to be processed in parallel with the joblib.
    :param transactions_df: A Pandas DataFrame with user transactions.
    :param trial: An int that represents a number of the experimental trial.
    :param n_folds: An int representing a number of the k folds.
    :param n_jobs: An int representing the number of cores used to parallel the operation.
    :return: A list composed of the fold in positions, each fold position has [0] as the k fold train transactions and [1] as the k fold test transactions.
    """
    # Preparing: users, results dataframe and shared queue over processes

    grouped_transactions = transactions_df.groupby(by=[Label.USER_ID])

    delayed_list = (
        delayed(user_split_in_kfolds)(transactions, trial, n_folds)
        for user_id, transactions in grouped_transactions
    )

    Parallel(n_jobs=n_jobs, verbose=10, batch_size=128, require='sharedmem')(delayed_list)

    resp = []
    for train_df, test_df in zip(global_train, global_test):
        resp.append(concat_folds(train_df, test_df))
    return resp


def split_with_joblib(
        transactions_df: pd.DataFrame, trial: int = Constants.N_TRIAL_VALUE, n_folds: int = Constants.K_FOLDS_VALUE
) -> list:
    """
    Prepare the users to be processed in parallel with the joblib.
    :param transactions_df: A Pandas DataFrame with user transactions.
    :param trial: An int that represents a number of the experimental trial.
    :param n_folds: An int representing a number of the k folds.
    :return: A list composed of the fold in positions, each fold position has [0] as the k fold train transactions and [1] as the k fold test transactions.
    """
    global global_train
    global global_test

    global_train = [[] for _ in range(n_folds)]
    global_test = [[] for _ in range(n_folds)]

    resp = compute_kfold(transactions_df=transactions_df, trial=trial, n_folds=n_folds)

    return resp


# ############################################################################################### #

class SequentialTimeSplit:

    def __init__(self, transactions_df: pd.DataFrame, n_folds: int):
        self.transactions_df = transactions_df
        self.train_list = []
        self.test_list = []
        self.n_folds = n_folds

    def user_split_in_sequential(
            self,
            user_transactions: pd.DataFrame
    ) -> None:
        """
        Split the user transaction in sequential way.

        :param user_transactions: A Pandas DataFrame with user transactions.
        :param n_folds: An int representing a number of the k folds.
        """
        user_transactions.reset_index(inplace=True)
        user_transactions.sort_values(by=[Label.TIME], inplace=True)
        split_list = array_split(user_transactions, self.n_folds)

        for ix, fold in enumerate(split_list):
            if ix >= self.n_folds - 1:
                self.test_list.append(pd.DataFrame(fold))
            else:
                self.train_list.append(pd.DataFrame(fold))

    def compute_kfold(self) -> tuple:
        """
        Prepare the users to be processed in parallel with the joblib.

        :return: A list composed of the fold in positions, each fold position has [0] as the k fold train transactions and [1] as the k fold test transactions.
        """
        # Preparing: users, results dataframe and shared queue over processes

        grouped_transactions = self.transactions_df.groupby(by=[Label.USER_ID])

        delayed_list = (
            delayed(self.user_split_in_sequential)(
                user_transactions=transactions
            )
            for user_id, transactions in grouped_transactions
        )

        Parallel(n_jobs=-1, verbose=10, batch_size=128, require='sharedmem')(delayed_list)

        return pd.concat(self.train_list, sort=False), pd.concat(self.test_list, sort=False)

    def main(self) -> tuple:
        """
        Prepare the users to be processed in parallel with the joblib.
        
        :return: A list composed of the fold in positions, each fold position has [0] as the k fold train transactions and [1] as the k fold test transactions.
        """

        train_df, test_df = self.compute_kfold()

        return train_df, test_df


class CrossValidationThroughTime:

    def __init__(self, transactions_df: pd.DataFrame, n_folds: int):
        self.transactions_df = transactions_df

        self.n_folds = n_folds

        self.train_list = [[] for _ in range(self.n_folds)]
        self.valid_list = [[] for _ in range(self.n_folds)]
        self.test_list = [[] for _ in range(self.n_folds)]

    def user_splitting(self, user_transactions: pd.DataFrame) -> None:
        """
        Split the user transaction in sequential way.

        :param user_transactions: A Pandas DataFrame with user transactions.

        """
        user_transactions.reset_index(inplace=True)
        user_transactions.sort_values(by=[Label.TIME], inplace=True)
        split_list = list(array_split(user_transactions, (self.n_folds * 2) - 1))

        for ix in range(self.n_folds):
            self.train_list[ix].append(
                pd.concat(
                    [pd.DataFrame(split_list[i]) for i in range(0, ix + 1)],
                )
            )
            self.valid_list[ix].append(pd.DataFrame(split_list[ix + 1]))
            self.test_list[ix].append(pd.DataFrame(split_list[ix + 2]))

    def compute_kfold(self) -> tuple:
        """
        Prepare the users to be processed in parallel with the joblib.

        :return: A list composed of the fold in positions, each fold position has [0] as the k fold train transactions and [1] as the k fold test transactions.
        """
        # Preparing: users, results dataframe and shared queue over processes

        grouped_transactions = self.transactions_df.groupby(by=[Label.USER_ID])

        delayed_list = (
            delayed(self.user_splitting)(
                user_transactions=transactions
            )
            for user_id, transactions in grouped_transactions
        )

        Parallel(n_jobs=-1, verbose=10, batch_size=128, require='sharedmem')(delayed_list)

        return self.train_list, self.valid_list, self.test_list

    def main(self) -> tuple:
        """
        Prepare the users to be processed in parallel with the joblib.

        :return: A list composed of the fold in positions, each fold position has [0] as the k fold train transactions and [1] as the k fold test transactions.
        """

        train_list, valid_list, test_list = self.compute_kfold()

        return train_list, valid_list, test_list
