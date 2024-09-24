from copy import deepcopy

import pandas as pd
import sklearn
from joblib import Parallel, delayed
from numpy import array_split

from settings.labels import Label


class KFoldTrainValidationTest:

    def __init__(self, transactions_df: pd.DataFrame, n_folds: int, n_trial: int):
        self.transactions_df = transactions_df

        self.n_folds = n_folds
        self.n_trial = n_trial

        self.train_list = [[] for _ in range(self.n_folds)]
        self.valid_list = [[] for _ in range(self.n_folds)]
        self.test_list = [[] for _ in range(self.n_folds)]

    def user_splitting(self, user_transactions: pd.DataFrame, n_folds: int, n_trial) -> None:
        """
        Split the user transaction in sequential way.

        :param user_transactions: A Pandas DataFrame with user transactions.

        """
        user_transactions.reset_index(inplace=True)
        shuffle_user_transactions = sklearn.utils.shuffle(user_transactions, random_state=42 * n_trial)
        split_list = list(array_split(shuffle_user_transactions, n_folds))
        n_slice_list = list(range(0, n_folds))

        for ix in n_slice_list:
            slice_list = deepcopy(n_slice_list)
            slice_list.remove(ix)
            validation_number = ix + 1 if ix != n_folds - 1 else 0

            slice_list.remove(validation_number)
            self.train_list[ix].append(
                pd.concat(
                    [pd.DataFrame(split_list[i]) for i in slice_list],
                )
            )
            self.valid_list[ix].append(pd.DataFrame(split_list[validation_number]))
            self.test_list[ix].append(pd.DataFrame(split_list[ix]))

    def compute_kfold(self) -> tuple:
        """
        Prepare the users to be processed in parallel with the joblib.

        :return: A list composed of the fold in positions, each fold position has [0] as the k fold train transactions and [1] as the k fold test transactions.
        """
        # Preparing: users, results dataframe and shared queue over processes

        grouped_transactions = self.transactions_df.groupby(by=[Label.USER_ID])

        delayed_list = (
            delayed(self.user_splitting)(
                user_transactions=transactions,
                n_trial=self.n_trial,
                n_folds=self.n_folds
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
