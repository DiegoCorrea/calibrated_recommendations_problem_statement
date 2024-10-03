from collections import defaultdict

import pandas as pd
from surprise import Reader, Dataset

from settings.constants import Constants
from settings.labels import Label


class PandasSurprise:

    @staticmethod
    def pandas_transform_trainset_to_surprise(trainset_df: pd.DataFrame) -> Dataset:
        value_min = trainset_df[Label.TRANSACTION_VALUE].min()
        value_max = trainset_df[Label.TRANSACTION_VALUE].max()
        reader_train = Reader(rating_scale=(value_min, value_max))
        data_train = Dataset.load_from_df(trainset_df[[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE]],
                                          reader_train)
        return data_train.build_full_trainset()

    @staticmethod
    def pandas_transform_testset_to_surprise(testset_df: pd.DataFrame) -> Dataset:
        value_min = testset_df[Label.TRANSACTION_VALUE].min()
        value_max = testset_df[Label.TRANSACTION_VALUE].max()
        reader_test = Reader(rating_scale=(value_min, value_max))
        data_test = Dataset.load_from_df(testset_df[[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE]],
                                         reader_test)
        testset = data_test.build_full_trainset()
        return testset.build_testset()

    @staticmethod
    def pandas_transform_all_dataset_to_surprise(dataset_df: pd.DataFrame) -> Dataset:
        value_min = dataset_df[Label.TRANSACTION_VALUE].min()
        value_max = dataset_df[Label.TRANSACTION_VALUE].max()
        reader_train = Reader(rating_scale=(value_min, value_max))
        return Dataset.load_from_df(dataset_df[[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE]], reader_train)

    @staticmethod
    def __paralleling_convert(user: tuple, n: int) -> pd.DataFrame:
        uid, user_ratings = user
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        rec_list = [
            pd.DataFrame(data=[[uid, iid, est]], columns=[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE])
            for iid, est, true_r in user_ratings[:n]
        ]
        # for iid, est, true_r in user_ratings[:n]:
            # top_n_df = pd.concat([top_n_df,
            #                       pd.DataFrame(data=[[uid, iid, est]],
            #                                    columns=[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE])
            #                       ])
            # rec_list.append(
            #     pd.DataFrame(data=[[uid, iid, est]], columns=[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE])
            # )
        return pd.concat(rec_list)

    @staticmethod
    def surprise_to_pandas_get_candidates_items(predictions: Dataset,
                                                n: int = Constants.CANDIDATES_LIST_SIZE) -> pd.DataFrame:
        """Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A pandas dataframe with the top n items.
        """
        # First, map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est, true_r))
        map_results_df = [PandasSurprise.__paralleling_convert(user, n) for user in top_n.items()]
        return pd.concat(map_results_df, sort=False)

    @staticmethod
    def surprise_to_pandas_get_candidates_items_optimized(
            predictions: Dataset, n: int = Constants.CANDIDATES_LIST_SIZE) -> pd.DataFrame:
        """Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A pandas dataframe with the top n items.
        """
        # First, map the predictions to each user.
        rec_list = [
            pd.DataFrame(data=[[uid, iid, est]], columns=[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE])
            for uid, iid, true_r, est, _ in predictions
        ]
        return pd.concat(rec_list, sort=False).sort_values(by=Label.TRANSACTION_VALUE, ascending=False).iloc[:n]

    @staticmethod
    def surprise_to_pandas_batch_get_candidates_items_optimized(
            predictions: Dataset, n: int = Constants.CANDIDATES_LIST_SIZE) -> pd.DataFrame:
        """Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A pandas dataframe with the top n items.
        """
        # First, map the predictions to each user.
        # print(predictions[0])
        rec_list = [
            pd.DataFrame(data=[[uid, iid, est]], columns=[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE])
            for uid, iid, true_r, est, _ in predictions
        ]
        concatened = [
            df.sort_values(by=Label.TRANSACTION_VALUE, ascending=False).iloc[:n]
            for user_id, df in pd.concat(rec_list, sort=False).groupby(by=[Label.USER_ID])
        ]
        return pd.concat(concatened)
