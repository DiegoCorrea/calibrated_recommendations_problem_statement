import itertools
from collections import Counter
from multiprocessing import Pool

import numpy as np
from pandas import DataFrame, concat

from settings.constants import Constants
from settings.labels import Label


def total_of_genres(user, user_df: DataFrame) -> DataFrame:
    genres_list = []
    for row in user_df.itertuples():
        item_genre = getattr(row, Label.GENRES)
        splitted = item_genre.split('|')
        genres_list = genres_list + [genre for genre in splitted]
    return DataFrame([[user, len(user_df), len(np.unique(genres_list))]],
                     columns=[Label.USER_ID, Label.USER_MODEL_SIZE_LABEL, Label.NUMBER_OF_CLASSES])


def user_genres_analysis(transactions_df: DataFrame) -> DataFrame:
    users_list = transactions_df[Label.USER_ID].unique().tolist()
    user_df = [total_of_genres(user, transactions_df[transactions_df[Label.USER_ID] == user]) for user in users_list]
    analysis_of_users_df = concat(user_df, sort=False)
    analysis_of_users_df.set_index(Label.USER_ID, inplace=True, drop=True)
    return analysis_of_users_df


# #######################################################################################################
#
# #######################################################################################################
def split_genres(user_transactions_df):
    transactions_genres_list = user_transactions_df[Label.GENRES].tolist()
    genres_list = []
    for item_genre in transactions_genres_list:
        splitted = item_genre.split('|')
        splitted_genre_list = [genre for genre in splitted]
        genres_list = genres_list + splitted_genre_list
    count_dict = Counter(genres_list)
    values_list = list(count_dict.values())
    # sum_values_list = sum(values_list)
    # values_list = [v / sum_values_list for v in values_list]
    df = DataFrame([values_list], columns=list(count_dict.keys()))
    return df


def genre_probability_distribution(transactions_df, label=Label.USER_ID):
    id_list = transactions_df[label].unique().tolist()
    pool = Pool(Constants.N_CORES)
    list_df = pool.map(split_genres, [transactions_df[transactions_df[label] == uid] for uid in id_list])
    pool.close()
    pool.join()
    result_df = concat(list_df, sort=False)
    result_df.fillna(0.0, inplace=True)
    return result_df


def genre_probability_distribution_mono(
        transactions_df: DataFrame, items_df: DataFrame, label : str = Label.USER_ID
) -> DataFrame:
    def split_genres_subinside(user_transactions_df: DataFrame) -> dict:
        transactions_genres_list = items_df[
            items_df[Label.ITEM_ID].isin(user_transactions_df[Label.ITEM_ID].tolist())
        ][Label.GENRES].tolist()
        genres_list = []
        for item_genre in transactions_genres_list:
            splitted = item_genre.split('|')
            splitted_genre_list = [genre for genre in splitted]
            genres_list = genres_list + splitted_genre_list

        results_dict = dict(Counter(genres_list))

        progress.update(1)
        progress.set_description("Genre Frequency Computation: ")
        return results_dict

    print("Processing Genres")
    genre_list = items_df[Label.GENRES].tolist()

    total_of_classes = list(set(list(itertools.chain.from_iterable(
        list(map(Dataset.classes, genre_list))
    ))))
    grouped_transactions = transactions_df.groupby(by=[label])

    progress = tqdm(total=len(grouped_transactions))

    list_df = [
        split_genres_subinside(df) for uid, df in grouped_transactions
    ]
    progress.close()
    print("Concat Results")

    results = DataFrame.from_dict(list_df)
    results.fillna(0.0, inplace=True)

    return results
