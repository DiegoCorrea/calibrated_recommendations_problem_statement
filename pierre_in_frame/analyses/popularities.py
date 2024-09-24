import pandas as pd

from settings.labels import Label


def count_item_popularity(transactions: pd.DataFrame) -> pd.DataFrame:
    item_popularity_count = transactions[Label.ITEM_ID].value_counts()
    popularity_dict = item_popularity_count.to_dict()
    analysis_of_items_df = pd.DataFrame(columns=[Label.ITEM_ID, Label.TOTAL_TIMES])
    for item in popularity_dict:
        analysis_of_items_df = pd.concat([analysis_of_items_df, pd.DataFrame(data=[[item, popularity_dict[item]]],
                                                                             columns=[Label.ITEM_ID, Label.TOTAL_TIMES])])
    return analysis_of_items_df

#
# def compute_popularity(transactions_df):
#     # Item Popularity
#     analysis_of_items_df = count_popularity_item(transactions_df)
#
#     item_short_tail_id_list = alternative_get_short_tail_items(analysis_of_items_df)
#
#     analysis_of_items_df[TYPE_OF_POPULARITY] = MEDIUM_TAIL_TYPE
#
#     analysis_of_items_df.loc[
#         analysis_of_items_df[ITEM_LABEL].isin(item_short_tail_id_list), TYPE_OF_POPULARITY] = SHORT_TAIL_TYPE
#     item_medium_tail_id_list = (analysis_of_items_df[analysis_of_items_df[TYPE_OF_POPULARITY] == MEDIUM_TAIL_TYPE])[
#         ITEM_LABEL].unique().tolist()
#
#     # User and Item
#     users_id_list = transactions_df[USER_LABEL].unique().tolist()
#     query = [(user_id,
#               transactions_df[transactions_df[USER_LABEL] == user_id],
#               item_short_tail_id_list,
#               item_medium_tail_id_list) for user_id in users_id_list]
#
#     pool = Pool(N_CORES)
#     list_df = pool.starmap(map_get_users_popularity, query)
#     pool.close()
#     pool.join()
#     analysis_of_users_df = pd.concat(list_df, sort=False)
#
#     niche_id_list = get_niche_users(analysis_of_users_df)
#     focused_id_list = get_focused_users(analysis_of_users_df)
#
#     analysis_of_users_df[TYPE_OF_POPULARITY] = DIVERSE_TYPE
#
#     analysis_of_users_df.loc[
#         analysis_of_users_df[USER_LABEL].isin(niche_id_list), TYPE_OF_POPULARITY] = NICHE_TYPE
#     analysis_of_users_df.loc[
#         analysis_of_users_df[USER_LABEL].isin(focused_id_list), TYPE_OF_POPULARITY] = FOCUSED_TYPE
#
#     return analysis_of_users_df, analysis_of_items_df
