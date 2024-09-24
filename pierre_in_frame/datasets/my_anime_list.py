import os

import numpy as np
import pandas as pd

from datasets.utils.base_preprocess import Dataset
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class MyAnimeList(Dataset):
    """
    My Anime List dataset.
    This class organize the work with the dataset.
    """
    # Class information.
    dir_name = "mal"
    verbose_name = "My Anime List"
    system_name = "mal"

    # Raw paths.
    dataset_raw_path = "/".join([PathDirFile.RAW_DATASETS_DIR, dir_name])
    raw_transaction_file = "UserAnimeList.csv"
    raw_items_file = "AnimeList.csv"

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

    # ######################################### #
    # ############# Transactions ############## #
    # ######################################### #

    def load_raw_transactions(self):
        """
        Load raw transactions into the instance variable.
        """
        self.raw_transactions = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_transaction_file),
            usecols=['username', 'anime_id', 'my_score'], sep=','
        )
        self.raw_transactions.rename(
            columns={"username": Label.USER_ID, "anime_id": Label.ITEM_ID, "my_score": Label.TRANSACTION_VALUE}, inplace=True
        )

    def clean_transactions(self):
        """
        Cleaning the raw transactions and save as clean transactions.
        """
        super().clean_transactions()

        print("Get Raw Transactions")
        # Load the raw transactions.
        raw_transactions = self.get_raw_transactions()

        # Filter transactions based on the items id list.
        print("Filtering Items")
        filtered_raw_transactions = raw_transactions[
            raw_transactions[Label.ITEM_ID].isin(self.items[Label.ITEM_ID].tolist())]
        del raw_transactions

        # Cut users and set the new data into the instance.
        print("Cut Users")
        self.set_transactions(
            new_transactions=MyAnimeList.cut_users(filtered_raw_transactions, 8))
        del filtered_raw_transactions

        print("Transforming Scores")
        if Constants.NORMALIZED_SCORE:
            self.transactions[Label.TRANSACTION_VALUE] = np.where(self.transactions[Label.TRANSACTION_VALUE] >= 8, 1, 0)

        # Save the clean transactions as CSV.
        self.transactions.to_csv(
            os.path.join(self.clean_dataset_dir, PathDirFile.TRANSACTIONS_FILE),
            index=False
        )

    # ######################################### #
    # ################# Items ################# #
    # ######################################### #

    def load_raw_items(self):
        """
        Load Raw Items into the instance variable.
        """
        self.raw_items = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_items_file),
            sep=',', usecols=['anime_id', 'title', 'genre']
        )
        self.raw_items.rename(
            columns={"anime_id": Label.ITEM_ID, "title": Label.TITLE, "genre": Label.GENRES}, inplace=True
        )

    def clean_items(self):
        """
        Cleaning the raw items and save as clean items.
        """
        # Load the raw items.
        print("Loading Raw Items")
        raw_items_df = self.get_raw_items()

        # Clean the items without information and with the label indicating no genre in the item.
        print("Dropping No Genres")
        raw_items_df = raw_items_df[raw_items_df[Label.GENRES].str.strip().astype(bool)]

        # raw_items_df[Label.GENRES] = raw_items_df[Label.GENRES].astype(str)
        #
        # raw_items_df[Label.GENRES].replace('', np.nan, inplace=True)
        raw_items_df.dropna(subset=[Label.GENRES], inplace=True)

        genre_clean_items = raw_items_df[~raw_items_df[Label.GENRES].isin([
            "s",
            "s & Go",
            "s & Go MAX",
            "s & Go Special",
            "s & Go WGP",
            "s & Go!! WGP Bousou Mini Yonku Daitsuiseki",
            "s & Magic",
            "s →NEXT LoveLive! 2014 - Endless Parade Encore Animation",
            "s →NEXT LoveLive! 2014 - Endless Parade Makuai Drama",
            "s 3 Animation Films",
            "s a Rumic World: 50th Anniversary Weekly★Shonen Sunday",
            "s a stage",
            "s Adventure",
            "s Autumn!",
            "s Blade OVA",
            "s Blade OVA Specials",
            "s Blade Rebellion vs. Hagure Yuusha no Aesthetica",
            "s Blade: Grimoire",
            "s Blade: Gyokuza wo Tsugu Mono",
            "s Blade: Gyokuza wo Tsugu Mono Specials",
            "s Blade: Rebellion",
            "s Blade: Rebellion Specials",
            "s Blade: Rurou no Senshi",
            "s Blade: Rurou no Senshi Specials",
            "s Blade: Unlimited",
            "s Blade: Utsukushiki Toushitachi",
            "s Blood Trail",
            "s Bowy",
            "s Case Files",
            "s Castle",
            "s Christmas",
            "s Circus",
            "s Collection of Short Stories",
            "s Comic Video",
            "s Company",
            "s Counterattack",
            "s Creed",
            "s Critical Event",
            "s Cut-ban",
            "s Daily Life",
            "s Dragon",
            "s Dream",
            "s Dream 2",
            "s Edition: Issenman no Kufura no Kiroku",
            "s End Umbrella",
            "s English Show",
            "s Eve 2014 Super Mega Special",
            "s Experience the Jomon Period!",
            "s Eye",
            "s Eye Love",
            "s Feather",
            "s Feel - I. Presage Flower",
            "s Feel - II. Lost Butterfly",
            "s Feel 3",
            "s Final Love Live! Opening Animation",
            "s Go☆Prix Paris",
            "s Gonna Change",
            "s Greetings 2017 from Dwarf",
            "s Heaven",
            "s Heroine Series: Kanojo wa Hanayome Kouhosei? Cinderella Collection",
            "s High",
            "s High Specials",
            "s Hotel",
            "s Ichisuisei Hiyoko to Fushigi na Ofuda",
            "s La Cooking? Showtime!",
            "s Letter",
            "s LOVE Nitro+CHiRAL",
            "s Memory",
            "s Memory: Shiawase Monogatari",
            "s Michael?",
            "s Michael? (TV)",
            "s Michael? 2",
            "s Mini Picture Drama",
            "s Most Audacious Animators",
            "s My Life",
            "s New Year!",
            "s Nupu Nupu",
            "s Oasis",
            "s Otherwise",
            "s Paradise",
            "s Picture Drama",
            "s Rain",
            "s Rain OVA",
            "s Report",
            "s Rondo",
            "s Sand Witch",
            "s Sand Witch Bangai-hen: Tanu no Kyuujitsu",
            "s Sand Witch Episode 0",
            "s Scar",
            "s Seven Mysteries",
            "s Song",
            "s Spring!",
            "s Summer!",
            "s Surprised Christmas",
            "s Sweet Home",
            "s Sweet Home OVA",
            "s Sweet Home: Atarashii Ouchi",
            "s The Dunwich Horror and Other Stories",
            "s the Night Before a Wedding",
            "s Toy Adventure",
            "s Trip The Animation",
            "s Try a Curious Experiment!",
            "s Unlucky Days",
            "s Walker",
            "s Wind-hen",
            "s Winter!",
            "s World",
            "s Zoo: Daisuki! Witzy",
            "s Zoo: Daisuki! Witzy - Happy Birthday",
            "s: Shinkasuru Kettou! Stardust vs. Red Demon&#039",
            "s/Start/Load/End",
            "s↑↑",
            "s★Angel",
            "s★Angel Gaiden: Susume! Super★Angels!",
            "s★Angel: Forever and Ever",
            "t Cry", "t Forget Me...", "t Pass Me By", "t Slow Me Down",
            "t Take My Eyes Off You",
            "T X",
            "T X Neo",
            "t You Wish You Were Here?",
            "z",
            "Univers",
            "Univers Specials",
            "Zoo: Daisuki! Witzy",
            "s",
            "Zoo: Daisuki! Witzy - Happy",
            "Birthday",
            "s: Shinkasuru",
            "Kettou! Stardust",
            "vs.Red",
            "Demon &  # 039",
        ])
        ]
        del raw_items_df
        genre_clean_items[Label.GENRES] = genre_clean_items[Label.GENRES].str.replace(", ", "|")
        genre_clean_items[Label.GENRES] = genre_clean_items[Label.GENRES].str.replace(",", "|")

        # Set the new data into the instance.
        print("Drop Duplicates")
        self.set_items(new_items=genre_clean_items)
        del genre_clean_items
        self.items.drop_duplicates(subset=[Label.ITEM_ID], inplace=True)

        # Save the clean transactions as CSV.
        self.items.to_csv(os.path.join(self.clean_dataset_dir, PathDirFile.ITEMS_FILE), index=False)
