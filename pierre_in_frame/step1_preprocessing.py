import multiprocessing

import itertools
import logging
import pandas as pd
from joblib import Parallel, delayed

from datasets.registred_datasets import RegisteredDataset
from graphics.dataset_chart import DatasetChart
from scikit_pierre.distributions.compute_distribution import computer_users_distribution_pandas
from scikit_pierre.models.item import ItemsInMemory
from settings.labels import Label
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad
from utils.clocker import Clocker
from utils.input import Input
from utils.logging_settings import setup_logging
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep1(Step):
    """
    This class is administrating the Step 1 of the framework (Pre-Processing).
    """

    def read_the_entries(self):
        """
        This method reads the terminal entries.
        """
        self.experimental_settings = Input.step1()

    def set_the_logfile(self):
        """
        This method is to config the log file.
        """
        # Setup Log configuration
        setup_logging(
            save_path=PathDirFile.log_preprocessing_path(
                dataset=self.experimental_settings['dataset']
            )
        )

    def print_basic_info(self):
        """
        This method is to print basic information about the step and machine.
        """
        # Logging machine data
        logger.info("$" * 50)

        self.machine_information()

        # Logging the experiment setup
        logger.info(f"> DATASET (PREPROCESSING STEP) - {self.experimental_settings['opt']}")
        logger.info(" ".join(['>>', 'Option:', self.experimental_settings['opt']]))
        logger.info(" ".join(['>>', 'Dataset:', str(self.experimental_settings['dataset'])]))
        if self.experimental_settings['opt'] == Label.DATASET_SPLIT:
            logger.info(" ".join(['>>', 'Number of Folds:', str(self.experimental_settings['n_folds'])]))
            logger.info(" ".join(['>>', 'Number of Trials:', str(self.experimental_settings['n_trials'])]))
        elif self.experimental_settings['opt'] == Label.DATASET_DISTRIBUTION:
            logger.info(" ".join(['>>', 'Fold to use:', str(self.experimental_settings['fold'])]))
            logger.info(" ".join(['>>', 'Trial to use:', str(self.experimental_settings['trial'])]))

        logger.info("$" * 50)

    def create_folds(self):
        """
        This method is to split the data and create the folds.
        """

        # Executing the pre-processing
        RegisteredDataset.preprocessing(
            experiment_name=self.experimental_settings['experiment_name'],
            dataset=self.experimental_settings['dataset'],
            n_trials=self.experimental_settings['n_trials'],
            n_folds=self.experimental_settings['n_folds'],
            cut_value=self.experimental_settings['cut_value'],
            item_cut_value=self.experimental_settings['item_cut_value'],
            profile_len_cut_value=self.experimental_settings['profile_len_cut_value'],
            test_len_cut_value=self.experimental_settings['test_len_cut_value'],
            split_methodology=self.experimental_settings["split_methodology"]
        )

        # Finishing the step
        # logger.info(" ".join(['->>', 'Time Execution:', str(self.get_total_time())]))

    def create_charts(self):
        """
        This method has the function of generating graphics with the dataset analyzes.
        """
        # Loading dataset information
        dt_chat = DatasetChart(
            dataset_name=self.experimental_settings['dataset'],
            split_methodology=self.experimental_settings["split_methodology"],
            experiment_name=self.experimental_settings['experiment_name']
        )
        # dt_chat.item_long_tail()
        # dt_chat.genres()
        dt_chat.items_genres_raw_and_clean()

    def create_analyzes(self):
        """
        This method is to lead with the dataset numbers.
        """

        # Load the dataset
        dataset_instance = RegisteredDataset.load_dataset(
            dataset=self.experimental_settings['dataset']
        )
        dataset_instance.set_environment(
            experiment_name=self.experimental_settings['experiment_name'],
            split_methodology=self.experimental_settings["split_methodology"]
        )
        # Print the Raw dataset information
        raw_dataset_info_df = dataset_instance.raw_data_basic_info()

        # Print the Clean dataset information
        clean_dataset_info_df = dataset_instance.clean_data_basic_info()

        dataset_info_df = pd.concat([raw_dataset_info_df, clean_dataset_info_df])
        print(dataset_info_df)

        # Save the distributions
        SaveAndLoad.save_dataset_analyze(
            data=dataset_info_df, dataset=self.experimental_settings['dataset'],
            experiment_name=self.experimental_settings['experiment_name'],
            split_methodology=self.experimental_settings["split_methodology"]
        )

    def create_folds_analyze(self) -> None:

        """
        This method is to lead with the dataset numbers.
        """

        dataset_info_df = []
        # Load the dataset
        dataset_instance = RegisteredDataset.load_dataset(
            dataset=self.experimental_settings['dataset']
        )
        dataset_instance.set_environment(
            experiment_name=self.experimental_settings['experiment_name'],
            split_methodology=self.experimental_settings["split_methodology"]
        )
        dataset_instance.load_clean_items()
        for t in self.experimental_settings['trial']:
            for f in self.experimental_settings['fold']:
                dataset_instance.set_environment(
                    experiment_name=self.experimental_settings['experiment_name'],
                    n_trials=t, n_folds=f, split_methodology=self.experimental_settings["split_methodology"]
                )
                dataset_instance.get_full_train_transactions(trial=t, fold=f)
                # Print the Clean dataset information
                dataset_info_df.append(dataset_instance.fold_basic_info())

        dataset_info_df = pd.concat(dataset_info_df)
        print(dataset_info_df)

        # Save the distributions
        SaveAndLoad.save_fold_analyze(
            data=dataset_info_df, dataset=self.experimental_settings['dataset'],
            experiment_name=self.experimental_settings['experiment_name'],
            split_methodology=self.experimental_settings["split_methodology"]
        )

    def compute_class_one_hot_encode(
            self, dataset: str, experiment_name: str, split_methodology: str):
        dataset_instance = RegisteredDataset.load_dataset(dataset)
        dataset_instance.set_environment(
            experiment_name=self.experimental_settings['experiment_name'],
            split_methodology=self.experimental_settings["split_methodology"]
        )
        _items = ItemsInMemory(data=dataset_instance.get_items())
        _items.one_hot_encode()
        encoded = _items.get_encoded()
        SaveAndLoad.save_item_class_one_hot_encode(
            data=encoded, dataset=dataset,
            experiment_name=experiment_name,
            split_methodology=split_methodology
        )

    def create_class_one_hot_encode(self):
        self.compute_class_one_hot_encode(
            dataset=self.experimental_settings['dataset'],
            experiment_name=self.experimental_settings['experiment_name'],
            split_methodology=self.experimental_settings["split_methodology"]
        )

    def create_distribution(self):
        """
        This method is to lead with the distribution file.
        """

        # It creates the combination of process entries
        combination = [
            [self.experimental_settings['experiment_name']], [self.experimental_settings["split_methodology"]],
            [self.experimental_settings['dataset']], self.experimental_settings['trial'],
            self.experimental_settings['fold'], self.experimental_settings['distribution'],
            self.experimental_settings['distribution_class']
        ]

        if self.experimental_settings['multiprocessing'] == "joblib":
            # Start the processes in parallel using joblib
            Parallel(
                n_jobs=self.experimental_settings['n_jobs'], verbose=10
            )(
                delayed(self.compute_distribution)(
                    experiment_name=experiment_name, split_methodology=split_methodology, dataset=dataset,
                    trial=trial, fold=fold,
                    distribution=distribution, distribution_class=distribution_class
                ) for experiment_name, split_methodology, dataset, trial, fold, distribution, distribution_class
                in list(itertools.product(*combination))
            )
        elif self.experimental_settings['multiprocessing'] == "starmap":
            process_args = []
            for experiment_name, split_methodology,dataset, trial, fold, distribution, distribution_class in list(itertools.product(*combination)):
                process_args.append((experiment_name, split_methodology,dataset, trial, fold, distribution, distribution_class))
            pool = multiprocessing.Pool(processes=self.experimental_settings["n_jobs"])
            pool.starmap(self.compute_distribution, process_args)
            pool.close()
            pool.join()

    @staticmethod
    def compute_distribution(
            experiment_name: str, split_methodology: str, dataset: str, trial: int, fold: int,
            distribution: str, distribution_class: str
    ) -> None:
        """
        This method is to compute the preference distribution.
        """
        clocker = Clocker()
        clocker.start_count()

        # Load the dataset
        dataset_instance = RegisteredDataset.load_dataset(dataset)
        dataset_instance.set_environment(
            experiment_name=experiment_name, n_trials=trial, n_folds=fold, split_methodology=split_methodology
        )

        # Get the users' preferences set
        users_preference_set = dataset_instance.get_full_train_transactions(trial=trial, fold=fold)

        if distribution_class == Label.CLASS_GENRE:
            data = computer_users_distribution_pandas(
                users_preference_set=users_preference_set, items_df=dataset_instance.get_items(),
                distribution=distribution
            )
        else:
            items_inst = ItemsInMemory(data=dataset_instance.get_items())
            items_inst.classifying_item_by_popularity(users_transactions=users_preference_set)

            items_dataframe = items_inst.transform_to_pandas_items()
            data = computer_users_distribution_pandas(
                users_preference_set=users_preference_set, items_df=items_dataframe,
                distribution=distribution
            )

        # Save the distributions
        SaveAndLoad.save_user_preference_distribution(
            experiment_name=experiment_name, split_methodology=split_methodology,
            data=data, dataset=dataset, fold=fold, trial=trial, distribution=distribution,
            distribution_class=distribution_class
        )

        clocker.finish_count()
        SaveAndLoad.save_distribution_time(
            experiment_name=experiment_name, split_methodology=split_methodology,
            data=clocker.clock_data(), dataset=dataset, fold=fold, trial=trial,
            distribution=distribution, distribution_class=distribution_class
        )

    def create_class_distribution(self):
        """
        This method is to lead with the distribution file.
        """
        for distribution_class in self.experimental_settings['distribution_class']:
            self.compute_class_distribution(
                experiment_name=self.experimental_settings['experiment_name'],
                split_methodology=self.experimental_settings["split_methodology"],
                dataset=self.experimental_settings['dataset'],
                distribution_class=distribution_class
            )

    @staticmethod
    def compute_class_distribution(
            experiment_name: str, split_methodology: str, dataset: str, distribution_class: str
    ) -> None:
        """
        This method is to compute the preference distribution.
        """        # Load the dataset
        dataset_instance = RegisteredDataset.load_dataset(dataset)
        dataset_instance.set_environment(
            experiment_name=experiment_name, split_methodology=split_methodology
        )

        # Get the users' preferences set
        users_preference_set = dataset_instance.get_transactions()
        original_item = dataset_instance.get_items()
        items_inst = ItemsInMemory(data=original_item)
        items_inst.classifying_item_by_popularity(users_transactions=users_preference_set)

        items_dataframe = items_inst.transform_to_pandas_items()
        items_dataframe.rename(columns={Label.CLASS_GENRE: Label.CLASS_POPULARITY}, inplace=True)
        items_dataframe.sort_values(by=Label.ITEM_ID, ascending=False, inplace=True)
        print(items_dataframe)
        original_item.sort_values(by=Label.ITEM_ID, ascending=False, inplace=True)

        data = original_item.merge(items_dataframe, how='left', on=Label.ITEM_ID)
        print(data)

        # Save the distributions
        SaveAndLoad.save_clean_items(
            experiment_name=experiment_name, split_methodology=split_methodology,
            data=data, dataset=dataset
        )

    def main(self):
        """
        Main method used to choice the run option.
        """
        if self.experimental_settings['opt'] == Label.DATASET_CHART:
            self.create_charts()
        elif self.experimental_settings['opt'] == Label.DATASET_ANALYZE:
            self.create_analyzes()
            self.create_folds_analyze()
        elif self.experimental_settings['opt'] == Label.DATASET_DISTRIBUTION:
            self.create_distribution()
        elif self.experimental_settings['opt'] == Label.CLASS_DISTRIBUTION:
            self.create_class_distribution()
        elif self.experimental_settings['opt'] == Label.DATASET_CLASS_ONE_HOT_ENCODE:
            self.create_class_one_hot_encode()
        elif self.experimental_settings['opt'] == Label.DATASET_SPLIT:
            self.create_folds()
        else:
            print("Option not found!")


if __name__ == '__main__':
    """
    It starts the pre-processing step
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep1()
    step.read_the_entries()
    # step.set_the_logfile()
    # step.print_basic_info()
    step.main()
    logger.info(" ".join(['+' * 10, 'System Shutdown', '+' * 10]))
