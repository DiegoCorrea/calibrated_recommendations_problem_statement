import itertools
import logging

from searches.conformity_search import ManualConformityAlgorithmSearch
from searches.implicit_search import ImplicitGridSearch
from searches.pierre_search import PierreGridSearch
from searches.recommender_search import RecommenderSearch, SurpriseRandomSearch
from settings.labels import Label
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad
from utils.clocker import Clocker
from utils.input import Input
from utils.logging_settings import setup_logging
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep2(Step):
    """
    This class is administrating the Step 2 of the framework (Hyperparameters search)
    """

    def read_the_entries(self) -> None:
        """
        This method reads the terminal entries.
        """
        self.experimental_settings = Input.step2()

    @staticmethod
    def set_the_logfile_step2(recommender: str, dataset: str) -> None:
        """
        This method is to config the log file.
        """
        # Setup Log configuration
        setup_logging(
            log_error="error.log", log_info="info.log",
            save_path=PathDirFile.set_log_search_path(
                algorithm=recommender,
                dataset=dataset
            )
        )

    def print_basic_info(self) -> None:
        """
        This method is to print basic information about the step and machine.
        """
        # Logging machine data
        logger.info("$" * 50)
        self.machine_information()
        logger.info("-" * 50)
        # Logging the experiment setup
        logger.info("-" * 50)
        logger.info("[Search Step] SEARCH FOR THE BEST PARAMETER VALUES")
        logger.info(" ".join(['>>', 'Option:', self.experimental_settings['opt']]))
        if self.experimental_settings['opt'] == Label.CONFORMITY:
            logger.info(" ".join(['>>', 'Cluster:', str(self.experimental_settings['cluster'])]))
        elif self.experimental_settings['opt'] == Label.RECOMMENDER:
            logger.info(" ".join(['>>', 'Recommender:', str(self.experimental_settings['recommender'])]))

        logger.info(" ".join(['>>', 'Dataset:', str(self.experimental_settings['dataset'])]))
        logger.info(" ".join(['>>', 'Fold to use:', str(self.experimental_settings['fold'])]))
        logger.info(" ".join(['>>', 'Trial to use:', str(self.experimental_settings['trial'])]))

        logger.info("$" * 50)

    # ############################################################################################ #
    # ############################# Clustering Algorithm Optimization ###########################  #
    # ############################################################################################ #

    def starting_cluster(self) -> None:
        """
        TODO: Docstring
        """

        # Starting the counter
        # clock = Clocker()
        # clock.start_count()
        for dataset in self.experimental_settings['dataset']:
            # # Executing the Random Search
            search_instance = ManualConformityAlgorithmSearch(
                dataset_name=dataset,
                experiment_name=self.experimental_settings["experiment_name"],
                based_on=self.experimental_settings["based_on"],
                distribution_list=self.experimental_settings["distribution"],
                n_jobs=self.experimental_settings["n_jobs"],
                fold=self.experimental_settings["fold"],
                trial=self.experimental_settings["trial"],
                n_inter=self.experimental_settings["n_inter"],
            )
            for algorithm in self.experimental_settings['cluster']:
                # logger.info(f"Starting Algorithm: {algorithm}")
                print(f"Starting Algorithm: {algorithm}")
                search_instance.run(
                    conformity_str=algorithm
                )
        #
        # Finishing the counter
        # clock.finish_count()
        # #
        # # Saving execution time
        # SaveAndLoad.save_search_conformity_time(
        #     data=clock.clock_data(),
        #     dataset=self.experimental_settings['dataset'],
        #     algorithm=self.experimental_settings['cluster'],
        #     distribution=self.experimental_settings['distribution']
        # )

    # ############################################################################################ #
    #  ############################ Recommender Algorithm Optimization ########################### #
    # ############################################################################################ #

    def preparing_to_batch_recommender_search(self) -> None:
        """
        TODO: Docstring
        """

        combination = [
            self.experimental_settings['recommender'], self.experimental_settings['dataset'],
            [self.experimental_settings['trial']], [self.experimental_settings['fold']]
        ]

        system_combination = list(itertools.product(*combination))
        print("The total of process is: " + str(len(system_combination)))

        for recommender, dataset, trial, fold in system_combination:
            PierreStep2.starting_recommender_search(
                experiment_name=self.experimental_settings["experiment_name"],
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                n_inter=self.experimental_settings['n_inter'],
                n_jobs=self.experimental_settings['n_jobs'],
                n_threads=self.experimental_settings['n_threads'],
                based_on=self.experimental_settings['based_on'],
                multiprocessing_lib=self.experimental_settings['multiprocessing']
            )

    @staticmethod
    def starting_recommender_search(
            experiment_name: str,
            recommender: str, dataset: str, trial: int, fold: int,
            n_inter: int, n_jobs: int, n_threads: int, based_on: str,
            multiprocessing_lib: str
    ) -> None:
        """
        Method to start the recommender algorithm hyperparameter search optimization.
        """
        # Executing the Random Search

        if recommender in Label.SURPRISE_RECOMMENDERS:
            search_instance = SurpriseRandomSearch(
                experiment_name=experiment_name,
                algorithm=recommender, dataset_name=dataset, trial=trial, fold=fold,
                n_jobs=n_jobs, n_inter=n_inter, based_on=based_on,
                multiprocessing_lib=multiprocessing_lib
            )
        elif recommender in Label.IMPLICIT_RECOMMENDERS:
            search_instance = ImplicitGridSearch(
                experiment_name=experiment_name,
                algorithm=recommender, dataset_name=dataset, trial=trial, fold=fold,
                n_jobs=n_jobs, n_threads=n_threads, n_inter=n_inter, based_on=based_on,
                multiprocessing_lib=multiprocessing_lib
            )
        elif recommender in Label.PIERRE_RECOMMENDERS:
            search_instance = PierreGridSearch(
                experiment_name=experiment_name,
                algorithm=recommender, dataset_name=dataset, trial=trial, fold=fold,
                n_jobs=n_jobs, n_threads=n_threads, n_inter=n_inter, based_on=based_on,
                multiprocessing_lib=multiprocessing_lib
            )
        else:
            exit(0)
        search_instance.fit()

    # ############################################################################################ #
    #  ################################ Main Method and Step Starts #############################  #
    # ############################################################################################ #

    def main(self) -> None:
        """
        Main method used to choice the run option.
        """
        if self.experimental_settings['opt'] == Label.CONFORMITY:
            self.starting_cluster()
        elif self.experimental_settings['opt'] == Label.RECOMMENDER:
            self.preparing_to_batch_recommender_search()
        else:
            logger.info("Option not found!")


if __name__ == '__main__':
    """
    It starts the parameter search
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep2()
    step.read_the_entries()
    step.print_basic_info()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
