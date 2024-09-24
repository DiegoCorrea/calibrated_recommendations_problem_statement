import itertools

import logging

from joblib import Parallel, delayed

from checkpoint_verification import CheckpointVerification
from processing.implicit_recommender_algorithms import ImplicitRecommenderAlgorithm
from processing.pierre_recommender_algorithms import PierreRecommenderAlgorithm
from processing.surprise_recommender_algorithms import SurpriseRecommenderAlgorithm
from settings.labels import Label
from utils.clocker import Clocker
from utils.logging_settings import setup_logging
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad
from utils.input import Input
from utils.step import Step

logger = logging.getLogger(__name__)


def starting_recommender(
        experiment_name: str,
        dataset: str, recommender: str, trial: int, fold: int,
        checkpoint: str, metric: str, list_size: int, based_on: str
) -> None:
    """
    Function to starting the recommender algorithm.

    :param dataset: A string that's representing the dataset name.
    :param recommender: A string that's representing the recommender algorithm name.
    :param trial: The trial number.
    :param fold: The fold number.
    :param checkpoint: TODO
    :param metric: TODO
    :param list_size: TODO
    """

    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step3_verification(
            experiment_name=experiment_name, based_on= based_on,
            dataset=dataset, trial=trial, fold=fold, recommender=recommender
    ):
        logger.info(">> Already Done... " + system_name)
    else:
        try:
            if recommender in Label.SURPRISE_RECOMMENDERS:
                # Executing the processing step
                recommender_algorithm = SurpriseRecommenderAlgorithm(
                    experiment_name=experiment_name,
                    dataset_name=dataset, trial=trial, fold=fold, recommender_name=recommender,
                    metric=metric,
                    list_size=list_size, based_on=based_on
                )
                recommender_algorithm.run()
            elif recommender in Label.IMPLICIT_RECOMMENDERS:
                recommender_algorithm = ImplicitRecommenderAlgorithm(
                    experiment_name=experiment_name,
                    dataset_name=dataset, trial=trial, fold=fold, recommender_name=recommender,
                    list_size=list_size, based_on=based_on
                )
                recommender_algorithm.run()
            elif recommender in Label.PIERRE_RECOMMENDERS:
                recommender_algorithm = PierreRecommenderAlgorithm(
                    experiment_name=experiment_name,
                    dataset_name=dataset, trial=trial, fold=fold, recommender_name=recommender,
                    list_size=list_size, based_on=based_on
                )
                recommender_algorithm.run()
            else:
                pass
        except Exception as e:
            logger.error(">> Error... " + system_name)
            logger.exception(e)


class PierreStep3(Step):
    """
    This class is administrating the Step 3 of the framework (Processing)
    """

    def read_the_entries(self) -> None:
        """
        This method reads the terminal entries.
        """
        self.experimental_settings = Input.step3()

    @staticmethod
    def set_the_logfile_by_instance(dataset: str, algorithm: str, trial: int, fold: int) -> None:
        """
        This method is to config the log file.
        """
        # Setup Log configuration
        setup_logging(
            log_error="error.log", log_info="info.log",
            save_path=PathDirFile.set_log_processing_path(
                algorithm=algorithm,
                trial=trial,
                fold=fold,
                dataset=dataset
            )
        )

    def print_basic_info_by_instance(self, dataset: str, algorithm: str, trial: int, fold: int) -> None:
        """
        This method is to print basic information about the step and machine.
        """

        logger.info("$" * 50)
        logger.info("$" * 50)
        # Logging machine data
        self.machine_information()
        logger.info("-" * 50)

        # Logging the experiment setup
        logger.info("[PROCESSING STEP] - RECOMMENDER ALGORITHM")
        logger.info(" ".join(['>>', 'Dataset:', dataset]))
        logger.info(" ".join(['>>', 'Trial:', str(trial)]))
        logger.info(" ".join(['>>', 'Fold:', str(fold)]))
        logger.info(" ".join(['>>', 'Algorithm:', algorithm]))
        logger.info("$" * 50)
        logger.info("$" * 50)

    def main(self) -> None:
        """
        Main method used to choice the run option.
        """
        self.recommender_parallelization()

    def recommender_parallelization(self) -> None:
        """
        Main method to start the processing step in parallel.
        """
        combination = [
            [self.experimental_settings['experiment_name']],
            self.experimental_settings['recommender'], self.experimental_settings['dataset'],
            self.experimental_settings['fold'], self.experimental_settings['trial'],
            [self.experimental_settings['checkpoint']], [self.experimental_settings['metric']],
            [self.experimental_settings['list_size']], [self.experimental_settings['based_on']]
        ]

        system_combination = list(itertools.product(*combination))
        print("The total of process is: " + str(len(system_combination)))

        # Starting the recommender algorithm
        Parallel(n_jobs=self.experimental_settings['n_jobs'])(
            delayed(starting_recommender)(
                experiment_name=experiment_name,
                recommender=recommender, dataset=dataset, trial=trial, fold=fold, checkpoint=checkpoint,
                metric=metric, list_size=list_size, based_on=based_on
            ) for experiment_name, recommender, dataset, fold, trial, checkpoint, metric, list_size, based_on in system_combination
        )

        # Finishing the Step
        logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))


if __name__ == '__main__':
    """
    It starts the processing step
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep3()
    step.read_the_entries()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
