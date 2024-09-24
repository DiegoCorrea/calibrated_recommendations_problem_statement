import sys
import os

from datasets.registred_datasets import RegisteredDataset
from settings.constants import Constants
from settings.labels import Label
from settings.save_and_load import SaveAndLoad


class Input:
    """
    This class is responsible for reading the terminal/keyboard entries.
    """

    @staticmethod
    def __verify_checkpoint__(value):
        if value not in ["YES", "NO"]:
            print('Checkpoint option not found! Options is:')
            print(["YES", "NO"])
            exit(1)

    @staticmethod
    def __verify_multiprocessing__(value):
        if value not in Label.REGISTERED_MULTIPROCESSING_LIBS:
            print(f"The multiprocessing option {value} does not exist! Please check for a possible option.")
            print(["joblib", "starmap"])
            exit(1)

    @staticmethod
    def __verify_n_jobs__(value):
        if int(value) < -2 or int(value) > Constants.N_CORES:
            print('Number of Core out of range!')
            exit(1)

    @staticmethod
    def __verify_n_inter__(value):
        if int(value) < 1:
            print('Number out of range!')
            exit(1)

    @staticmethod
    def __verify_n_cv__(value):
        if int(value) < 1:
            print('Number out of range!')
            exit(1)

    @staticmethod
    def __verify_trial__(value):
        if int(value) <= 0:
            print('Trial out of range!')
            exit(1)

    @staticmethod
    def __verify_fold__(value):
        if int(value) <= 0:
            print('Fold out of range!')
            exit(1)

    @staticmethod
    def __verify_dataset__(value):
        if value not in RegisteredDataset.DATASET_LIST:
            print('Dataset not registered! All possibilities are:')
            print(RegisteredDataset.DATASET_LIST)
            exit(1)

    @staticmethod
    def __verify_cluster__(value):
        if value not in Label.REGISTERED_UNSUPERVISED:
            print('Cluster algorithm not registered! All possibilities are:')
            print(Label.REGISTERED_UNSUPERVISED)
            exit(1)

    @staticmethod
    def __verify_recommender__(value):
        if value not in Label.REGISTERED_RECOMMENDERS:
            print('Recommender not found! All possibilities are:')
            print(Label.REGISTERED_RECOMMENDERS)
            exit(1)

    @staticmethod
    def __verify_tradeoff__(value):
        if value not in Label.ACCESSIBLE_TRADEOFF_LIST:
            print('Tradeoff not registered! Options is:')
            print(Label.ACCESSIBLE_TRADEOFF_LIST)
            exit(1)

    @staticmethod
    def __verify_relevance__(value):
        if value not in Label.ACCESSIBLE_RELEVANCE_LIST:
            print('Relevance not registered! Options is:')
            print(Label.ACCESSIBLE_RELEVANCE_LIST)
            exit(1)

    @staticmethod
    def __verify_calibration__(value):
        if value not in Label.ACCESSIBLE_CALIBRATION_LIST:
            print('Calibration measure not registered! Options is:')
            print(Label.ACCESSIBLE_CALIBRATION_LIST)
            exit(1)

    @staticmethod
    def __verify_distribution__(value):
        if value not in Label.ACCESSIBLE_DISTRIBUTION_LIST:
            print('Distribution not registered! Options is:')
            print(Label.ACCESSIBLE_DISTRIBUTION_LIST)
            exit(1)

    @staticmethod
    def __verify_selector__(value):
        if value not in Label.ACCESSIBLE_SELECTOR_LIST:
            print('Selector not registered! Options is:')
            print(Label.ACCESSIBLE_SELECTOR_LIST)
            exit(1)

    @staticmethod
    def __verify_weight__(value):
        if value not in Label.ACCESSIBLE_WEIGHT_LIST:
            print('Tradeoff Weight not registered! Options is:')
            print(Label.ACCESSIBLE_WEIGHT_LIST)
            exit(1)

    @staticmethod
    def __verify_conformity__(value):
        if value not in Label.REGISTERED_UNSUPERVISED:
            print('Cluster algorithm not registered! All possibilities are:')
            print(Label.REGISTERED_UNSUPERVISED)
            exit(1)

    @staticmethod
    def __verify_metric__(value):
        if value not in Label.REGISTERED_METRICS:
            print(f'Metric {value} not found! Options is:')
            print(Label.REGISTERED_METRICS)
            exit(1)

    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    @staticmethod
    def default() -> dict:
        experimental_setup = dict()
        # Experimental setup information
        experimental_setup['experiment_name'] = "experiment_1"
        experimental_setup['opt'] = Label.DATASET_SPLIT
        experimental_setup['checkpoint'] = "NO"
        experimental_setup['opt'] = Label.EVALUATION_METRICS
        experimental_setup['metrics'] = Label.REGISTERED_METRICS

        experimental_setup['dataset'] = RegisteredDataset.DEFAULT_DATASET
        experimental_setup['n_folds'] = Constants.K_FOLDS_VALUE
        experimental_setup['n_trials'] = Constants.N_TRIAL_VALUE

        experimental_setup['recommender'] = Label.DEFAULT_REC
        experimental_setup['cluster'] = Label.DEFAULT_CLUSTERING

        experimental_setup['tradeoff'] = Label.ACCESSIBLE_TRADEOFF_LIST
        experimental_setup['fairness'] = Label.ACCESSIBLE_CALIBRATION_LIST
        experimental_setup['relevance'] = Label.ACCESSIBLE_RELEVANCE_LIST
        experimental_setup['weight'] = Label.ACCESSIBLE_WEIGHT_LIST
        experimental_setup['distribution'] = Label.ACCESSIBLE_DISTRIBUTION_LIST
        experimental_setup['selector'] = Label.ACCESSIBLE_SELECTOR_LIST

        return experimental_setup

    @staticmethod
    def step1() -> dict:
        """
        Method to read the settings from the terminal/keyboard. The possible options are:

        - opt can be: SPLIT, CHART, ANALYZE, and DISTRIBUTION. Ex: -opt=CHART

        - dataset can be: ml-1m, yahoo-movies (see the registered datasets). Ex: --dataset=ml-1m

        - n_folds can be: 1, 2, 3 or higher. Ex: --n_folds=5

        - n_trials can be: 1, 2, 3 or higher. Ex --n_trials=7

        - distribution can be: CWS, or WPS. Ex: --distribution=CWS

        - fold can be: 1, 2, 3 and others (based on the n_folds). Ex: --fold=5

        - trial can be: 1, 2, 3 and others (based on the n_trials). Ex: --trial=3

        :return: A dict with the input settings.
        """
        experimental_setup = dict()

        # Experimental setup information
        experimental_setup['experiment_name'] = "experiment_1"

        experimental_setup['opt'] = Label.DATASET_SPLIT
        experimental_setup['n_jobs'] = Constants.N_CORES

        experimental_setup['dataset'] = RegisteredDataset.DEFAULT_DATASET
        experimental_setup['n_folds'] = Constants.K_FOLDS_VALUE
        experimental_setup['n_trials'] = Constants.N_TRIAL_VALUE

        experimental_setup['distribution'] = Label.DEFAULT_DISTRIBUTION
        experimental_setup['fold'] = 1
        experimental_setup['trial'] = 1

        if sys.argv[1].split('=')[0] != "from_file" and len(sys.argv) > 2:
            for arg in sys.argv[1:]:
                param, value = arg.split('=')

                # Reading the work 'Option' (-opt) from the terminal entrance
                if param == '-opt':
                    if value not in Label.PREPROCESSING_OPTS:
                        print(f'Option {value} does not exists!')
                        print("The possibilities are: ", Label.PREPROCESSING_OPTS)
                        exit(1)
                    experimental_setup['opt'] = str(value)

                # Reading the work 'Number of Jobs' (-n_jobs) from the terminal entrance
                elif param == '-n_jobs':
                    Input.__verify_n_jobs__(value=value)
                    experimental_setup['n_jobs'] = value

                # Reading the work 'Number of Folds' (--n_folds) from the terminal entrance
                elif param == '--n_folds':
                    if int(value) < 3:
                        print('The lower accepted value is 3!')
                        exit(1)
                    experimental_setup['n_folds'] = int(value)

                # Reading the work 'Number of Trials' (--n_trials) from the terminal entrance
                elif param == '--n_trials':
                    if int(value) < 1:
                        print('Only positive numbers are accepted!')
                        exit(1)
                    experimental_setup['n_trials'] = int(value)

                # Reading the work 'Dataset' (--dataset) from the terminal entrance
                elif param == '--dataset':
                    Input.__verify_dataset__(value)
                    experimental_setup['dataset'] = value

                # Reading the work 'Fold Number' (--fold) from the terminal entrance
                elif param == '--fold':
                    Input.__verify_fold__(value=value)
                    experimental_setup['fold'] = value

                # Reading the work 'Trial Number' (--trial) from the terminal entrance
                elif param == '--trial':
                    Input.__verify_trial__(value=value)
                    experimental_setup['trial'] = value

                # Reading the work 'Distribution' (--distribution) from the terminal entrance
                elif param == '--distribution':
                    Input.__verify_distribution__(value=value)
                    experimental_setup['distribution'] = [value]

                else:
                    print(f"The parameter {param} is not configured in this feature.")
                    exit(1)
        elif sys.argv[1].split('=')[0] == "from_file" and sys.argv[1].split('=')[1] == "YES" and sys.argv[2].split('=')[
            0] == "file_name":
            experimental_setup = SaveAndLoad.load_step_file(step="step1", file_name=sys.argv[2].split('=')[1])
            os.environ = experimental_setup
        else:
            print("More information are needed!")
            print("All params possibilities are: \n"
                  "-opt, --dataset, --n_folds, --n_trials, --fold, --trial, --distribution.")
            print("Example: python step1_preprocessing.py -opt=SPLIT --dataset=ml-1m --n_trials=10 --n_folds=5")
            exit(1)
        return experimental_setup

    @staticmethod
    def step2() -> dict:
        """
        Function to read the settings from the terminal. The possible options are:

        - opt: TODO: Docstring

        - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

        - recommender can be: SVD, SVD++, NMF and others.

        - cluster: TODO: Docstring

        - distribution can be: CWS, or WPS. Ex: --distribution=CWS

        :return: A dict with the input settings.
        """
        experimental_setup = dict()
        experimental_setup['experiment_name'] = "experiment_1"
        experimental_setup['n_inter'] = Constants.N_INTER
        experimental_setup['n_jobs'] = Constants.N_CORES
        experimental_setup['n_cv'] = Constants.K_FOLDS_VALUE
        experimental_setup['dataset'] = RegisteredDataset.DEFAULT_DATASET
        experimental_setup['recommender'] = Label.DEFAULT_REC
        experimental_setup['distribution'] = Label.DEFAULT_DISTRIBUTION
        experimental_setup['cluster'] = Label.REGISTERED_UNSUPERVISED
        experimental_setup['fold'] = None
        experimental_setup['trial'] = None

        if sys.argv[1].split('=')[0] != "from_file" and len(sys.argv) > 1:
            for arg in sys.argv[1:]:
                param, value = arg.split('=')

                # Reading the work 'Option' (-opt) from the terminal entrance
                if param == '-opt':
                    if value not in Label.SEARCH_OPTS:
                        print(f'This option does not exists! {value}')
                        exit(1)
                    experimental_setup['opt'] = str(value)

                # Reading the work 'Number of Cross Validation' (-n_cv) from the terminal entrance
                elif param == '-n_cv':
                    Input.__verify_n_cv__(value=value)
                    experimental_setup['n_cv'] = value

                # Reading the work 'Number of Interactions' (-n_inter) from the terminal entrance
                elif param == '-n_inter':
                    Input.__verify_n_jobs__(value=value)
                    experimental_setup['n_inter'] = value

                # Reading the work 'Number of Jobs' (-n_jobs) from the terminal entrance
                elif param == '-n_jobs':
                    Input.__verify_n_jobs__(value=value)
                    experimental_setup['n_jobs'] = value

                # Reading the work 'Recommender Algorithm' (--recommender) from the terminal entrance
                elif param == '--recommender':
                    Input.__verify_recommender__(value=value)
                    experimental_setup['recommender'] = value

                # Reading the work 'Dataset' (--dataset) from the terminal entrance
                elif param == '--dataset':
                    Input.__verify_dataset__(value)
                    experimental_setup['dataset'] = [value]

                # Reading the work 'Cluster Algorithm' (--cluster) from the terminal entrance
                elif param == '--cluster':
                    Input.__verify_cluster__(value=value)
                    experimental_setup['cluster'] = [value]

                # Reading the work 'Distribution' (--distribution) from the terminal entrance
                elif param == '--distribution':
                    Input.__verify_distribution__(value=value)
                    experimental_setup['distribution'] = value

                # Reading the work 'Fold Number' (--fold) from the terminal entrance
                elif param == '--fold':
                    Input.__verify_fold__(value=value)
                    experimental_setup['fold'] = value

                # Reading the work 'Trial Number' (--trial) from the terminal entrance
                elif param == '--trial':
                    Input.__verify_trial__(value=value)
                    experimental_setup['trial'] = value
                else:
                    print(f"The parameter {param} is not configured in this feature.")
        elif sys.argv[1].split('=')[0] == "from_file" and sys.argv[1].split('=')[1] == "YES" and sys.argv[2].split('=')[
            0] == "file_name":
            experimental_setup = SaveAndLoad.load_step_file(step="step2", file_name=sys.argv[2].split('=')[1])
            os.environ = experimental_setup
        else:
            print("More information are needed!")
            print("All params possibilities are: -opt, --dataset, --recommender, --cluster, --distribution.")
            print("Example: python step2_random_search.py -opt-RECOMMENDER --recommender=SVD --dataset=ml-1m")
            exit(1)
        return experimental_setup

    @staticmethod
    def step3() -> dict:
        """
        Function to read the settings from the terminal. The possible options are:

        - opt: TODO: Docstring

        - recommender can be: SVD, SVD++, NMF and others.

        - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

        - fold can be: 1, 2, 3 and others (based on the preprocessing n_folds).

        - trial can be: 1, 2, 3 and others (based on the preprocessing n_trials).

        - metric: Metric used to load the best hyperparameters. It can be: rmse, mae, mse, and fcp.

        :return: A dict with the input settings.
        """
        experimental_setup = dict()
        experimental_setup['experiment_name'] = "experiment_1"
        experimental_setup['opt'] = Label.RECOMMENDER
        experimental_setup['recommender'] = Label.DEFAULT_REC
        experimental_setup['dataset'] = RegisteredDataset.DEFAULT_DATASET
        experimental_setup['fold'] = [fold for fold in range(1, Constants.K_FOLDS_VALUE + 1)]
        experimental_setup['trial'] = [trial for trial in range(1, Constants.N_TRIAL_VALUE + 1)]
        experimental_setup['metric'] = 'rmse'

        if sys.argv[1].split('=')[0] != "from_file" and len(sys.argv) > 2:
            for arg in sys.argv[1:]:
                param, value = arg.split('=')

                # Reading the work 'Option' (-opt) from the terminal entrance
                if param == '-opt':
                    if value not in Label.SEARCH_OPTS:
                        print(f'This option does not exists! {value}! All possibilities are:')
                        print(Label.SEARCH_OPTS)
                        exit(1)
                    experimental_setup['opt'] = str(value)

                # Reading the work 'Recommender Algorithm' (--recommender) from the terminal entrance
                elif param == '--recommender':
                    Input.__verify_recommender__(value=value)
                    experimental_setup['recommender'] = value

                # Reading the work 'Dataset' (--dataset) from the terminal entrance
                elif param == '--dataset':
                    Input.__verify_dataset__(value)
                    experimental_setup['dataset'] = [value]

                # Reading the work 'Fold Number' (--fold) from the terminal entrance
                elif param == '--fold':
                    Input.__verify_fold__(value=value)
                    experimental_setup['fold'] = [value]

                # Reading the work 'Trial Number' (--trial) from the terminal entrance
                elif param == '--trial':
                    Input.__verify_trial__(value=value)
                    experimental_setup['trial'] = [value]

                # Reading the work 'Search Optimization Metric' (--metric) from the terminal entrance
                elif param == '--metric':
                    if value not in Label.SEARCH_METRICS:
                        print(f'This Metric does not exists! {value}! All possibilities are:')
                        print(Label.SEARCH_METRICS)
                        exit(1)
                    experimental_setup['metric'] = str(value)
                else:
                    print(f"The parameter {param} is not configured in this feature.")
        elif sys.argv[1].split('=')[0] == "from_file" and sys.argv[1].split('=')[1] == "YES" and sys.argv[2].split('=')[
            0] == "file_name":
            experimental_setup = SaveAndLoad.load_step_file(step="step3", file_name=sys.argv[2].split('=')[1])
            os.environ = experimental_setup
        else:
            print("More information are needed!")
            print("All params possibilities are: --dataset, --recommender, --trial and --fold.")
            print("Example: python step3_processing.py --dataset=yahoo-movies --recommender=SVD --trial=1 --fold=1")
            exit(1)
        return experimental_setup

    @staticmethod
    def step4() -> dict:
        """
        Function to read the settings from the terminal. The possible options are:

        - checkpoint: TODO: Docstring

        - recommender can be: SVD, SVD++, NMF and others.

        - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

        - fold can be: 1, 2, 3 and others (based on the preprocessing n_folds).

        - trial can be: 1, 2, 3 and others (based on the preprocessing n_trials).

        - tradeoff: TODO: Docstring

        - calibration: TODO: Docstring

        - relevance: TODO: Docstring

        - weight: TODO: Docstring

        - distribution: TODO: Docstring

        - selector: TODO: Docstring

        :return: A dict with the input settings.
        """
        experimental_setup = dict()
        experimental_setup['experiment_name'] = "experiment_1"
        experimental_setup['n_jobs'] = Constants.N_CORES
        experimental_setup['multiprocessing'] = Label.DEFAULT_MULTIPROCESSING_LIB
        experimental_setup['checkpoint'] = "NO"

        experimental_setup['recommender'] = Label.REGISTERED_RECOMMENDERS

        experimental_setup['dataset'] = [RegisteredDataset.DEFAULT_DATASET]
        experimental_setup['fold'] = list(range(1, Constants.K_FOLDS_VALUE + 1))
        experimental_setup['trial'] = list(range(1, Constants.N_TRIAL_VALUE + 1))

        experimental_setup['tradeoff'] = Label.ACCESSIBLE_TRADEOFF_LIST
        experimental_setup['fairness'] = Label.ACCESSIBLE_CALIBRATION_LIST
        experimental_setup['relevance'] = Label.ACCESSIBLE_RELEVANCE_LIST
        experimental_setup['weight'] = Label.ACCESSIBLE_WEIGHT_LIST
        experimental_setup['distribution'] = Label.ACCESSIBLE_DISTRIBUTION_LIST
        experimental_setup['selector'] = Label.ACCESSIBLE_SELECTOR_LIST

        experimental_setup['list_size'] = [Constants.RECOMMENDATION_LIST_SIZE]
        experimental_setup['alpha'] = [Constants.ALPHA_VALUE]

        experimental_setup['d'] = [Constants.DIMENSION_VALUE]

        if sys.argv[1].split('=')[0] != "from_file" and len(sys.argv) > 2:
            for arg in sys.argv[1:]:
                param, value = arg.split('=')

                # Reading the work 'Checkpoint' (-checkpoint) from the terminal entrance
                if param == '-checkpoint':
                    Input.__verify_checkpoint__(value=value)
                    experimental_setup["checkpoint"] = value

                # Reading the work 'Number of Jobs' (-n_jobs) from the terminal entrance
                elif param == '-n_jobs':
                    Input.__verify_n_jobs__(value=value)
                    experimental_setup['n_jobs'] = value

                # Reading the work 'Multiprocessing Library' (-multiprocessing) from the terminal entrance
                elif param == '-multiprocessing':
                    Input.__verify_multiprocessing__(value=value)
                    experimental_setup['multiprocessing'] = value

                # Reading the work 'Recommender Algorithm' (--recommender) from the terminal entrance
                elif param == '--recommender':
                    Input.__verify_recommender__(value=value)
                    experimental_setup['recommender'] = [value]

                # Reading the work 'Dataset' (--dataset) from the terminal entrance
                elif param == '--dataset':
                    Input.__verify_dataset__(value)
                    experimental_setup['dataset'] = [value]

                # Reading the work 'Fold Number' (--fold) from the terminal entrance
                elif param == '--fold':
                    Input.__verify_fold__(value=value)
                    experimental_setup['fold'] = [value]

                # Reading the work 'Trial Number' (--trial) from the terminal entrance
                elif param == '--trial':
                    Input.__verify_trial__(value=value)
                    experimental_setup['trial'] = [value]

                # Reading the work 'Tradeoff Balance' (--tradeoff) from the terminal entrance
                elif param == '--tradeoff':
                    Input.__verify_tradeoff__(value=value)
                    experimental_setup['tradeoff'] = [value]

                # Reading the work 'Relevance Measure' (--relevance) from the terminal entrance
                elif param == '--relevance':
                    Input.__verify_relevance__(value=value)
                    experimental_setup['relevance'] = [value]

                # Reading the work 'Calibration Measure' (--calibration) from the terminal entrance
                elif param == '--calibration':
                    Input.__verify_calibration__(value=value)
                    experimental_setup['fairness'] = [value]

                # Reading the work 'Distribution Equation' (--distribution) from the terminal entrance
                elif param == '--distribution':
                    Input.__verify_distribution__(value=value)
                    experimental_setup['distribution'] = [value]

                # Reading the work 'Selector Algorithm' (--selector) from the terminal entrance
                elif param == '--selector':
                    Input.__verify_selector__(value=value)
                    experimental_setup['selector'] = [value]

                # Reading the work 'Tradeoff Weight' (--weight) from the terminal entrance
                elif param == '--weight':
                    Input.__verify_weight__(value=value)
                    experimental_setup['weight'] = [value]
                else:
                    print("The parameter {} is not configured in this feature.".format(param))
        elif sys.argv[1].split('=')[0] == "from_file" and sys.argv[1].split('=')[1] == "YES" and sys.argv[2].split('=')[
            0] == "file_name":
            experimental_setup = SaveAndLoad.load_step_file(step="step4", file_name=sys.argv[2].split('=')[1])
            os.environ = experimental_setup
        else:
            print("More information are needed!")
            exit(1)
        return experimental_setup

    @staticmethod
    def step5() -> dict:
        """
        Function to read the settings from the terminal. The possible options are:

        - opt: TODO: Docstring

        - checkpoint: TODO: Docstring

        - recommender can be: SVD, SVD++, NMF and others.

        - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

        - fold can be: 1, 2, 3 and others (based on the preprocessing n_folds).

        - trial can be: 1, 2, 3 and others (based on the preprocessing n_trials).

        - cluster: TODO: Docstring

        - tradeoff: TODO: Docstring

        - calibration: TODO: Docstring

        - relevance: TODO: Docstring

        - weight: TODO: Docstring

        - distribution: TODO: Docstring

        - selector: TODO: Docstring

        :return: A dict with the input settings.
        """
        experimental_setup = dict()
        experimental_setup['experiment_name'] = "experiment_1"
        experimental_setup['n_jobs'] = Constants.N_CORES
        experimental_setup['multiprocessing'] = Label.DEFAULT_MULTIPROCESSING_LIB
        experimental_setup['checkpoint'] = "NO"
        experimental_setup['opt'] = Label.EVALUATION_METRICS
        experimental_setup['metric'] = Label.REGISTERED_METRICS

        experimental_setup['recommender'] = Label.REGISTERED_RECOMMENDERS
        experimental_setup['cluster'] = Label.REGISTERED_UNSUPERVISED

        experimental_setup['dataset'] = [RegisteredDataset.DEFAULT_DATASET]
        experimental_setup['fold'] = list(range(1, Constants.K_FOLDS_VALUE + 1))
        experimental_setup['trial'] = list(range(1, Constants.N_TRIAL_VALUE + 1))

        experimental_setup['tradeoff'] = Label.ACCESSIBLE_TRADEOFF_LIST
        experimental_setup['fairness'] = Label.ACCESSIBLE_CALIBRATION_LIST
        experimental_setup['relevance'] = Label.ACCESSIBLE_RELEVANCE_LIST
        experimental_setup['weight'] = Label.ACCESSIBLE_WEIGHT_LIST
        experimental_setup['distribution'] = Label.ACCESSIBLE_DISTRIBUTION_LIST
        experimental_setup['selector'] = Label.ACCESSIBLE_SELECTOR_LIST

        if sys.argv[1].split('=')[0] != "from_file" and len(sys.argv) > 2:
            for arg in sys.argv[1:]:
                param, value = arg.split('=')

                # Reading the work 'Option' (-opt) from the terminal entrance
                if param == '-opt':
                    if value not in Label.METRIC_OPT:
                        print(f'This option does not exists! {value}... All possibilities are:')
                        print(Label.METRIC_OPT)
                        exit(1)
                    experimental_setup['opt'] = str(value)

                # Reading the work 'Checkpoint' (-checkpoint) from the terminal entrance
                elif param == '-checkpoint':
                    Input.__verify_checkpoint__(value=value)
                    experimental_setup["checkpoint"] = value

                # Reading the work 'Number of Jobs' (-n_jobs) from the terminal entrance
                elif param == '-n_jobs':
                    Input.__verify_n_jobs__(value=value)
                    experimental_setup['n_jobs'] = value

                # Reading the work 'Multiprocessing Library' (-multiprocessing) from the terminal entrance
                elif param == '-multiprocessing':
                    Input.__verify_multiprocessing__(value=value)
                    experimental_setup['multiprocessing'] = value

                # Reading the work 'Evaluation Metric' (--metric) from the terminal entrance
                elif param == '--metric':
                    Input.__verify_metric__(value=value)
                    experimental_setup['metric'] = [value]

                # Reading the work 'Cluster Algorithm' (--cluster) from the terminal entrance
                elif param == '--cluster':
                    Input.__verify_cluster__(value=value)
                    experimental_setup['cluster'] = [value]

                # Reading the work 'Recommender Algorithm' (--recommender) from the terminal entrance
                elif param == '--recommender':
                    Input.__verify_recommender__(value=value)
                    experimental_setup['recommender'] = [value]

                # Reading the work 'Dataset' (--dataset) from the terminal entrance
                elif param == '--dataset':
                    Input.__verify_dataset__(value)
                    experimental_setup['dataset'] = [value]

                # Reading the work 'Fold Number' (--fold) from the terminal entrance
                elif param == '--fold':
                    Input.__verify_fold__(value=value)
                    experimental_setup['fold'] = [value]

                # Reading the work 'Trial Number' (--trial) from the terminal entrance
                elif param == '--trial':
                    Input.__verify_trial__(value=value)
                    experimental_setup['trial'] = [value]

                # Reading the work 'Tradeoff Balance' (--tradeoff) from the terminal entrance
                elif param == '--tradeoff':
                    Input.__verify_tradeoff__(value=value)
                    experimental_setup['tradeoff'] = [value]

                # Reading the work 'Relevance Measure' (--relevance) from the terminal entrance
                elif param == '--relevance':
                    Input.__verify_relevance__(value=value)
                    experimental_setup['relevance'] = [value]

                # Reading the work 'Calibration Measure' (--calibration) from the terminal entrance
                elif param == '--calibration':
                    Input.__verify_calibration__(value=value)
                    experimental_setup['fairness'] = [value]

                # Reading the work 'Distribution Equation' (--distribution) from the terminal entrance
                elif param == '--distribution':
                    Input.__verify_distribution__(value=value)
                    experimental_setup['distribution'] = [value]

                # Reading the work 'Selector Algorithm' (--selector) from the terminal entrance
                elif param == '--selector':
                    Input.__verify_selector__(value=value)
                    experimental_setup['selector'] = [value]

                # Reading the work 'Tradeoff Weight' (--weight) from the terminal entrance
                elif param == '--weight':
                    Input.__verify_weight__(value=value)
                    experimental_setup['weight'] = [value]
                else:
                    print("The parameter {} is not configured in this feature.".format(param))
        elif sys.argv[1].split('=')[0] == "from_file" and sys.argv[1].split('=')[1] == "YES" and sys.argv[2].split('=')[
            0] == "file_name":
            experimental_setup = SaveAndLoad.load_step_file(step="step5", file_name=sys.argv[2].split('=')[1])
            os.environ = experimental_setup
        else:
            print("More information are needed!")
            exit(1)
        return experimental_setup

    @staticmethod
    def step6() -> dict:
        """
        Function to read the settings from the terminal. The possible options are:

        - opt: TODO: Docstring

        - metrics: TODO: Docstring

        - recommender can be: SVD, SVD++, NMF and others.

        - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

        - cluster: TODO: Docstring

        - tradeoff: TODO: Docstring

        - calibration: TODO: Docstring

        - relevance: TODO: Docstring

        - weight: TODO: Docstring

        - distribution: TODO: Docstring

        - selector: TODO: Docstring

        :return: A dict with the input settings.
        """
        experimental_setup = dict()
        experimental_setup['experiment_name'] = "experiment_1"
        experimental_setup['file'] = "NO"
        experimental_setup['opt'] = Label.EVALUATION_METRICS

        experimental_setup['metric'] = Label.REGISTERED_METRICS
        experimental_setup['recommender'] = Label.REGISTERED_RECOMMENDERS
        experimental_setup['conformity'] = Label.REGISTERED_UNSUPERVISED

        experimental_setup['dataset'] = RegisteredDataset.DATASET_LIST

        experimental_setup['tradeoff'] = Label.ACCESSIBLE_TRADEOFF_LIST
        experimental_setup['fairness'] = Label.ACCESSIBLE_CALIBRATION_LIST
        experimental_setup['relevance'] = Label.ACCESSIBLE_RELEVANCE_LIST
        experimental_setup['weight'] = Label.ACCESSIBLE_WEIGHT_LIST
        experimental_setup['distribution'] = Label.ACCESSIBLE_DISTRIBUTION_LIST
        experimental_setup['selector'] = Label.ACCESSIBLE_SELECTOR_LIST

        if sys.argv[1].split('=')[0] != "from_file" and len(sys.argv) > 2:
            for arg in sys.argv[1:]:
                param, value = arg.split('=')

                # Reading the work 'Option' (-opt) from the terminal entrance
                if param == '-opt':
                    if value not in Label.METRIC_OPT:
                        print(f'This option does not exists! {value}... All possibilities are:')
                        print(Label.METRIC_OPT)
                        exit(1)
                    experimental_setup['opt'] = str(value)

                # Reading the work 'Evaluation Metric' (--metric) from the terminal entrance
                elif param == '--metric':
                    Input.__verify_metric__(value=value)
                    experimental_setup['metric'] = [value]

                # Reading the work 'Conformity Metric' (--conformity) from the terminal entrance
                elif param == '--conformity':
                    Input.__verify_conformity__(value)
                    experimental_setup['conformity'] = [value]

                # Reading the work 'Recommender Algorithm' (--recommender) from the terminal entrance
                elif param == '--recommender':
                    Input.__verify_recommender__(value=value)
                    experimental_setup['recommender'] = [value]

                # Reading the work 'Dataset' (--dataset) from the terminal entrance
                elif param == '--dataset':
                    Input.__verify_dataset__(value)
                    experimental_setup['dataset'] = [value]

                # Reading the work 'Tradeoff Balance' (--tradeoff) from the terminal entrance
                elif param == '--tradeoff':
                    Input.__verify_tradeoff__(value=value)
                    experimental_setup['tradeoff'] = [value]

                # Reading the work 'Relevance Measure' (--relevance) from the terminal entrance
                elif param == '--relevance':
                    Input.__verify_relevance__(value=value)
                    experimental_setup['relevance'] = [value]

                # Reading the work 'Calibration Measure' (--calibration) from the terminal entrance
                elif param == '--calibration':
                    Input.__verify_calibration__(value=value)
                    experimental_setup['fairness'] = [value]

                # Reading the work 'Distribution Equation' (--distribution) from the terminal entrance
                elif param == '--distribution':
                    Input.__verify_distribution__(value=value)
                    experimental_setup['distribution'] = [value]

                # Reading the work 'Selector Algorithm' (--selector) from the terminal entrance
                elif param == '--selector':
                    Input.__verify_selector__(value=value)
                    experimental_setup['selector'] = [value]

                # Reading the work 'Tradeoff Weight' (--weight) from the terminal entrance
                elif param == '--weight':
                    Input.__verify_weight__(value=value)
                    experimental_setup['weight'] = [value]
                else:
                    print("The parameter {} is not configured in this feature.".format(param))
        elif sys.argv[1].split('=')[0] == "from_file" and sys.argv[1].split('=')[1] == "YES" and sys.argv[2].split('=')[
            0] == "file_name":
            experimental_setup = SaveAndLoad.load_step_file(step="step6", file_name=sys.argv[2].split('=')[1])
            os.environ = experimental_setup
        else:
            print("More information are needed!")
            exit(1)
        return experimental_setup

    @staticmethod
    def step7() -> dict:
        """
        Function to read the settings from the terminal. The possible options are:

        - opt can be: CHART, ANALYZE.

        - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

        :return: A dict with the input settings.
        """
        experimental_setup = dict()

        experimental_setup['experiment_name'] = "experiment_1"
        experimental_setup['opt'] = Label.EVALUATION_METRICS
        experimental_setup['view'] = Label.DATASET_CHART
        experimental_setup['dataset'] = RegisteredDataset.DATASET_LIST

        experimental_setup['conformity'] = Label.REGISTERED_UNSUPERVISED
        experimental_setup['type'] = Label.EVALUATION_VIEWS
        experimental_setup['goal'] = Label.EVALUATION_VIEWS

        experimental_setup['metric'] = Label.REGISTERED_METRICS

        if sys.argv[1].split('=')[0] != "from_file" and len(sys.argv) > 2:
            for arg in sys.argv[1:]:
                param, value = arg.split('=')

                # Reading the work 'Option' (-opt) from the terminal entrance
                if param == '-opt':
                    if value not in Label.METRIC_OPT:
                        print(f'This option does not exists! {value}... All possibilities are:')
                        print(Label.METRIC_OPT)
                        exit(1)
                    experimental_setup['opt'] = str(value)

                # Reading the work 'View' (-view) from the terminal entrance
                elif param == '-view':
                    if value not in Label.EVALUATION_VIEWS:
                        print(f'This View does not exists! {value}... All possibilities are:')
                        print(Label.EVALUATION_VIEWS)
                        exit(1)
                    experimental_setup['view'] = str(value)

                # Reading the work 'Dataset' (--dataset) from the terminal entrance
                elif param == '--dataset':
                    Input.__verify_dataset__(value)
                    experimental_setup['dataset'] = [value]

                # Reading the work 'Graphic Type' (--type) from the terminal entrance
                elif param == '--type':
                    if value not in Label.REGISTERED_GRAPHICS_TYPE:
                        print(f'This Graphic Type does not exists! {value}... All possibilities are:')
                        print(Label.REGISTERED_GRAPHICS_TYPE)
                        exit(1)
                    experimental_setup['type'] = str(value)

                # Reading the work 'Graphic Goal' (--goal) from the terminal entrance
                elif param == '--goal':
                    if value not in Label.REGISTERED_GRAPHICS_GOALS:
                        print(f'This Graphic Goal does not exists! {value}... All possibilities are:')
                        print(Label.REGISTERED_GRAPHICS_GOALS)
                        exit(1)
                    experimental_setup['goal'] = str(value)

                # Reading the work 'Evaluation Metric' (-metric) from the terminal entrance
                elif param == '--metric':
                    Input.__verify_metric__(value=value)
                    experimental_setup['metric'] = [value]

                # Reading the work 'Conformity Metric' (--conformity) from the terminal entrance
                elif param == '--conformity':
                    Input.__verify_conformity__(value)
                    experimental_setup['conformity'] = [value]

                # Reading the work 'Recommender Algorithm' (--recommender) from the terminal entrance
                elif param == '--recommender':
                    Input.__verify_recommender__(value=value)
                    experimental_setup['recommender'] = [value]

                # Reading the work 'Tradeoff Balance' (--tradeoff) from the terminal entrance
                elif param == '--tradeoff':
                    Input.__verify_tradeoff__(value=value)
                    experimental_setup['tradeoff'] = [value]

                # Reading the work 'Relevance Measure' (--relevance) from the terminal entrance
                elif param == '--relevance':
                    Input.__verify_relevance__(value=value)
                    experimental_setup['relevance'] = [value]

                # Reading the work 'Calibration Measure' (--calibration) from the terminal entrance
                elif param == '--calibration':
                    Input.__verify_calibration__(value=value)
                    experimental_setup['fairness'] = [value]

                # Reading the work 'Distribution Equation' (--distribution) from the terminal entrance
                elif param == '--distribution':
                    Input.__verify_distribution__(value=value)
                    experimental_setup['distribution'] = [value]

                # Reading the work 'Selector Algorithm' (--selector) from the terminal entrance
                elif param == '--selector':
                    Input.__verify_selector__(value=value)
                    experimental_setup['selector'] = [value]

                # Reading the work 'Tradeoff Weight' (--weight) from the terminal entrance
                elif param == '--weight':
                    Input.__verify_weight__(value=value)
                    experimental_setup['weight'] = [value]
                else:
                    print("The parameter {} is not configured in this feature.".format(param))
        elif sys.argv[1].split('=')[0] == "from_file" and sys.argv[1].split('=')[1] == "YES" and sys.argv[2].split('=')[
            0] == "file_name":
            experimental_setup = SaveAndLoad.load_step_file(step="step7", file_name=sys.argv[2].split('=')[1])
            os.environ = experimental_setup
        else:
            print("More information are needed!")
            print("All params possibilities are: -opt and --dataset.")
            print("Example: python step7_charts_analises.py -opt=CHART --dataset=yahoo-movies")
            exit(1)
        return experimental_setup
