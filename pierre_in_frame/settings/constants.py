import multiprocessing
from utils.utils import get_size

import os


class Constants:
    """
    Constants class.
    Responsible for leading with all constant values used by the system
    """

    # Machine config
    MEM_RAM = get_size()
    N_CORES = multiprocessing.cpu_count()
    # N_CORES = 1
    os.environ['NUMEXPR_MAX_THREADS'] = str(N_CORES)

    # SEARCH
    N_INTER = 50

    # Algorithm hyper param
    ALPHA_VALUE = 0.01

    # Data Model config
    RECOMMENDATION_LIST_SIZE = 10
    CANDIDATES_LIST_SIZE = 100

    # Minkowski Distance param
    DIMENSION_VALUE = 3

    # Experimental
    K_FOLDS_VALUE = 3
    N_TRIAL_VALUE = 1
    PROFILE_LEN_CUT_VALUE = 150
    # PROFILE_LEN_CUT_VALUE = K_FOLDS_VALUE * RECOMMENDATION_LIST_SIZE
    NORMALIZED_SCORE = True
