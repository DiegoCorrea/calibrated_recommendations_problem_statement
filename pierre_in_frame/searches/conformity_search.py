import random

import itertools

import logging

from fcmeans import FCM
from joblib import Parallel, delayed
from numpy import mean
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.cluster import (DBSCAN, OPTICS, Birch, AgglomerativeClustering, KMeans,
                             SpectralClustering, BisectingKMeans)
from sklearn.linear_model import SGDOneClassSVM
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.metrics import silhouette_score

from searches.parameters import ConformityParams
from settings.labels import Label
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class ManualConformityAlgorithmSearch:
    """
    Class used to lead with the Manual Search of Unsupervised Algorithms
    """

    def __init__(
        self,
        dataset_name: str, distribution_list: list,
        n_jobs: int, fold: int, trial: int, n_inter: int
    ):
        self.dataset_name = dataset_name
        self.n_jobs = n_jobs
        self.fold = fold
        self.trial = trial
        self.n_inter = n_inter

        self.distribution_list = distribution_list

        self.param_grid = ConformityParams.CLUSTER_PARAMS_GRID
        self.cluster_params = ConformityParams.CLUSTER_PARAMS
        self.component_params = ConformityParams.COMPONENT_PARAMS_GRID
        self.estimators_params = ConformityParams.ESTIMATORS_PARAMS_GRID
        self.neighbor_params = ConformityParams.NEIGHBOR_PARAMS_GRID
        self.outlier_params = ConformityParams.OUTLIEAR_PARAMS_GRID

    def preparing_data(self, distribution):
        """
        Prepares the dataset for pierre search.
        """
        users_distribution_list = []
        for trial in range(1, self.trial + 1):
            for fold in range(1, self.fold + 1):
                users_distribution_list.append(SaveAndLoad.load_user_preference_distribution(
                    dataset=self.dataset_name, trial=trial, fold=fold,
                    distribution=distribution
                ))
        return users_distribution_list

    @staticmethod
    def load_conformity_algorithm_instance(conformity_str, params):
        """
        It prepares the algorithm instance.
        """

        # Cluster Models

        # # K-Means Variations
        if conformity_str == Label.KMEANS:
            return KMeans(n_clusters=params['n_clusters'], init='k-means++', n_init="auto")
        elif conformity_str == Label.FCM:
            return FCM(n_clusters=params['n_clusters'])
        elif conformity_str == Label.BISECTING:
            return BisectingKMeans(n_clusters=params['n_clusters'], init='k-means++')
        # # Hierarchical Variations
        elif conformity_str == Label.AGGLOMERATIVE:
            return AgglomerativeClustering(n_clusters=params['n_clusters'])
        # # Spectral Variations
        elif conformity_str == Label.SPECTRAL:
            return SpectralClustering(n_clusters=params['n_clusters'])
        # # Tree Variations
        elif conformity_str == Label.BIRCH:
            return Birch(n_clusters=params['n_clusters'])
        # # Search Variations
        elif conformity_str == Label.DBSCAN:
            return DBSCAN(min_samples=params['min_samples'], eps=params['eps'], metric=params['metric'])
        elif conformity_str == Label.OPTICS:
            return OPTICS(min_samples=params['min_samples'], eps=params['eps'], metric=params['metric'])

        # Mixture Models
        elif conformity_str == Label.GAUSSIAN_MIXTURE:
            return GaussianMixture(n_components=params['n_components'], init_params='k-means++')
        elif conformity_str == Label.BAYESIAN_MIXTURE:
            return BayesianGaussianMixture(n_components=params['n_components'], init_params='k-means++')

        # Ensemble Models
        elif conformity_str == Label.IF:
            return IsolationForest(n_estimators=params['n_estimators'])

        # Outlier Models
        elif conformity_str == Label.OSVM:
            return OneClassSVM(nu=params['nu'])
        elif conformity_str == Label.SGD:
            return SGDOneClassSVM(nu=params['nu'])
        elif conformity_str == Label.ENVELOPE:
            return EllipticEnvelope(contamination=params['nu'])

        # Neighbor Models
        elif conformity_str == Label.LOF:
            return LocalOutlierFactor(n_neighbors=params['n_neighbors'], metric=params['metric'])

    @staticmethod
    def fit(conformity_str, users_pref_dist_df, users_preferences_instance):
        """
        It trains and find the clusters in the data.
        """
        # Train
        if conformity_str != Label.FCM:
            users_preferences_instance = users_preferences_instance.fit(
                X=users_pref_dist_df
            )
        else:
            users_preferences_instance.fit(
                X=users_pref_dist_df.to_numpy()
            )

        if conformity_str == Label.KMEANS or conformity_str == Label.BISECTING or \
                conformity_str == Label.GAUSSIAN_MIXTURE or conformity_str == Label.BAYESIAN_MIXTURE:
            return users_preferences_instance.predict(
                X=users_pref_dist_df
            )
        elif conformity_str == Label.AGGLOMERATIVE or conformity_str == Label.IF or \
                conformity_str == Label.BIRCH or conformity_str == Label.OPTICS or \
                conformity_str == Label.SPECTRAL or conformity_str == Label.DBSCAN or conformity_str == Label.LOF or \
                conformity_str == Label.SGD or conformity_str == Label.OSVM or conformity_str == Label.ENVELOPE:
            return users_preferences_instance.fit_predict(
                X=users_pref_dist_df
            )
        elif conformity_str == Label.FCM:
            return users_preferences_instance.predict(
                X=users_pref_dist_df.to_numpy()
            )

    @staticmethod
    def search(params, conformity_str, users_distribution_list):
        """
        It inits the search for the value.
        """
        silhouette_list = []

        for users_dist in users_distribution_list:
            users_preferences_instance = ManualConformityAlgorithmSearch.load_conformity_algorithm_instance(
                conformity_str=conformity_str, params=params
            )
            clusters = ManualConformityAlgorithmSearch.fit(
                conformity_str=conformity_str, users_pref_dist_df=users_dist,
                users_preferences_instance=users_preferences_instance
            )

            if len(set(clusters)) == 1:
                silhouette_list.append(0)
            else:
                silhouette_list.append(abs(silhouette_score(users_dist, clusters)))

        return {
            "silhouette": mean(silhouette_list) if len(silhouette_list) else 0,
            "params": params
        }

    def get_params_to_use(self, conformity_str: str):

        # Chosen the parameter structure
        if conformity_str in Label.CLUSTERING_LABEL_ALGORITHMS:
            selected_params_list = self.cluster_params
        elif conformity_str in Label.MIXTURE_LABEL_ALGORITHMS:
            selected_params_list = self.component_params
        elif conformity_str in Label.ENSEMBLE_LABEL_ALGORITHMS:
            selected_params_list = self.estimators_params
        elif conformity_str in Label.NEIGHBOR_LABEL_ALGORITHMS:
            selected_params_list = self.neighbor_params
        elif conformity_str in Label.OUTLIEAR_LABEL_ALGORITHMS:
            selected_params_list = self.outlier_params
        else:
            selected_params_list = self.param_grid

        params_list = list(ParameterGrid(selected_params_list))

        if self.n_inter < len(params_list):
            params_to_use = list(random.sample(params_list, self.n_inter))
        else:
            params_to_use = params_list

        return params_to_use

    def run(self, conformity_str: str):
        """
        Start to run the Manual Grid Search for Unsupervised Learning Clustering Algorithms.
        """

        params_to_use = self.get_params_to_use(conformity_str=conformity_str)

        for distribution in self.distribution_list:
            users_distribution_list = self.preparing_data(distribution=distribution)
            # Performing manual gridsearch
            payload = Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(ManualConformityAlgorithmSearch.search)(
                    params=params, conformity_str=conformity_str,
                    users_distribution_list=users_distribution_list
                ) for params in params_to_use
            )

            best_silhouette = 0
            best_param = None
            for params in payload:
                if abs(params["silhouette"]) > abs(best_silhouette):
                    best_silhouette = abs(params["silhouette"])
                    best_param = params

            # Saving the best
            SaveAndLoad.save_hyperparameters_conformity(
                best_params=best_param, dataset=self.dataset_name,
                cluster=conformity_str, distribution=distribution
            )
