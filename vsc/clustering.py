import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


class VerbSenseClustering:
    def __init__(self, n_seeds, covariance_type):
        self.n_seeds = n_seeds
        self.covariance_type = covariance_type

    def _decide_seed(self, vec_array, n):
        if self.n_seeds == 1:
            best_seed = 0
        else:
            best_score, best_seed = -float("inf"), -1
            for seed in range(self.n_seeds):
                gmm = self.gmm_fit(vec_array, n, seed)
                score = gmm.score(vec_array)
                if score >= best_score:
                    best_score = score
                    best_seed = seed
        return best_seed

    def run_clustering(self, vec_array, n):
        seed = self._decide_seed(vec_array, n)
        gmm = self.fit_gmm(vec_array, n, seed)
        return gmm.predict(vec_array)

    def run_all_in_one_cluster(self, vec_array):
        return np.zeros(len(vec_array)).astype(int)

    def run_clustering_abic(self, vec_array, n_clusters):
        outputs = []
        for n in range(1, n_clusters + 1):
            seed = self._decide_seed(vec_array, n)
            gmm = self.fit_gmm(vec_array, n, seed)
            outputs.append(
                {
                    "n_clusters": n,
                    "first": -2 * gmm.score(vec_array) * vec_array.shape[0],
                    "second": gmm._n_parameters() * np.log(vec_array.shape[0]),
                }
            )
        return pd.DataFrame(outputs)

    def fit_gmm(self, vec_array, n, seed):
        try:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=self.covariance_type,
                random_state=seed,
            )
            gmm.fit(vec_array)
        except:
            gmm = GaussianMixture(
                n_components=1,
                covariance_type=self.covariance_type,
                random_state=seed,
            )
            gmm.fit(vec_array)
        return gmm
