import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

from vsc.scoring import eval_confusion_matrix


class VerbSenseClustering:
    def __init__(self, n_seeds, covariance_type):
        self.n_seeds = n_seeds
        self.covariance_type = covariance_type

    def _repeat_clustering(self, vec_array, n_components):
        max_score, max_seed = -float("inf"), -1
        for s in range(self.n_seeds):
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=self.covariance_type,
                random_state=s,
            )
            gmm.fit(vec_array)
            score = gmm.score(vec_array)
            if score >= max_score:
                max_score = score
                max_seed = s
        return max_seed

    def run_clustering(self, vec_array, n):
        seed = 0 if self.n_seeds == 1 else self._repeat_clustering(vec_array, n)
        gmm = GaussianMixture(
            n_components=n,
            covariance_type=self.covariance_type,
            random_state=seed,
        )
        gmm.fit(vec_array)
        return gmm.predict(vec_array)

    def run_all_in_one_cluster(self, vec_array):
        return np.zeros(len(vec_array)).astype(int)

    def calc_matching_scores(self, true_array, pred_array):
        df_matrix = pd.DataFrame(
            index=sorted(set(true_array)), columns=sorted(set(pred_array))
        ).fillna(0)
        for t, p in zip(true_array, pred_array):
            df_matrix.loc[t, p] += 1
        lsa_index, lsa_columns = linear_sum_assignment(df_matrix.values, maximize=True)

        true2pred = {
            k: v
            for k, v in zip(df_matrix.index[lsa_index], df_matrix.columns[lsa_columns])
        }
        count = 0
        for k, v in true2pred.items():
            count += df_matrix.loc[k, v]
        n_ex = df_matrix.sum().sum()
        score = count / n_ex
        return score

    def run_clustering_abic(self, vec_array, n_clusters):
        output_list = []
        for n in range(1, n_clusters + 1):
            seed = 0 if self.n_seeds == 1 else self._repeat_clustering(vec_array, n)
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=self.covariance_type,
                random_state=seed,
            )
            gmm.fit(vec_array)

            output_list.append(
                {
                    "n_clusters": n,
                    "first": -2 * gmm.score(vec_array) * vec_array.shape[0],
                    "second": gmm._n_parameters() * np.log(vec_array.shape[0]),
                }
            )
        return pd.DataFrame(output_list)

    def aggregate_abic(self, df, c, max_n_frames, max_n_clusters):
        df["bic"] = df["first"] + df["second"] * c

        df_cm = pd.DataFrame(
            index=range(1, max_n_frames + 1), columns=range(1, max_n_clusters + 1)
        ).fillna(0)

        output_list = []
        for verb in sorted(set(df["verb"])):
            df_verb = df[df["verb"] == verb]
            df_min = df_verb[df_verb["bic"] == df_verb["bic"].min()]
            n_frames = df_min["n_frames"].values[0]
            n_clusters = df_min["n_clusters"].values[0]
            df_cm.loc[n_frames, n_clusters] += 1

            output_list.append(
                {"verb": verb, "n_frames": n_frames, "n_clusters": n_clusters}
            )
        return df_cm, pd.DataFrame(output_list)

    def scoring(self, df):
        return eval_confusion_matrix(df)
