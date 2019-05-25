#! /usr/bin/python
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from math import log, inf
from scipy.spatial import distance

data_frame = pd.read_csv("./DataSets/base.csv")
x = data_frame.iloc[:, 0:2].values
y = data_frame.iloc[:, 2].values


def cohesion(clusters):
    values, counts = np.unique(clusters, return_counts=True)

    if values[0] == -1:
        values = np.delete(values, 0)

    if len(values) == 0:
        return inf

    sse = [0] * len(values)
    mse = 0.0

    for n_cluster in range(len(values)):
        for i in range(len(clusters)):
            if clusters[i] == n_cluster:
                for j in range(len(clusters)):
                    if clusters[j] == n_cluster and clusters[j] != -1:
                        sse[n_cluster] += np.power(distance.euclidean(x[i], x[j]), 2)
        mse = mse + sse[n_cluster] / counts[n_cluster]

    return mse / len(values)


def entropy(clusters):
    if len(clusters) <= 1:
        return 0

    clusters_values, clusters_counts = np.unique(clusters, return_counts=True)
    original_values, original_counts = np.unique(y, return_counts=True)

    if clusters_values[0] == -1:
        clusters_counts = np.delete(clusters_counts, 0)

    if len(clusters_counts) == 0:
        return inf

    hits = [[0 for _ in range(len(original_values))] for _ in range(len(clusters_counts))]
    entropy_for_cluster = [0] * len(clusters_counts)

    for i in range(len(clusters_counts)):
        for j in range(len(clusters)):  # Calculo de cada classe no cluster
            if clusters[j] == i:
                hits[i][y[j] - 1] += 1
        for k in range(len(original_values)):  # calculo da entropia para cada cluster
            if hits[i][k] != 0:
                entropy_for_cluster[i] = entropy_for_cluster[i] - (
                        (hits[i][k] / clusters_counts[i]) * log((hits[i][k] / clusters_counts[i]), 2))

    return np.mean(entropy_for_cluster)


def separability(clusters):
    values, counts = np.unique(clusters, return_counts=True)

    if values[0] == -1:
        values = np.delete(values, 0)

    if len(values) == 0:
        return -inf

    sse = [[[0 for _ in range(int(((len(values) * len(values)) - len(values)) / 2))] for _ in range(len(values))] for _
           in range(2)]
    mse = []

    for n_cluster in range(int(((len(values) * len(values)) - len(values)) / 2)):
        for i in range(len(clusters)):
            if clusters[i] == n_cluster:
                for j in range(len(clusters)):
                    if clusters[j] != n_cluster and clusters[j] != -1:
                        sse[0][clusters[j]][n_cluster] += np.power(distance.euclidean(x[i], x[j]), 2)
                        sse[1][clusters[j]][n_cluster] += 1

    for n_cluster in range(int(((len(values) * len(values)) - len(values)) / 2)):
        for i in range(len(values)):
            if sse[1][i][n_cluster] != 0:
                mse.append(sse[0][i][n_cluster] / sse[1][i][n_cluster])

    return np.mean(mse)


def silhouette(clusters):
    values, counts = np.unique(clusters, return_counts=True)

    if values[0] == -1:
        values = np.delete(values, 0)
        counts = np.delete(counts, 0)

    if len(values) == 0 or len(counts) == 0:
        return -inf

    sse_temp = 0.0
    sse_a = []
    sse_b = []
    mse = inf

    for n_cluster in range(len(values)):
        for i in range(len(clusters)):
            if clusters[i] == n_cluster:
                for j in range(len(clusters)):
                    if clusters[j] == n_cluster:
                        sse_temp += distance.euclidean(x[i], x[j])
                sse_a.append(sse_temp / counts[n_cluster])
                sse_temp = 0

    sse = [[[0 for _ in range(len(values))] for _ in range(len(values))] for _ in range(2)]

    for n_cluster in range(len(values)):
        for i in range(len(clusters)):
            if clusters[i] == n_cluster:
                for j in range(len(clusters)):
                    if clusters[j] != n_cluster and clusters[j] != -1:
                        sse[0][clusters[j]][n_cluster] += distance.euclidean(x[i], x[j])
                        sse[1][clusters[j]][n_cluster] += 1

                for k in range(len(values)):
                    if sse[1][k][n_cluster] != 0 and mse > sse[0][k][n_cluster] / sse[1][k][n_cluster]:
                        mse = sse[0][k][n_cluster] / sse[1][k][n_cluster]

                sse_b.append(mse)
                mse = inf

    coefficient_sil = []

    for i in range(len(sse_b)):
        coefficient_sil.append((sse_b[i] - sse_a[i]) / max(sse_b[i], sse_a[i]))

    return np.mean(coefficient_sil)


# Encontrando melhores parametros=======================================================================================

def fbp_db():
    print('DBScan Calculando melhor eps e min_samples')

    best_eps = [0.0] * 4
    best_min_samples = [0.0] * 4
    aux_cohesion = inf
    aux_entropy = inf
    aux_separability = -inf
    aux_silhouette = -inf

    for eps in range(10, 21):
        for min_samples in range(100, 270, 10):
            ClusterDBScan = DBSCAN(eps=eps, min_samples=min_samples)
            ClusterDBScan.fit(x)

            # print("Calculando com métrica de coesão. eps =", eps, "min_samples =", min_samples)
            aux_aux = cohesion(ClusterDBScan.fit_predict(x))
            if aux_aux < aux_cohesion:
                aux_cohesion = aux_aux
                best_eps[0] = eps
                best_min_samples[0] = min_samples

            # print("Calculando com métrica de entropia. eps =", eps, "min_samples =", min_samples)
            aux_aux = entropy(ClusterDBScan.fit_predict(x))
            if aux_aux < aux_entropy:
                aux_entropy = aux_aux
                best_eps[1] = eps
                best_min_samples[1] = min_samples

            # print("Calculando com métrica de separabilidade. eps =", eps, "min_samples =", min_samples)
            aux_aux = separability(ClusterDBScan.fit_predict(x))
            if aux_aux > aux_separability:
                aux_separability = aux_aux
                best_eps[2] = eps
                best_min_samples[2] = min_samples

            # print("Calculando com métrica de silhueta. eps =", eps, "min_samples =", min_samples)
            aux_aux = silhouette(ClusterDBScan.fit_predict(x))
            if aux_aux > aux_silhouette:
                aux_silhouette = aux_aux
                best_eps[3] = eps
                best_min_samples[3] = min_samples
            # print("=============================================================================")

    print("DBScan cohesion =", aux_cohesion, "eps = ", best_eps[0], "min_samples = ", best_min_samples[0])
    print("DBScan entropy =", aux_entropy, "eps = ", best_eps[1], "min_samples = ", best_min_samples[1])
    print("DBScan separability =", aux_separability, "eps = ", best_eps[2], "min_samples = ", best_min_samples[2])
    print("DBScan silhouette =", aux_silhouette, "eps = ", best_eps[3], "min_samples = ", best_min_samples[3])


def fbp_km():
    print('K-Means Calculando melhor n_clusters, init e max_iter')

    best_n_clusters = [0.0] * 4
    best_init = [''] * 4
    best_max_iter = [0.0] * 4
    aux_cohesion = inf
    aux_entropy = inf
    aux_separability = -inf
    aux_silhouette = -inf
    for i in range(1, 2):
        init = 'k-means++' if i == 0 else 'random'
        for n_clusters in range(1, 17):
            for max_iter in range(50, 650, 50):
                ClusterKMeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter)
                ClusterKMeans.fit(x)

                # print("Calculando com métrica de coesão. n_clusters =", n_clusters, "max_iter =", max_iter)
                aux_aux = cohesion(ClusterKMeans.fit_predict(x))
                if aux_aux < aux_cohesion:
                    aux_cohesion = aux_aux
                    best_n_clusters[0] = n_clusters
                    best_max_iter[0] = max_iter
                    best_init[0] = init

                # print("Calculando com métrica de entropia. n_clusters =", n_clusters, "max_iter =", max_iter)
                aux_aux = entropy(ClusterKMeans.fit_predict(x))
                if aux_aux < aux_entropy:
                    aux_entropy = aux_aux
                    best_n_clusters[1] = n_clusters
                    best_max_iter[1] = max_iter
                    best_init[1] = init

                # print("Calculando com métrica de separabilidade. n_clusters =", n_clusters, "max_iter =", max_iter)
                aux_aux = separability(ClusterKMeans.fit_predict(x))
                if aux_aux > aux_separability:
                    aux_separability = aux_aux
                    best_n_clusters[2] = n_clusters
                    best_max_iter[2] = max_iter
                    best_init[2] = init

                # print("Calculando com métrica de silhueta. n_clusters =", n_clusters, "max_iter =", max_iter)
                aux_aux = silhouette(ClusterKMeans.fit_predict(x))
                if aux_aux > aux_silhouette:
                    aux_silhouette = aux_aux
                    best_n_clusters[3] = n_clusters
                    best_max_iter[3] = max_iter
                    best_init[3] = init
                # print("=============================================================================")

    print("K-Means cohesion =", aux_cohesion, "n_clusters = ", best_n_clusters[0], "init = ", best_init[0],
          "max_iter = ", best_max_iter[0])
    print("K-Means entropy =", aux_entropy, "n_clusters = ", best_n_clusters[1], "init = ", best_init[1], "max_iter = ",
          best_max_iter[1])
    print("K-Means separability =", aux_separability, "n_clusters = ", best_n_clusters[2], "init = ", best_init[2],
          "max_iter = ", best_max_iter[2])
    print("K-Means silhouette =", aux_silhouette, "n_clusters = ", best_n_clusters[3], "init = ", best_init[3],
          "max_iter = ", best_max_iter[3])


def fbp_agnes():
    print('Agnes Calculando nada')

    aux_cohesion = inf
    aux_entropy = inf
    aux_separability = inf
    aux_silhouette = -inf

    ClusterAgnes = AgglomerativeClustering(linkage='complete')
    ClusterAgnes.fit(x)

    # print("Calculando com métrica de coesão.")
    aux_aux = cohesion(ClusterAgnes.fit_predict(x))
    if aux_aux < aux_cohesion:
        aux_cohesion = aux_aux

    # print("Calculando com métrica de entropia.")
    aux_aux = entropy(ClusterAgnes.fit_predict(x))
    if aux_aux < aux_entropy:
        aux_entropy = aux_aux

    # print("Calculando com métrica de separabilidade.")
    aux_aux = separability(ClusterAgnes.fit_predict(x))
    if aux_aux < aux_separability:
        aux_separability = aux_aux

    # print("Calculando com métrica de silhueta.")
    aux_aux = silhouette(ClusterAgnes.fit_predict(x))
    if aux_aux > aux_silhouette:
        aux_silhouette = aux_aux
    # print("=============================================================================")

    print("Agnes cohesion =", aux_cohesion)
    print("Agnes entropy =", aux_entropy)
    print("Agnes separability =", aux_separability)
    print("Agnes silhouette =", aux_silhouette)
