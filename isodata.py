#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import help_funcs as hf
from numpy import linalg as LA
from scipy.cluster import vq


# from pyradar.utils import take_snapshot


def initialize_parameters(parameters=None):
    """Auxiliar function to set default values to all the parameters not
    given a value by the user.

    """
    parameters = {} if not parameters else parameters

    def safe_pull_value(parameters, key, default):
        return parameters.get(key, default)

    # number of clusters desired
    K = safe_pull_value(parameters, 'K', 5)

    # maximum number of iterations
    I = safe_pull_value(parameters, 'I', 100)

    # maximum of number of pairs of clusters which can be merged
    P = safe_pull_value(parameters, 'P', 4)

    # threshold value for  minimum number of samples in each cluster
    # (discarding clusters)
    THETA_N = safe_pull_value(parameters, 'THETA_N', 10)

    # threshold value for standard deviation (for split)
    THETA_S = safe_pull_value(parameters, 'THETA_S', 1)
    # threshold value for pairwise distances (for merge)
    THETA_C = safe_pull_value(parameters, 'THETA_C', 20)

    # percentage of change in clusters between each iteration
    # (to stop algorithm)
    THETA_O = 0.05

    # can use any of both fixed or random
    # number of starting clusters
    # k = np.random.randint(1, K)
    k = safe_pull_value(parameters, 'k', K)

    ret = locals()
    ret.pop('safe_pull_value')
    ret.pop('parameters')
    globals().update(ret)


def quit_low_change_in_clusters(centers, last_centers, iter):
    """Stop algorithm by low change in the clusters values between each
    iteration.

    :returns: True if should stop, otherwise False.

    """
    quit = False
    if centers.shape == last_centers.shape:
        thresholds = LA.norm((centers - last_centers) / (last_centers + 1))

        if np.all(thresholds <= THETA_O):  # percent of change in [0:1]
            quit = True
    #            print "Isodata(info): Stopped by low threshold at the centers."
    #            print "Iteration step: %s" % iter

    return quit


def merge_clusters(objects_classes, centers, clusters_list):
    """
    Merge by pair of clusters in 'below_threshold' to form new clusters.
    """
    pair_dists = compute_pairwise_distances(centers)

    first_p_elements = pair_dists[:P]

    below_threshold = [(c1, c2) for d, (c1, c2) in first_p_elements
                       if d < THETA_C]

    if below_threshold:
        k = centers.shape[0]
        count_per_cluster = np.zeros(k)
        to_add = []  # new clusters to add
        to_delete = set()  # clusters to delete

        for cluster in range(0, k):
            indices = np.where(objects_classes == clusters_list[cluster])[0]
            count_per_cluster[cluster] = indices.size

        # TODO: merge conflicts

        for c1, c2 in below_threshold:
            c1_count = float(count_per_cluster[c1]) + 1
            c2_count = float(count_per_cluster[c2])
            factor = 1.0 / (c1_count + c2_count)
            weight_c1 = c1_count * centers[c1]
            weight_c2 = c2_count * centers[c2]

            new_center = np.around(factor * (weight_c1 + weight_c2))

            to_add.append(new_center)
            to_delete.update([c1, c2])

        to_add = np.array(to_add)
        to_delete = np.array(list(to_delete))

        # delete old clusters and their indices from the availables array
        centers = np.delete(centers, to_delete, 0)
        clusters_list = np.delete(clusters_list, to_delete, 0)

        # generate new indices for the new clusters
        # starting from the max index 'to_add.size' times
        start = int(clusters_list.max()) if clusters_list.size != 0 else 0
        end = to_add.shape[0] + start
        centers = np.append(centers, to_add, 0)
        clusters_list = np.append(clusters_list, range(start, end))

        centers, clusters_list = sort_arrays_by_first(centers, clusters_list)

    return centers, clusters_list


def compute_pairwise_distances(centers):
    """
    Compute the pairwise distances 'pair_dists', between every two clusters
    centers and returns them sorted.
    Returns:
           - a list with tuples, where every tuple has in it's first coord the
             distance between to clusters, and in the second coord has a tuple,
             with the numbers of the clusters measured.
             Output example:
                [(d1,(cluster_1,cluster_2)),
                 (d2,(cluster_3,cluster_4)),
                 ...
                 (dn, (cluster_n,cluster_n+1))]
    """
    pair_dists = []
    size = centers.shape[0]

    for i in range(0, size):
        for j in range(0, size):
            if i > j:
                d = LA.norm(centers[i] - centers[j])
                pair_dists.append((d, (i, j)))

    # return it sorted on the first elem
    return sorted(pair_dists)


def split_clusters(img_flat, img_class_flat, centers, clusters_list):
    """
    Split clusters to form new clusters.
    """
    assert centers.size == clusters_list.size, \
        "ERROR: split() centers and clusters_list size are different"

    delta = 10
    k = centers.size
    count_per_cluster = np.zeros(k)
    stddev = np.array([])

    avg_dists_to_clusters = compute_avg_distance(img_flat, img_class_flat,
                                                 centers, clusters_list)
    d = compute_overall_distance(img_class_flat, avg_dists_to_clusters,
                                 clusters_list)

    # compute all the standard deviation of the clusters
    for cluster in range(0, k):
        indices = np.where(img_class_flat == clusters_list[cluster])[0]
        count_per_cluster[cluster] = indices.size
        value = ((img_flat[indices] - centers[cluster]) ** 2).sum()
        value /= count_per_cluster[cluster]
        value = np.sqrt(value)
        stddev = np.append(stddev, value)

    cluster = stddev.argmax()
    max_stddev = stddev[cluster]
    max_clusters_list = int(clusters_list.max())

    if max_stddev > THETA_S:
        if avg_dists_to_clusters[cluster] >= d:
            if count_per_cluster[cluster] > (2.0 * THETA_N):
                old_cluster = centers[cluster]
                new_cluster_1 = old_cluster + delta
                new_cluster_2 = old_cluster - delta

                centers = np.delete(centers, cluster)
                clusters_list = np.delete(clusters_list, cluster)

                centers = np.append(centers, [new_cluster_1, new_cluster_2])
                clusters_list = np.append(clusters_list, [max_clusters_list,
                                                          (max_clusters_list + 1)])

                centers, clusters_list = sort_arrays_by_first(centers,
                                                              clusters_list)

                assert centers.size == clusters_list.size, \
                    "ERROR: split() centers and clusters_list size are different"

    return centers, clusters_list


def compute_overall_distance(img_class_flat, avg_dists_to_clusters,
                             clusters_list):
    """
    Computes the overall distance of the samples from their respective cluster
    centers.
    """
    k = avg_dists_to_clusters.size
    total = img_class_flat.size
    count_per_cluster = np.zeros(k)

    for cluster in range(0, k):
        indices = np.where(img_class_flat == clusters_list[cluster])[0]
        count_per_cluster[cluster] = indices.size

    d = ((count_per_cluster / total) * avg_dists_to_clusters).sum()

    return d


def compute_avg_distance(img_flat, img_class_flat, centers, clusters_list):
    """
    Computes all the average distances to the center in each cluster.
    """
    k = centers.size
    avg_dists_to_clusters = np.array([])

    for cluster in range(0, k):
        indices = np.where(img_class_flat == clusters_list[cluster])[0]

        total_per_cluster = indices.size + 1
        sum_per_cluster = (np.abs(img_flat[indices] - centers[cluster])).sum()

        dj = (sum_per_cluster / float(total_per_cluster))

        avg_dists_to_clusters = np.append(avg_dists_to_clusters, dj)

    return avg_dists_to_clusters


def discard_clusters(objects_clusters, centers, clusters_list):
    """
    Discard clusters with fewer than THETA_N.
    """
    k = centers.shape[0]
    to_delete = []

    assert centers.shape[0] == clusters_list.size, \
        "ERROR: discard_cluster() centers and clusters_list size are different"

    for cluster in range(0, k):
        indices = np.where(objects_clusters == clusters_list[cluster])[0]
        total_per_cluster = indices.size
        if total_per_cluster <= THETA_N:
            to_delete.append(cluster)
    to_delete = np.array(to_delete)

    if to_delete.size:
        new_centers = np.delete(centers, to_delete, 0)
        new_clusters_list = np.delete(clusters_list, to_delete, 0)
    else:
        new_centers = centers
        new_clusters_list = clusters_list

    new_centers, new_clusters_list = sort_arrays_by_first(new_centers,
                                                          new_clusters_list)

    #        shape_bef = centers.shape[0]
    #        shape_aft = new_centers.shape[0]
    #        print "Isodata(info): Discarded %s clusters." % (shape_bef - shape_aft)

    #        if to_delete.size:
    #            print "Clusters discarded %s" % to_delete

    assert new_centers.shape[0] == new_clusters_list.size, \
        "ERROR: discard_cluster() centers and clusters_list size are different"

    return new_centers, new_clusters_list


def update_clusters(objects, objects_classes, centers, clusters_list):
    """ Update clusters. """
    k = centers.shape[0]
    new_centers = np.empty_like(centers)
    new_clusters_list = np.empty_like(clusters_list)

    assert centers.shape[0] == clusters_list.size, \
        "ERROR: update_clusters() centers and clusters_list size are different"

    for cluster in range(0, k):
        indices = np.where(objects_classes == clusters_list[cluster])[0]
        # get whole cluster
        cluster_values = objects[indices]
        # sum and count the values
        sum_per_cluster = np.sum(cluster_values, axis=0)
        total_per_cluster = (cluster_values.shape[0]) + 1
        # compute the new center of the cluster
        new_cluster = sum_per_cluster / total_per_cluster
        new_cluster = new_cluster.astype(cluster_values.dtype)

        new_centers[cluster] = new_cluster
        new_clusters_list[cluster] = cluster

    new_centers, new_clusters_list = sort_arrays_by_first(new_centers,
                                                          new_clusters_list)

    assert new_centers.shape[0] == new_clusters_list.shape[0], \
        "ERROR: update_clusters() centers and clusters_list size are different"

    return new_centers, new_clusters_list


def initial_clusters(objects, k, method="random", case_number=None):
    """
    Define initial clusters centers as startup.
    By default, the method is "linspace". Other method available is "random".
    :param objects: numpy array of feature vectors
    """
    methods_available = ["random", "preset"]
    assert method in methods_available, "ERROR: method %s is no valid." \
                                        "Methods available: %s" \
                                        % (method, methods_available)

    # множество X должно быть массивом векторов признаков, то есть размерности 2
    assert objects.ndim == 2, f"ERROR: array X of feature vectors has wrong dims: {objects.ndim}\n" \
                        "X should be of shape == ( N (vector count), M (vec size) )"

    print(f"Given {objects.shape[0]} objects with {objects.shape[1]} features")

    if method == "random":
        start, end = 0, objects.shape[0]
        indices = np.random.randint(start, end, k)
        centers = objects.take(indices, 0)
    elif method == "preset":
        cases = [
            [[103, 87,  97], [80,  85,  62]],
            [[39,  45,  57], [99, 113,  77], [105, 116, 108]]
        ]
        centers = np.array(cases[case_number])

    print(f"Chosen initial {k} centers: \n{centers}")

    return centers


def sort_arrays_by_first(centers, clusters_list):
    """
    Sort the array 'centers' and the with indices of the sorted centers
    order the array 'clusters_list'.
    Example: centers=[22, 33, 0, 11] and cluster_list=[7,6,5,4]
    returns  (array([ 0, 11, 22, 33]), array([5, 4, 7, 6]))
    """
    assert centers.shape[0] == clusters_list.size, \
        "ERROR: sort_arrays_by_first centers and clusters_list size are not equal"

    indices = np.argsort(list(map(lambda vec: LA.norm(vec), centers)))
    # test comment
    sorted_centers = centers[indices]
    sorted_clusters_list = clusters_list[indices]

    return sorted_centers, sorted_clusters_list


def isodata_classification(objects, parameters=None):
    """
    Classify a numpy array X of objects (arrays of features) x using Isodata algorithm.
    Parameters: a dictionary with the following keys.
            - X: an input array of arrays of features.
            - parameters: a dictionary with the initial values.
              If 'parameters' are not specified, the algorithm uses the default
              ones.
                  + number of clusters desired.
                    K = 15
                  + max number of iterations.
                    I = 100
                  + max number of pairs of clusters which can be ,erged.
                    P = 2
                  + threshold value for min number in each cluster.
                    THETA_N = 10
                  + threshold value for standard deviation (for split).
                    THETA_S = 0.1
                  + threshold value for pairwise distances (for merge).
                    THETA_C = 2
                  + threshold change in the clusters between each iter_number.
                    THETA_O = 0.01
        Note: if some(or all) parameters are nos providen, default values
              will be used.
    Returns:
            - img_class: a numpy array with the classification.
    """

    global K, I, P, THETA_N, THETA_S, THEHTA_C, THETA_O, k
    initialize_parameters(parameters)

    available_clusters = np.arange(k)  # number of clusters available

    print(f"Isodata(info): Starting algorithm with {k} classes")
    centers = initial_clusters(objects, k, "preset", 1)
    # centers = initial_clusters(objects, k, "random")

    for iter_number in range(I):
        # print "Isodata(info): Iteration:%s Num Clusters:%s" % (iter_number, k)
        last_centers = centers.copy()

        # assigning each of the samples to the closest cluster center
        objects_classes, dists = vq.vq(objects, centers)  # ВООБЩЕ не понятно, как работает эта строка

        centers, available_clusters = discard_clusters(objects_classes,
                                                       centers, available_clusters)
        centers, available_clusters = update_clusters(objects,
                                                      objects_classes,
                                                      centers, available_clusters)
        k = centers.size

        if k <= (K / 2.0):  # too few clusters => split clusters
            centers, available_clusters = split_clusters(objects, objects_classes,
                                                         centers, available_clusters)

        elif k > (K * 2.0):  # too many clusters => merge clusters
            centers, available_clusters = merge_clusters(objects_classes, centers,
                                                         available_clusters)
        else:  # nor split or merge are needed
            pass

        k = centers.size
        ###############################################################################
        if quit_low_change_in_clusters(centers, last_centers, iter_number):
            break

    #        take_snapshot(objects_classes.reshape(N, M), iteration_step=iter_number)
    ###############################################################################
    print((f"Isodata(info): Finished with {k} classes"))
    print(f"Isodata(info): Number of Iterations: {iter_number + 1}")

    return objects_classes
