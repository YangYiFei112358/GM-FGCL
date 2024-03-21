#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 15:37
# @Author  : YOUR-NAME
# @FileName: metric.py
# @Software: PyCharm

import numpy as np
import yaml
from path_file_real_world import path_real_world
from work2 import read_file_func


def cal_normalized_indicator_mat(pred_labels):
    num_classes = len(np.unique(pred_labels))
    indicator_matrix = np.eye(num_classes)[pred_labels]

    # normalization
    column_norms = np.linalg.norm(indicator_matrix, axis=0)
    normalized_matrix = indicator_matrix / column_norms
    return indicator_matrix, normalized_matrix


def compute_ass(w_mat, pred_labels):
    num_classes = len(np.unique(pred_labels))
    _, normalized_matrix = cal_normalized_indicator_mat(pred_labels)

    ass_array = np.diag(normalized_matrix.T @ w_mat @ normalized_matrix)
    ass_average = np.trace(normalized_matrix.T @ w_mat @ normalized_matrix) / num_classes
    return ass_array, ass_average


def compute_multi_layer_ass(w_list, pred_labels):
    layer_num = len(w_list)
    num_classes = len(np.unique(pred_labels))

    ass_mat = np.zeros((layer_num, num_classes))
    ass_sum_list = []

    for i in range(layer_num):
        w_mat = w_list[i]
        ass_list, ass_average = compute_ass(w_mat, pred_labels)
        ass_mat[i, :] = ass_list
        ass_sum_list.append(ass_average)

    return ass_mat, ass_sum_list


def compute_density(w_mat, pred_labels):
    indicator_matrix, normalized_matrix = cal_normalized_indicator_mat(pred_labels)
    nums_per_clusters = np.diag(indicator_matrix.T @ indicator_matrix)

    num_classes = len(np.unique(pred_labels))

    temp_array = np.diag(normalized_matrix.T @ w_mat @ normalized_matrix)
    density_array = temp_array.copy()

    for i in range(num_classes):
        if nums_per_clusters[i] != 1:
            density_array[i] = density_array[i] / (nums_per_clusters[i] - 1)

    return density_array


def compute_multi_layer_density(w_list, pred_labels):
    layer_num = len(w_list)
    num_classes = len(np.unique(pred_labels))

    density_mat = np.zeros((layer_num, num_classes))

    for i in range(layer_num):
        w_mat = w_list[i]
        density_array = compute_density(w_mat, pred_labels)
        density_mat[i, :] = density_array

    return density_mat


def generate_w_list(n, layer_num):
    adj_list = []
    for i in range(layer_num):
        matrix = np.random.randint(2, size=(n, n))

        # 使矩阵对称
        matrix = np.triu(matrix) + np.triu(matrix, k=1).T
        np.fill_diagonal(matrix, 0)
        print("Random symmetric matrix:")
        print(matrix)
        adj_list.append(matrix)
    return adj_list


if __name__ == '__main__':
    dataset_name_list = ['amazon', 'cancer', 'cellphone', 'dblp', 'p2p']

    dataset = 'p2p'
    f = open('../yaml/real_world_info.yaml')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    parameters = cfg["real_world_datasets"]

    data_file_paths, parameter_dict, output_file_path = path_real_world(dataset,
                                                                        parameters[dataset]['node_num'],
                                                                        parameters[dataset]['reduce_dim'],
                                                                        parameters[dataset]['layer_num'],
                                                                        parameters[dataset]['clusters_num'])
    w_list, _ = read_file_func(data_file_paths)

    # w_list = generate_w_list(5, 3)
    ass_array1, ass1 = compute_multi_layer_ass(w_list, np.load("../output/" + dataset + "/pred_labels.npy"))
    print(ass_array1, ass1)
    temp_array1 = compute_multi_layer_density(w_list, np.load("../output/" + dataset + "/pred_labels.npy"))
    print(temp_array1)
