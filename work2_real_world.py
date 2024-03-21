#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 20:17
# @Author  : YOUR-NAME
# @FileName: work2_real_world.py
# @Software: PyCharm

import math
import os
import sys
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
import yaml
from method.logger_writing import Logger
from method.optimize_realworld import update_b_list, update_f_mat, update_f_list, update_z_mat, compute_loss
from path_file_real_world import path_real_world
from metric.metric import compute_multi_layer_ass, compute_multi_layer_density
from work2 import init_func


def algorithm_func_no_labels(file_paths, para_list, save_file_path):
    w_list, laplacian_list, b_list, f_list, f_mat, z_mat = init_func(file_paths, para_list)

    loss_last = 1e16
    loss_list = []
    epochs = para_list['epochs']
    order = para_list['order']
    error_radio = math.pow(10, -order)
    for epoch in range(epochs):
        b_list = update_b_list(w_list, b_list, f_list, para_list)
        f_list = update_f_list(w_list, b_list, f_list, laplacian_list, f_mat, para_list)
        f_mat = update_f_mat(w_list, f_list, f_mat, z_mat, para_list)
        z_mat = update_z_mat(f_mat, z_mat)
        loss = compute_loss(w_list, b_list, f_list, laplacian_list, f_mat, z_mat, para_list)

        output_sentence_1 = "[epoch][{:>3}]: loss is {:.4f}".format(epoch, loss)
        print(output_sentence_1)
        loss_list.append(loss)

        if (abs(abs(loss - loss_last) / abs(loss))) <= error_radio:
            print("The convergence condition meet!!!")
            break
        else:
            loss_last = loss
    cluster_number = para_list['clusters_num']
    c_mat = 0.5 * (np.fabs(z_mat) + np.fabs(z_mat.T))
    u, s, v = sp.linalg.svds(c_mat, k=cluster_number, which='LM')
    # Clustering
    kmeans = KMeans(n_clusters=cluster_number, random_state=7).fit(u)
    predict_labels = kmeans.predict(u)

    loss_path = save_file_path['loss_path']
    f_mat_path = save_file_path['f_mat_path']
    z_mat_path = save_file_path['z_mat_path']
    np.save(loss_path, loss_list)
    np.save(f_mat_path, f_mat)
    np.save(z_mat_path, z_mat)

    ass_mat_path = save_file_path['ass_mat_path']
    ass_list_path = save_file_path['ass_list_path']
    ass_mat, ass_list = compute_multi_layer_ass(w_list, predict_labels)
    print(ass_mat, ass_list)
    np.save(ass_mat_path, ass_mat)
    np.save(ass_list_path, ass_list)

    density_mat_path = save_file_path['density_mat_path']
    density_mat = compute_multi_layer_density(w_list, predict_labels)
    print(density_mat)
    np.save(density_mat_path, density_mat)


if __name__ == '__main__':
    sys.stdout = Logger()
    sys.stdout.show_version()

    dataset_name_list = ['amazon', 'cancer', 'cellphone', 'dblp', 'p2p', 'cell_phone']

    # 指定 dataset
    dataset = 'p2p'
    print(dataset)

    directory = "./output_real_world/" + dataset + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 加载参数
    f = open('./yaml/real_world_info.yaml')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    parameters = cfg["real_world_datasets"]

    data_file_paths, parameter_dict, output_file_path = path_real_world(dataset,
                                                                        parameters[dataset]['node_num'],
                                                                        parameters[dataset]['reduce_dim'],
                                                                        parameters[dataset]['layer_num'],
                                                                        parameters[dataset]['clusters_num'])

    para_list = [0.01, 0.1, 1, 10, 100]
    a = 0.01
    b = 0.01
    for c in para_list:
        parameter_dict['alpha'], parameter_dict['beta'], parameter_dict['gamma'] = a, b, c
        output_sentence_4 = "alpha = {}, beta = {}, gamma = {}".format(a, b, c)
        print(output_sentence_4)
        algorithm_func_no_labels(data_file_paths, parameter_dict, output_file_path)
