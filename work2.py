#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 9:50
# @Author  : YOUR-NAME
# @FileName: work21.py
# @Software: PyCharm
import math
import sys
from method.logger_writing import Logger
import scipy.sparse as sp
from sklearn.cluster import KMeans
from method.optimize import init_bl_fl, update_b_list, update_f_list, update_f_mat, update_z_mat, compute_loss
import numpy as np
from method import metric_all
from pandas import DataFrame
from path_file import path_func_lfr, path_func_lfr_add_del
import os
from path_file import miu_modify

os.environ["OMP_NUM_THREADS"] = '1'


def read_file_func(file_paths):
    w_list = []
    laplacian_list = []

    for path in file_paths:
        wl_mat = np.load(path)
        w_list.append(wl_mat)
        dl_mat = np.diag(np.sum(wl_mat, axis=0))
        laplacian_mat = dl_mat - wl_mat
        laplacian_list.append(laplacian_mat)

    return w_list, laplacian_list


def read_labels_func(file_paths):
    labels = np.load(file_paths)
    return labels


def init_func(file_paths, para_list):
    reduce_dim = para_list['reduce_dim']
    layer_num = para_list['layer_num']

    w_list, laplacian_list = read_file_func(file_paths)
    b_list = []
    f_list = []
    z_mat = np.zeros_like(w_list[0])
    for wl_mat in w_list:
        b_tmp, f_tmp = init_bl_fl(wl_mat, reduce_dim)
        b_list.append(b_tmp)
        f_list.append(f_tmp)
        z_mat += wl_mat

    f_mat = np.zeros_like(f_list[0])
    for fl_mat in f_list:
        f_mat += fl_mat

    f_mat = f_mat / layer_num
    z_mat = z_mat / layer_num

    return w_list, laplacian_list, b_list, f_list, f_mat, z_mat


def algorithm_func(file_paths, labels_path, para_list, save_file_path):
    w_list, laplacian_list, b_list, f_list, f_mat, z_mat = init_func(file_paths, para_list)

    loss_last = 1e16
    loss_list = []
    epochs = para_list['epochs']
    order = para_list['order']
    error_radio = math.pow(10, -order)
    labels = read_labels_func(labels_path)
    for epoch in range(epochs):
        b_list = update_b_list(w_list, b_list, f_list, para_list)
        f_list = update_f_list(w_list, b_list, f_list, laplacian_list, f_mat, para_list)
        f_mat = update_f_mat(w_list, f_list, f_mat, z_mat, para_list)
        z_mat = update_z_mat(f_mat, z_mat)
        loss = compute_loss(w_list, b_list, f_list, laplacian_list, f_mat, z_mat, para_list)

        output_sentence_1 = "[epoch][{:>3}]: loss is {:.4f}".format(epoch, loss)
        print(output_sentence_1)
        loss_list.append(loss)

        if abs(loss - loss_last) / loss <= error_radio:
            print("The convergence condition meet!!!")
            break
        else:
            loss_last = loss
    cluster_number = len(np.unique(labels))
    c_mat = 0.5 * (np.fabs(z_mat) + np.fabs(z_mat.T))
    u, s, v = sp.linalg.svds(c_mat, k=cluster_number, which='LM')
    # Clustering
    kmeans = KMeans(n_clusters=cluster_number, random_state=7).fit(u)
    predict_labels = kmeans.predict(u)
    re = metric_all.ClusteringMetrics(predict_labels, labels)
    acc, nmi, ari, f1 = re.evaluation_cluster_model_from_label()
    output_sentence_2 = "Result: ACC={},NMI={},ARI={},F1={}.".format(acc, nmi, ari, f1)
    print(output_sentence_2)

    # loss_path = save_file_path['loss_path']
    # f_mat_path = save_file_path['f_mat_path']
    # z_mat_path = save_file_path['z_mat_path']
    # np.save(loss_path, loss_list)
    # np.save(f_mat_path, f_mat)
    # np.save(z_mat_path, z_mat)
    print("Done.")

    return predict_labels, acc, nmi, ari, f1


def save_to_excel(data_dict, p):
    data_dict['alpha'].append(p[0])
    data_dict['beta'].append(p[1])
    data_dict['gamma'].append(p[2])

    data_dict['ACC'].append(p[3])
    data_dict['NMI'].append(p[4])
    data_dict['ARI'].append(p[5])
    data_dict['F-score'].append(p[6])

    return data_dict


if __name__ == '__main__':
    # 纯LFR
    data_file_paths, gnd_path, parameter_dict, output_file_path = path_func_lfr()

    sys.stdout = Logger()
    sys.stdout.show_version()

    output_sentence = "0.{} run on para [0.1, 1, 10]".format(miu_modify)
    excel_path = './xlsx_all_para/mu{}_all_para_direct.xlsx'.format(miu_modify)

    print(output_sentence)
    parameter_list = [0.1, 1, 10]

    data = {'alpha': [], 'beta': [], 'gamma': [],
            'ACC': [], 'NMI': [], 'ARI': [], 'F-score': []}

    # for a in parameter_list:
    a = 10
    for c in parameter_list:
        # for c in parameter_list:
        parameter_dict['alpha'], parameter_dict['beta'], parameter_dict['gamma'] = a, 0, c
        output_sentence_4 = "alpha = {}, beta = {}, gamma = {}".format(a, 0, c)
        print(output_sentence_4)
        _, acc, nmi, ari, f1 = algorithm_func(data_file_paths, gnd_path, parameter_dict, output_file_path)
        data = save_to_excel(data, [a, 0, c, acc, nmi, ari, f1])

    df = DataFrame(data)
    df.to_excel(excel_path)
    print("All Done.")

    # sys.stdout = Logger()
    # sys.stdout.show_version()
    # print("0.8 run on para 1, 1, 1")
    # algorithm_func(data_file_paths, gnd_path, parameter_dict, output_file_path)
    # print("All Done.")

# 后续想加自己让他运行10次产出均值和方差的函数


# t_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     for tt in t_list: