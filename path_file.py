#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 10:42
# @Author  : YOUR-NAME
# @FileName: path_file.py
# @Software: PyCharm

miu_modify = 70


def basic_set_small_lfr():
    m_para_dict = {'alpha': 1, 'beta': 1, 'gamma': 1,
                   't': 0.5, 'node_num': 1000, 'reduce_dim': 100, 'layer_num': 5,
                   'epochs': 100, 'order': 3}
    m_gnd_path = "./data/LFR_labels.npy"
    return m_para_dict, m_gnd_path


def path_func_lfr():
    mu = miu_modify
    layer_nums = 5

    prefix = "E:/Dataset/homo_npy/" + str(mu) + "/layer"
    suffix = "/W.npy"
    my_data_file_paths = []
    for i in range(1, layer_nums + 1):
        my_data_file_paths.append(prefix + str(i) + suffix)

    my_para_dict, my_gnd_path = basic_set_small_lfr()

    output_prefix = "./output/"
    my_output_file_path = {
        'loss_path': output_prefix + str(mu) + '/loss.npy',
        'f_mat_path': output_prefix + str(mu) + '/f_mat.npy',
        'z_mat_path': output_prefix + str(mu) + '/z_mat.npy',
    }
    return my_data_file_paths, my_gnd_path, my_para_dict, my_output_file_path


def path_func_lfr_add_del(choice, percentage):
    mu = 70
    layer_nums = 5

    prefix = "E:/Dataset/homo_npy/" + str(mu) + "/layer"
    suffix1 = "/W_"
    suffix2 = ".npy"
    my_data_file_paths = []
    for i in range(1, layer_nums + 1):
        abs_path = prefix + str(i) + suffix1 + choice + '_' + percentage + suffix2
        my_data_file_paths.append(abs_path)

    my_para_dict, my_gnd_path = basic_set_small_lfr()

    output_prefix = "./output/" + choice + "_" + percentage
    my_output_file_path = {
        'loss_path': output_prefix + '/loss.npy',
        'f_mat_path': output_prefix + '/f_mat.npy',
        'z_mat_path': output_prefix + '/z_mat.npy',
    }
    return my_data_file_paths, my_gnd_path, my_para_dict, my_output_file_path


def path_func_big_lfr():
    mu = miu_modify
    layer_nums = 5

    prefix = "E:/Dataset/Big_LFR1/" + str(mu) + "/W"
    suffix = ".npy"
    my_data_file_paths = []
    for i in range(1, layer_nums + 1):
        my_data_file_paths.append(prefix + str(i) + suffix)

    my_para_dict = {'alpha': 1, 'beta': 1, 'gamma': 1,
                    't': 0.5, 'node_num': 10000, 'reduce_dim': 1000, 'layer_num': 5,
                    'epochs': 100, 'order': 3}
    my_gnd_path = "E:/Dataset/Big_LFR1/" + str(mu) + "/label.npy"

    output_prefix = "./output/Big_LFR/"
    my_output_file_path = {
        'loss_path': output_prefix + str(mu) + '/loss.npy',
        'f_mat_path': output_prefix + str(mu) + '/f_mat.npy',
        'z_mat_path': output_prefix + str(mu) + '/z_mat.npy',
    }
    return my_data_file_paths, my_gnd_path, my_para_dict, my_output_file_path
