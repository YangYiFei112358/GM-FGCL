#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 16:56
# @Author  : YOUR-NAME
# @FileName: work2_cycle_10_times_add_del.py
# @Software: PyCharm
import sys
import numpy as np
from method.logger_writing import Logger
from path_file import path_func_lfr_add_del
from work2 import algorithm_func, save_to_excel
from work2_cycle_10_times import mean_and_std
from pandas import DataFrame


def cal_mean_std(value_list):
    list_mean = np.mean(np.array(value_list))
    list_std = np.std(np.array(value_list), ddof=1)
    return list_mean, list_std


def add_del_func(add_del_modify, percentage_modify):
    alpha = 0.1
    beta = 10
    gamma = 1

    times = 10

    data_file_paths, gnd_path, parameter_dict, output_file_path = path_func_lfr_add_del(add_del_modify,
                                                                                        percentage_modify)
    output_sentence_modify = "{} {} run 10 times on para [alpha, beta, gamma] = [{}, {}, {}]".format(add_del_modify,
                                                                                                     percentage_modify,
                                                                                                     alpha, beta, gamma)
    path = './xlsx_mean_and_std/'
    execl_path_name_modify = path + '{}_{}_10_times_all_para.xlsx'.format(add_del_modify, percentage_modify)
    print(output_sentence_modify)

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    print(data_file_paths[0])
    for i in range(times):
        parameter_dict['alpha'], parameter_dict['beta'], parameter_dict['gamma'] = alpha, beta, gamma
        _, acc, nmi, ari, f1 = algorithm_func(data_file_paths, gnd_path, parameter_dict, output_file_path)
        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)
        f1_list.append(f1)

    acc_mean, acc_std = cal_mean_std(acc_list)
    nmi_mean, nmi_std = cal_mean_std(nmi_list)
    ari_mean, ari_std = cal_mean_std(ari_list)
    f1_mean, f1_std = cal_mean_std(f1_list)

    output_sentence_format = 'ACC_mean=%f, ACC_std=%f, nmi_mean=%f, nmi_std=%f, f1_mean=%f, f1_std=%f, ari_mean=%f, ' \
                             'ari_std=%f' % (acc_mean, acc_std, nmi_mean, nmi_std, f1_mean, f1_std, ari_mean, ari_std)
    print(output_sentence_format)

    data = {'alpha': [], 'beta': [], 'gamma': [],
            'ACC': [], 'NMI': [], 'ARI': [], 'F-score': []}
    data = save_to_excel(data, [alpha, beta, gamma,
                                mean_and_std(acc_mean, acc_std), mean_and_std(nmi_mean, nmi_std),
                                mean_and_std(ari_mean, ari_std), mean_and_std(f1_mean, f1_std)])
    df = DataFrame(data)
    df.to_excel(execl_path_name_modify)
    print("All Done.")


if __name__ == '__main__':
    sys.stdout = Logger()
    sys.stdout.show_version()
    add_del_choice_list = ['add', 'del']
    percentage_choice_list = ['2', '4', '6', '8', '10']
    for ch1 in add_del_choice_list:
        for ch2 in percentage_choice_list:
            add_del_func(ch1, ch2)

    print('All Done.')
