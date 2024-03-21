#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 20:46
# @Author  : YOUR-NAME
# @FileName: work2_cycle_10_times.py
# @Software: PyCharm

import sys
from method.logger_writing import Logger
import numpy as np
from work2 import algorithm_func, save_to_excel
from path_file import path_func_lfr
from pandas import DataFrame
from path_file import miu_modify


def mean_and_std(mean, std):
    return str(mean) + '±' + str(std)


if __name__ == '__main__':
    sys.stdout = Logger()
    sys.stdout.show_version()

    alpha = 0.1
    beta = 10
    gamma = 1

    times = 10

    output_sentence_modify = "0.{} run 10 times on para [alpha, beta, gamma] = [{}, {}, {}]".format(miu_modify, alpha,
                                                                                                    beta, gamma)
    execl_path_name_modify = 'mu_{}_10_times_all_para.xlsx'.format(miu_modify)
    print(output_sentence_modify)

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    data_file_paths, gnd_path, parameter_dict, output_file_path = path_func_lfr()
    print(data_file_paths[0])
    for i in range(times):
        parameter_dict['alpha'], parameter_dict['beta'], parameter_dict['gamma'] = alpha, beta, gamma
        _, acc, nmi, ari, f1 = algorithm_func(data_file_paths, gnd_path, parameter_dict, output_file_path)
        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)
        f1_list.append(f1)

    acc_mean = np.mean(np.array(acc_list))
    acc_std = np.std(np.array(acc_list), ddof=1)
    f1_mean = np.mean(np.array(f1_list))
    f1_std = np.std(np.array(f1_list), ddof=1)
    nmi_mean = np.mean(np.array(nmi_list))
    nmi_std = np.std(np.array(nmi_list), ddof=1)
    ari_mean = np.mean(np.array(ari_list))
    ari_std = np.std(np.array(ari_list), ddof=1)
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
