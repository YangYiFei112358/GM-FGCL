#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 14:43
# @Author  : YOUR-NAME
# @FileName: work2_big_lfr.py
# @Software: PyCharm

import sys
from method.logger_writing import Logger
import numpy as np
from work2 import algorithm_func, save_to_excel
from path_file import path_func_big_lfr
from pandas import DataFrame

if __name__ == '__main__':
    # 纯LFR
    # data_file_paths, gnd_path, parameter_dict, output_file_path = path_func_lfr()
    data_file_paths, gnd_path, parameter_dict, output_file_path = path_func_big_lfr()

    sys.stdout = Logger()
    sys.stdout.show_version()
    # print("0.80 diag (z_mat = 0) run on para [0.1, 1, 10]")
    print("add 10 run on para [0.1, 1, 10]")
    parameter_list = [0.1, 1, 10]

    data = {'alpha': [], 'beta': [], 'gamma': [],
            'ACC': [], 'NMI': [], 'ARI': [], 'F-score': []}

    for a in parameter_list:
        for b in parameter_list:
            for c in parameter_list:
                parameter_dict['alpha'], parameter_dict['beta'], parameter_dict['gamma'] = a, b, c
                output_sentence_4 = "alpha = {}, beta = {}, gamma = {}".format(a, b, c)
                print(output_sentence_4)
                _, acc, nmi, ari, f1 = algorithm_func(data_file_paths, gnd_path, parameter_dict, output_file_path)
                data = save_to_excel(data, [a, b, c, acc, nmi, ari, f1])

    df = DataFrame(data)
    df.to_excel('mu80_z_mat_all_para.xlsx')
    print("All Done.")

    # sys.stdout = Logger()
    # sys.stdout.show_version()
    # print("0.8 run on para 1, 1, 1")
    # algorithm_func(data_file_paths, gnd_path, parameter_dict, output_file_path)
    # print("All Done.")

# 后续想加自己让他运行10次产出均值和方差的函数