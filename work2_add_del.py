#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 16:31
# @Author  : YOUR-NAME
# @FileName: work2_add_del.py
# @Software: PyCharm
import sys

from pandas import DataFrame

from method.logger_writing import Logger
from path_file import path_func_lfr_add_del
from work2 import algorithm_func, save_to_excel

if __name__ == '__main__':
    # 纯LFR
    add_del_modify = "del"
    percentage_modify = "10"
    data_file_paths, gnd_path, parameter_dict, output_file_path = path_func_lfr_add_del(add_del_modify,
                                                                                        percentage_modify)

    sys.stdout = Logger()
    sys.stdout.show_version()

    output_sentence = "{} {} run on para [0.1, 1, 10]".format(add_del_modify, percentage_modify)
    excel_path = './xlsx_all_para/{}_{}_all_para.xlsx'.format(add_del_modify, percentage_modify)
    print(output_sentence)

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
    df.to_excel(excel_path)
    print("All Done.")

    # sys.stdout = Logger()
    # sys.stdout.show_version()
    # print("0.8 run on para 1, 1, 1")
    # algorithm_func(data_file_paths, gnd_path, parameter_dict, output_file_path)
    # print("All Done.")
