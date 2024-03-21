#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/23 20:04
# @Author  : YOUR-NAME
# @FileName: generate_sorted_adj.py
# @Software: PyCharm

import numpy as np
import random


def generate_adj_npy(node_num, network_path, community_path):
    # network_path : network.dat, index from 1 to node_num
    # community_path : community.dat, index from 1 to node_num

    # 首先处理社区矩阵，使得具有相同标签的节点放在一块，方便形成块对角结构
    community_file = open(community_path, "rt")

    community_list = []
    for line in community_file:
        node_index, community_index = line.split()
        node_index = node_index - 1
        community_list.append(node_index, community_index)
    community_file.close()

    community_list.sort(key=lambda x: x[1])
    mapping_dict = {}
    for i in range(node_num):
        mapping_dict[i] = community_list[i][0]

    network_file = open(network_path, "rt")

    adj_mat_ori = np.zeros((node_num, node_num))
    # 根据原始建立的人工网络建立邻接矩阵
    for line in network_file:
        node_index_1st, node_index_2nd = line.split()
        node_index_1st = node_index_1st - 1
        node_index_2nd = node_index_2nd - 1
        adj_mat_ori[node_index_1st, node_index_2nd] = 1
    network_file.close()

    adj_mat = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            adj_mat[i, j] = adj_mat_ori[mapping_dict[i], mapping_dict[j]]

    return adj_mat, community_list


def save_adj_txt(adj_mat):
    node_num, _ = adj_mat.shape
    adj_txt_path = open('Adjacency.txt', 'w')

    for i in range(node_num):
        for j in range(node_num):
            if adj_mat[i, j] == 1:
                adj_txt_path.write(str(i) + ' ' + str(j))
    adj_txt_path.close()


def get_triple_community(community_list):
    """
    community_list = [
            [1, 1],
            [2, 2],
            [3, 1],
            [4, 2],
            [5, 2]
        ]
    """
    # 统计每个社团的大小
    community_sizes = {}
    for node in community_list:
        community = node[1]
        if community in community_sizes:
            community_sizes[community] += 1
        else:
            community_sizes[community] = 1

    # 遍历列表并为每个元素添加社团大小
    for node in community_list:
        node.append(community_sizes.get(node[1], 0))
    triple_community = community_list
    return triple_community


def shuffle_triple_community(triple_community):
    unique_values = list(set(item[1] for item in triple_community))
    random.shuffle(unique_values)
    # 创建映射字典
    mapping = {val: idx for idx, val in enumerate(unique_values)}

    # 使用映射字典将原始列表中的元素映射到新的值
    mapped_list = [[item[0], mapping[item[1]], item[2]] for item in triple_community]

    return mapped_list


def sort_triple_community(triple_community):
    sorted_triple_community = sorted(triple_community, key=lambda x: (x[2], x[1]))
    return sorted_triple_community


def generate_big_lfr(node_num, network_path, community_path):
    adj_mat, community_list = generate_adj_npy(node_num, network_path, community_path)
    triple_community = get_triple_community(community_list)
    sorted_triple_community = sort_triple_community(shuffle_triple_community(triple_community))

    map_dict = {i: sorted_triple_community[i][0] for i in range(node_num)}
    indices_i = [map_dict[i] for i in range(node_num)]
    indices_j = [map_dict[j] for j in range(node_num)]

    final_adj_mat = adj_mat[indices_i][:, indices_j]

    return final_adj_mat

