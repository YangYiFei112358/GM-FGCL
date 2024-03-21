#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/23 20:36
# @Author  : YOUR-NAME
# @FileName: generate_big_lfr.py
# @Software: PyCharm

import random
import numpy as np

layer_list = ['80']

qian_str = "D:\\Dataset\\Big_LFR2\\"

network_list = []
community_list = []
save_list = []

for idx in layer_list:
    network_list.append(qian_str + str(idx) + "\\network.dat")
    community_list.append(qian_str + str(idx) + "\\community.dat")
    save_list.append(qian_str + str(idx) + "\\W.npy")

node = 10000
clus_num = 510


def takeSecond(elem):
    return elem[1]


def takeThird(elem):
    return elem[2]


for layer_num_rand_seed in range(1, 6):
    for network_str, community_str, save_str in zip(network_list, community_list, save_list):
        # 打开文件
        networkfile = open(network_str, "rt")
        # 创建一个n*n大小的矩阵，用作我们的邻接矩阵
        # 这个邻接矩阵是从原始的网络中读取的
        Wold = np.zeros((node, node))
        for line in networkfile:
            l = line.split()
            # 减1的操作是因为numpy从0开始计数
            Wold[int(l[0]) - 1, int(l[1]) - 1] = 1
        # print(np.sum(Wold))
        networkfile.close()

        communityfile = open(community_str, "rt")
        community = []
        for line in communityfile:
            l = line.split()
            # 社区的下标从1开始
            community.append((int(l[0]) - 1, int(l[1])))
        # 对社区按照编号进行排序
        community.sort(key=takeSecond)
        # print(community)
        communityfile.close()
        # 这里我们想统计一下每个社区第一个元素的下标，进一步统计每个社区的大小
        com_size = []
        for j in range(1, clus_num + 1):
            for i in range(node):
                if community[i][1] == j:
                    com_size.append(i)
                    break
        com_size.append(node)
        # print(com_size)
        # print(len(com_size))
        # 这里想统计每一个社区的大小，作为第3个元素放到社区变量中
        com_size_idx = []
        # 这里想统计每一个社区的大小
        size_every = []
        for i in range(clus_num):
            tmp = [com_size[i + 1] - com_size[i]] * (com_size[i + 1] - com_size[i])
            com_size_idx = com_size_idx + tmp
            size_every.append((com_size[i + 1] - com_size[i]))
        size_every.sort()
        # 这里产生我们的三元组社区标签 【idx， 原始社区标签， 社区大小】
        tri_com = []
        for i in range(node):
            temp = [community[i][0], community[i][1], com_size_idx[i]]
            tri_com.append(temp)

        random.seed(layer_num_rand_seed)
        random.shuffle(tri_com)
        # 先打乱，然后对第3个维度进行排序
        tri_com.sort(key=takeThird)
        tmp_tir = []
        temp = []
        idx1 = 0
        tri_com.append("last")
        # 在第3维度的情况下，对第2维度进行排序
        for tir_item in tri_com:
            if tir_item[2] == size_every[idx1]:
                temp.append(tir_item)
            else:
                temp.sort(key=takeSecond)
                tmp_tir = tmp_tir + temp
                idx1 = idx1 + 1
                temp.clear()
                temp.append(tir_item)
        tri_com = tmp_tir
        # print(len(tri_com))

        f_tri_com = open('layer' + str(layer_num_rand_seed) + '/community_triple.txt', 'w')
        idx = 0
        for c in community:
            f_tri_com.write(str(idx) + ' ' + str(tri_com[idx][1]) + ' ' + str(tri_com[idx][2]) + '\n')
            idx += 1
        f_tri_com.close()

        dic1 = {}
        for i in range(node):
            dic1[i] = tri_com[i][0]
        #print(dic1)

        W = np.zeros((node, node))
        for i in range(node):
            for j in range(node):
                W[i, j] = Wold[dic1[i], dic1[j]]

        f_adj = open('layer' + str(layer_num_rand_seed) + '/Adjacency.txt', 'w')
        for i in range(node):
            for j in range(node):
                if W[i, j] == 1:
                    f_adj.write(str(i) + ' ' + str(j) + '\n')
        f_adj.close()
        #print(W)
        f_community = open('layer' + str(layer_num_rand_seed) + '/Community.txt', 'w')

        # 这个变量的目的是产生新的社区的下标
        clus_total = []
        for i in range(clus_num):
            tmp1 = [i + 1] * size_every[i]
            clus_total = clus_total + tmp1
        idx = 0
        for c in clus_total:
            f_community.write(str(idx) + ' ' + str(c) + '\n')
            idx += 1
        f_community.close()

        f_degree = open('layer' + str(layer_num_rand_seed) + '/Degree.txt', 'w')
        degree_list = np.sum(W, axis=1)
        for i in range(node):
            f_degree.write(str(i) + ' ' + str(degree_list[i]) + '\n')
        f_degree.close()
