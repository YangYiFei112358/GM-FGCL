#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/6 15:12
# @Author  : YOUR_NAME
# @FileName: optimize_01.py
# @Software: PyCharm

import numpy as np


def init_bl_fl(adj, k):
    # this method aims to get init of B, F
    [u_mat, s_mat, v_mat] = np.linalg.svd(adj)
    # u_mat: m*m
    # s_mat: min(m,n)*1
    # v_mat: n*n
    u_mat = u_mat[:, :k]
    s_mat = np.diag(s_mat[:k])
    v_mat = v_mat.T
    v_mat = v_mat[:, :k]
    v_mat = v_mat.T
    b_mat = abs(u_mat.dot(np.sqrt(s_mat)))
    f_mat = abs(np.sqrt(s_mat).dot(v_mat))
    return b_mat, f_mat


def update_b_list(w_list, b_list, f_list, node_num, my_eps=1e-10):
    # B: node_num * reduce_dim
    row_num = node_num
    layer_num = len(w_list)
    for k in range(layer_num):
        numerators = np.dot(w_list[k], f_list[k].T)
        denominators = np.dot(np.dot(b_list[k], f_list[k]), f_list[k].T)
        b_list[k] = b_list[k] * numerators / np.maximum(denominators, my_eps)
        # normalization 列归一化
        # for i in range(col_num):
        #     norm_L2 = np.linalg.norm(B[:, i])
        #     if norm_L2 > 0:
        #         B[:, i] = B[:, i] / norm_L2

        # normalization 行归一化
        for i in range(row_num):
            norm_l2 = np.linalg.norm(b_list[k][i, :])
            if norm_l2 > 0:
                b_list[k][i, :] = b_list[k][i, :] / norm_l2
    return b_list


def update_f_list(w_list, b_list, f_list, laplacian_list, f_mat, para_list, my_eps=1e-10):
    layer_num = para_list['layer_num']
    for k in range(layer_num):
        wl_mat = w_list[k]
        bl_mat = b_list[k]
        fl_mat = f_list[k]
        laplacian_mat = laplacian_list[k]
        f_list[k] = update_fl(wl_mat, bl_mat, fl_mat, laplacian_mat, f_mat, para_list)
    return f_list


# 计算余弦相似度矩阵
def calculate_sim_mat(wl_mat, fl_mat, f_mat, para_list):
    t = para_list['t']
    node_num = para_list['node_num']
    # 计算余弦相似度
    # 这里是想计算向量的模的长度
    row_norms_l = 1 / np.linalg.norm(fl_mat, axis=0)
    diag_row_norms_l = np.diag(row_norms_l)
    # 这里也是想计算向量的模的长度
    row_norms = 1 / np.linalg.norm(f_mat, axis=0)
    diag_row_norms = np.diag(row_norms)
    # 这里计算余弦相似度
    sim_mat = (diag_row_norms_l @ fl_mat.T @ f_mat @ diag_row_norms) / t
    p_mat_l = np.ones((node_num, node_num)) - wl_mat - np.identity(node_num)
    # 计算点积
    sim_mat = sim_mat * p_mat_l
    # 计算点积的指数
    sim_mat = np.exp(sim_mat)
    # 需要进行求和
    sim_mat_l = (diag_row_norms_l @ fl_mat.T @ fl_mat @ diag_row_norms_l) / t
    sim_mat_l = sim_mat_l * p_mat_l
    sim_mat_l = np.exp(sim_mat_l)
    sim_mat_all = sim_mat + sim_mat_l

    sim_mat_all_row_sums = np.sum(sim_mat_all, axis=1)  # 行求和
    diag_k_mat = np.diag(1 / sim_mat_all_row_sums)

    return sim_mat, sim_mat_l, diag_k_mat


def normalize_mat(mat):
    col_num = mat.shape[1]
    for i in range(col_num):
        norm_l2 = np.linalg.norm(mat[:, i])
        if norm_l2 > 0:
            mat[:, i] = mat[:, i] / norm_l2
    return mat


def update_fl(wl_mat, bl_mat, fl_mat, laplacian_mat, f_mat, para_list, my_eps=1e-10):
    alpha = para_list['alpha']
    beta = para_list['beta']
    t = para_list['t']
    node_num = para_list['node_num']
    col_num = node_num

    col_sum = 1 / np.sum(f_mat, axis=0)  # 列求和
    lambda_mat = np.diag(col_sum)  # 对角阵
    col_l_sum = 1 / np.sum(fl_mat, axis=0)  # 列求和
    lambda_l_mat = np.diag(col_l_sum)  # 对角阵
    col_l_sum_pow3 = np.sum(fl_mat, axis=0) ** 3  # 列求和
    lambda_l_mat_pow3 = np.diag(col_l_sum_pow3)  # 对角阵
    all_one_mat = np.ones((node_num, node_num))

    sim_mat, sim_mat_l, diag_k_mat = calculate_sim_mat(wl_mat, fl_mat, f_mat, para_list)

    numerators = 2 * bl_mat.T @ bl_mat @ fl_mat + 2 * fl_mat @ laplacian_mat + 2 * alpha * fl_mat
    numerators += beta * (1 / t) * f_mat @ lambda_l_mat @ all_one_mat @ lambda_mat
    numerators += beta * (1 / t) * f_mat @ lambda_l_mat @ sim_mat @ lambda_mat @ diag_k_mat
    numerators += beta * (1 / t) * fl_mat @ lambda_l_mat @ sim_mat_l @ lambda_l_mat @ diag_k_mat

    temp1 = fl_mat.T @ f_mat @ lambda_mat @ all_one_mat
    diagonal_matrix1 = np.diagflat(np.diag(temp1))
    temp2 = fl_mat.T @ f_mat @ lambda_mat @ sim_mat
    diagonal_matrix2 = np.diagflat(np.diag(temp2))
    temp3 = fl_mat.T @ fl_mat @ lambda_l_mat @ sim_mat_l
    diagonal_matrix3 = np.diagflat(np.diag(temp3))

    denominators = 2 * alpha * fl_mat + 2 * bl_mat.T @ wl_mat
    denominators += beta * (1 / t) * fl_mat @ lambda_l_mat_pow3 @ diagonal_matrix1
    denominators += beta * (1 / t) * fl_mat @ lambda_l_mat_pow3 @ diagonal_matrix2 @ diag_k_mat
    denominators += beta * (1 / t) * fl_mat @ lambda_l_mat_pow3 @ diagonal_matrix3 @ diag_k_mat

    fl_mat = fl_mat * denominators / np.maximum(numerators, my_eps)
    return normalize_mat(fl_mat)


def update_f_mat(w_list, f_list, f_mat, z_mat, para_list, my_eps=1e-10):
    alpha = para_list['alpha']
    beta = para_list['beta']
    gamma = para_list['gamma']
    t = para_list['t']
    node_num = para_list['node_num']
    layer_num = para_list['layer_num']

    col_sum = 1 / np.sum(f_mat, axis=0)  # 列求和
    lambda_mat = np.diag(col_sum)  # 对角阵
    all_one_mat = np.ones((node_num, node_num))

    numerators = 4 * alpha * f_mat @ f_mat.T @ f_mat + 2 * f_mat + 2 * gamma * f_mat @ z_mat @ z_mat.T
    denominators = 2 * gamma * f_mat @ z_mat.T + 2 * gamma * f_mat @ z_mat

    for i in range(layer_num):
        wl_mat = w_list[i]
        fl_mat = f_list[i]

        col_l_sum = 1 / np.sum(fl_mat, axis=0)  # 列求和
        lambda_l_mat = np.diag(col_l_sum)  # 对角阵
        col_l_sum_pow3 = np.sum(f_mat, axis=0) ** 3  # 列求和
        lambda_l_mat_pow3 = np.diag(col_l_sum_pow3)  # 对角阵

        sim_mat, _, diag_k_mat = calculate_sim_mat(wl_mat, fl_mat, f_mat, para_list)

        temp1 = f_mat.T @ fl_mat @ lambda_l_mat @ all_one_mat
        diagonal_matrix1 = np.diagflat(np.diag(temp1))
        temp2 = f_mat.T @ fl_mat @ lambda_l_mat @ sim_mat
        diagonal_matrix2 = np.diagflat(np.diag(temp2))

        numerators += beta * (1 / t) * fl_mat @ lambda_mat @ all_one_mat @ lambda_l_mat
        numerators += beta * (1 / t) * fl_mat @ lambda_mat @ sim_mat @ lambda_l_mat @ diag_k_mat
        denominators += 4 * alpha * f_mat @ fl_mat.T @ fl_mat
        denominators += beta * (1 / t) * f_mat @ lambda_l_mat_pow3 @ diagonal_matrix1
        denominators += beta * (1 / t) * f_mat @ lambda_l_mat_pow3 @ diagonal_matrix2 @ diag_k_mat

    f_mat = f_mat * denominators / np.maximum(numerators, my_eps)
    return normalize_mat(f_mat)


def update_z_mat(f_mat, z_mat, my_eps=1e-10):
    # Z: node_num * node_num
    numerators = np.dot(f_mat.T, f_mat)
    denominators = np.dot(np.dot(f_mat.T, f_mat), z_mat)
    # z_mat = z_mat * numerators / np.maximum(denominators, my_eps)
    z_mat = numerators / np.maximum(denominators, my_eps)
    z_mat = z_mat / np.linalg.norm(z_mat, axis=0)
    return z_mat


def compute_loss(w_list, b_list, f_list, laplacian_list, f_mat, z_mat, para_list):
    alpha = para_list['alpha']
    beta = para_list['beta']
    gamma = para_list['gamma']

    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    loss4 = 0.0
    for wl_mat, bl_mat, fl_mat, laplacian_mat in zip(w_list, b_list, f_list, laplacian_list):
        temp = wl_mat - np.dot(bl_mat, fl_mat)
        loss1 += np.linalg.norm(temp) ** 2
        loss2 += np.trace(np.dot(fl_mat, np.dot(laplacian_mat, fl_mat.T)))
        loss3 += alpha * (np.linalg.norm(f_mat - fl_mat) ** 2)

        sim_mat, _, diag_k_mat = calculate_sim_mat(wl_mat, fl_mat, f_mat, para_list)
        diag_sim_mat = np.diag(sim_mat)
        k_mat = np.diag(diag_k_mat)
        result = -1 * np.sum(np.log(diag_sim_mat / k_mat))
        loss4 += beta * result

    loss5 = gamma * np.linalg.norm(f_mat - np.dot(f_mat, z_mat)) ** 2
    loss = (loss1 + loss2 + loss3 + loss4 + loss5)
    # print(loss1, loss2, loss3, loss4, loss5)
    return loss
