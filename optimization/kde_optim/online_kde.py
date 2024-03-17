#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   online_kde.py
@Time    :   2022/10/25 14:35:36
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

"""
tools for online kde construction
"""


import datetime
import itertools
import math
import pickle as pkl
import random
import time
from typing import Dict, List
import copy 

import kmeans1d
import numpy as np
import torch
import yaml
from KDEpy import BaseKDE, NaiveKDE, TreeKDE
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from shiqingTools.tools import Logger, count_time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from kde_factory import KdeFactory
from utils import read_array, read_config, save_array

logger = Logger()

if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    logger = Logger()
    config = read_config("config/configures.yml")
    online_kde_config = config["online_kde"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(online_kde_config["device_id"])

    kernel = online_kde_config["kernel"]
    bw = online_kde_config["bandwidth"]
    H = online_kde_config["H"]
    W = online_kde_config["W"]
    C = online_kde_config["C"]

    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d-%H-%M-%S')
    writer = SummaryWriter(f'./logs/{time_str}')

    activation_arrays = read_array(
        f"./process_data/cifar10_activation/features.3_labels_activation.pkl")
    iter = 0
    factory = KdeFactory()
    for h, w, c in tqdm(itertools.product(range(H), range(W), range(C))):
        pool_points = activation_arrays[0, :, h, w, c].tolist()
        pool_weights = [1]*len(pool_points)
        cluster_kde_approximation = factory.get_kde_approximation(
            "cluster")(copy.deepcopy(pool_points),copy.deepcopy(pool_weights))
        optim_kde_approximation = factory.get_kde_approximation(
            "sgd")(copy.deepcopy(pool_points),copy.deepcopy(pool_weights))
        # online 更新9次(因为共有10个label)
        cluster_time = 0
        optim_time = 0
        for i in range(1, 10):
            append_points = activation_arrays[i, :, h, w, c].tolist()
            append_weights = [1]*len(append_points)

            t1 = time.time()
            cluster_kde_approximation.update_positions(append_points,append_weights)
            t2 = time.time()
            optim_kde_approximation.update_positions(append_points,append_weights)
            t3 = time.time()

            cluster_time += t2-t1
            optim_time += t3-t2
            

        all_points = activation_arrays[:, :, h, w, c].reshape(1, -1)[0]
        all_weights = [1]*len(all_points)
        base_kde_approximation = factory.get_kde_approximation(
            "base")(all_points,all_weights)
        base_kde_approximation.build_kde()
        x_pdf = base_kde_approximation.evaluate(all_points)

        y_pdf_cluster = cluster_kde_approximation.evaluate(all_points)
        y_pdf_optim = optim_kde_approximation.evaluate(all_points)
        # 计算总样本点得到的分布与近似样本点得到的分布的JS散度
        js_km_cluster = jensenshannon(x_pdf, y_pdf_cluster)
        js_km_optim = jensenshannon(x_pdf, y_pdf_optim)


        logger.info(f"js cluster:{js_km_cluster}")
        logger.info(f"js optim:{js_km_optim}")

        writer.add_scalar("js_cluster", js_km_cluster, global_step=iter)
        writer.add_scalar("js_optim", js_km_optim, global_step=iter)
        writer.add_scalar("cluster_update_time",cluster_time, global_step=iter)
        writer.add_scalar("optim_update_time",optim_time, global_step=iter)
        iter += 1
