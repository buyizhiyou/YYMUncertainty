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
import multiprocessing
import pickle as pkl
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from KDEpy import BaseKDE, NaiveKDE, TreeKDE
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from shiqingTools.tools import Logger, count_time
from sklearn.neighbors import KernelDensity
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from kde_factory import KdeFactory
from utils import callback_error
from utils2 import read_array, read_config


def kde_approximation(activation_arrays: np.array, factory: KdeFactory, h_lst: List[int], w_lst: List[int], c_lst: List[int], js_error: Dict[str, float]) -> None:

    # streaming 更新9次(因为共有10个label)
    for h, w, c in tqdm(itertools.product(h_lst, w_lst, c_lst)):
        pool_points = list(activation_arrays[0, :, h, w, c])
        cluster_kde_approximation = factory.get_kde_approximation(
            "cluster")(pool_points)
        mid_kde_approximation = factory.get_kde_approximation(
            "mid")(pool_points)
        optimize_kde_approximation = factory.get_kde_approximation(
            "optimize")(pool_points)
        for i in range(1, 10):
            append_points = activation_arrays[i, :, h, w, c].tolist()

            t1 = time.time()
            cluster_kde_approximation.update_points(append_points)
            t2 = time.time()
            mid_kde_approximation.update_points(append_points)
            t3 = time.time()
            optimize_kde_approximation.update_points(append_points)
            t4 = time.time()

            print(f"cluster elapsed {t2-t1}")
            print(f"mid elapse {t3-t2}")
            print(f"optim elapse {t4-t3}")
            
        all_points = activation_arrays[:, :, h, w, c].reshape(1, -1)[0]
        base_kde_approximation = factory.get_kde_approximation(
            "base")(all_points)
        base_kde_approximation.build_kde()
        x_pdf = base_kde_approximation.evaluate(all_points)

        y_pdf_cluster = cluster_kde_approximation.evaluate(all_points)
        y_pdf_mid = mid_kde_approximation.evaluate(all_points)
        y_pdf_optim = optimize_kde_approximation.evaluate(all_points)

        # 计算总样本点得到的分布与近似样本点得到的分布的JS散度
        js_km_cluster = jensenshannon(x_pdf, y_pdf_cluster)
        js_km_optim = jensenshannon(x_pdf, y_pdf_optim)
        js_km_mid = jensenshannon(x_pdf, y_pdf_mid)

        # Mock data
        # js_km_cluster = np.random.randint(100)
        # js_km_optim = np.random.randint(100)
        # js_km_mid = np.random.randint(100)

        js_error[f"{h}_{w}_{c}"] = [js_km_cluster, js_km_optim, js_km_mid]


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    logger = Logger()

    config = read_config("config/configures.yml")
    online_kde_config = config["online_kde"]
    cpus = online_kde_config["cpu_nums"]
    kernel = online_kde_config["kernel"]
    bw = online_kde_config["bandwidth"]
    H = online_kde_config["H"]
    W = online_kde_config["W"]
    C = online_kde_config["C"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    torch.cuda.set_device(online_kde_config["device_id"])

    activation_arrays = read_array(
        f"./process_data/cifar10_activation/features.3_labels_activation.pkl")
    # activation_arrays = np.random.rand(10,20,20,20) #Mock data
    factory = KdeFactory()

    js_error = multiprocessing.Manager().dict()  
    pool = multiprocessing.Pool(processes=cpus) 
    h_lst = list(range(H))
    w_lst = list(range(W))
    c_lst = list(range(C))
    step = len(h_lst)//cpus
    for i in range(0, len(h_lst), step):
        sub_h_lst = h_lst[i:i+step]
        kde_approximation(activation_arrays, factory, sub_h_lst,w_lst,c_lst,js_error)
        # pool.apply_async(
        #     kde_approximation,
        #     (activation_arrays, factory, sub_h_lst, w_lst, c_lst, js_error),
        #     error_callback=callback_error,
        # )

    pool.close()
    pool.join()

    # save js error and plot
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d-%H-%M-%S')
    writer = SummaryWriter(f'./logs/{time_str}')
    for step, (js_km_cluster, js_km_optim, js_km_mid) in enumerate(js_error.values()):
        writer.add_scalar("js_cluster", js_km_cluster, global_step=step)
        writer.add_scalar("js_mid", js_km_mid, global_step=step)
        writer.add_scalar("js_optim", js_km_optim, global_step=step)
