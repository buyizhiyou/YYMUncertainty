#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   kde_factory.py
@Time    :   2022/12/06 10:26:23
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''
import datetime
import math
import pickle as pkl
import random
import time
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple
import os

import kmeans1d
import numpy as np
import torch
from KDEpy import BaseKDE, NaiveKDE, TreeKDE
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.spatial.distance import jensenshannon
from scipy.stats import rv_discrete
import scipy.io as sio
from shiqingTools.tools import Logger, count_time
from sklearn.neighbors import KernelDensity
from torch import autograd, nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from cvxopt import solvers
from cvxopt import matrix

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.set_printoptions(precision=8)  # 打印tensor显示8位小数


class KdeApproximation():
    def __init__(self, pool_positions: List[float], pool_weights: List[float], sample_num: int = 10000, pool_size: int = 100) -> None:
        self.pool_positions = pool_positions
        self.pool_weights = pool_weights
        self.kde = None
        self.sample_num = sample_num
        self.pool_size = pool_size

    def update_positions(self, append_positions: List[float], append_weights: List[int]) -> None:

        raise NotImplementedError("Not implement update_positions!")

    def build_kde(self, positions: List[float] = None, positions_w: List[float] = None, kernel: str = "gaussian", bw: int = 0.01) -> None:
        if positions is None:
            positions = self.pool_positions

        positions = np.array(positions)

        self.kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(
            positions.reshape(-1, 1), y=None, sample_weight=positions_w)
        '''
        bandwidthfloat or {“scott”, “silverman”}, default=1.0
        algorithm{‘kd_tree’, ‘ball_tree’, ‘auto’}, default=’auto’
        kernel{‘gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’}, default=’gaussian’
        '''

    def evaluate(self, positions: List[float]) -> List[float]:
        if self.kde is None:
            self.build_kde(self.pool_positions, self.pool_weights)

        samples = np.array(positions)
        pdf = np.exp(self.kde.score_samples(samples.reshape(-1, 1)))

        return pdf

    def sample(self, n_samples: int = 10000) -> List[float]:

        samples = self.kde.sample(n_samples)

        return samples

    def cluster(self, positions: List[float], weights: List[float], pool_size: int = 100) -> Tuple[List[float], List[float]]:
        """使用kmeans聚类得到近似的positions

        Args:
            positions (List[float]): pool里面所有positions位置
            pool_size (int, optional): 聚类数. Defaults to 100.

        Returns:
            Tuple[List[float], List[float]]: [优化后的positions位置，优化后的positions权重]
        """

        all_positions = [item for n, position in zip(
            weights, positions) for item in [position]*(int(n))]
        clusters, centriods = kmeans1d.cluster(all_positions, pool_size)
        # TODO:这里需要实现加权kmeans,优化上述代码

        positions_center = np.asarray(centriods)
        positions_w = np.asarray(
            [np.sum(np.asarray(clusters) == i) for i in range(pool_size)])
        # positions_w = positions_w/positions_w.sum()  # 归一化 这里归一化对结果的影响？？？

        return positions_center, positions_w


class ClusterKdeApproximation(KdeApproximation):
    def __init__(self, pool_positions: List[float], pool_weights: List[float], kernel: str = "gaussian", bw: int = 0.01, pool_size: int = 100) -> None:
        super().__init__(pool_positions, pool_weights)
        self.kernel = kernel
        self.bw = bw
        self.pool_size = pool_size
        self.logger = Logger()

    def update_positions(self, append_positions: List[float], append_weights: List[int]) -> None:
        all_positions = self.pool_positions
        all_positions.extend(append_positions)
        all_weights = self.pool_weights
        all_weights.extend(append_weights)

        solved_positions, positions_weights = self.cluster(
            all_positions, all_weights, self.pool_size)

        self.pool_positions = list(solved_positions)
        self.pool_weights = list(positions_weights)


class LBFGSKdeApproximation(KdeApproximation):
    def __init__(self, pool_positions: List[float], pool_weights: List[float], kernel: str = "gaussian", bw: int = 0.01) -> None:
        super().__init__(pool_positions, pool_weights)
        self.kernel = kernel
        self.bw = bw
        self.pool_size = 100
        self.logger = Logger()

    def update_positions(self, append_positions: List[float], append_weights: List[int]) -> None:
        all_positions = self.pool_positions
        all_positions.extend(append_positions)
        all_weights = self.pool_weights
        all_weights.extend(append_weights)

        self.build_kde(all_positions, all_weights, self.kernel, self.bw)
        # 控制采样多少个samples
        samples = self.kde.sample(self.sample_num)
        # x_pdf = self.evaluate(samples)

        # solved_positions, positions_weights = self.cluster(
        #     all_positions, all_weights, self.pool_size)
        # self.build_kde(solved_positions, positions_weights,
        #                self.kernel, self.bw)  # 使用聚类得到的结果构建kde
        # y_pdf = self.evaluate(samples)
        # js_km = jensenshannon(x_pdf, y_pdf)
        # self.logger.info(f"cluster initialization js:{js_km}")

        solved_positions, positions_weights = self.optimize_pts_lbfgs(
            all_positions, all_weights, samples)
        # self.build_kde(solved_positions, positions_weights,
        #                self.kernel, self.bw)  # 使用优化得到的结果构建kde
        # y_pdf_lbfgs = self.evaluate(samples)
        # js_km_lbfgs = jensenshannon(x_pdf, y_pdf_lbfgs)
        # self.logger.info(f"lbfgs optim js:{js_km_lbfgs}")

        self.pool_positions = solved_positions
        self.pool_weights = positions_weights

    def optimize_pts_lbfgs(self, all_positions: List[float], all_weights: List[float], samples: List[float]) -> Tuple[List[float], List[float]]:
        """使用lbfgs优化

        Args:
            all_positions (List[float]): pool里面所有positions位置
            positions_weights (List[float]): positions的权重 
            samples (List[float]): 采样得到的一系列离散点

        Returns:
            Tuple[List[float], List[float]]: [优化后的positions位置，优化后的positions权重]
        """
        N = self.pool_size
        M = self.sample_num

        init_method = "cluster"
        if init_method == "cluster":
            init_positions, init_weights = self.cluster(
                all_positions, all_weights, N)
        elif init_method == "random_sample":  # TODO:待优化点的位置和权重应该以何种策略更新???
            init_positions, init_weights = zip(
                *random.sample(list(zip(all_positions, all_weights)), N))
            init_positions = np.array(init_positions)
            init_weights = np.array(init_weights)
        elif init_method == "gaussian":
            # 用高斯分布去初始化，或者其他初始化策略????
            pass
        elif init_method == "sort":
            # 先排序，然后均分区间，采样
            pass

        solved_positions = torch.Tensor(init_positions).reshape(1, N)
        positions_weights = torch.IntTensor(init_weights).reshape(1, N)
        solved_positions.requires_grad = True  # positions to be optimized
        # positions_weights.requires_grad = True  #关闭权重优化

        samples = torch.Tensor(samples).reshape(M, 1)
        all_positions = torch.Tensor(all_positions).repeat(M, 1)
        all_weights = torch.IntTensor(all_weights).repeat(M, 1)

        kde_true = ((1/(math.sqrt(2*math.pi)*self.bw))*torch.exp(-1 /
                                                                 (2*self.bw**2)*(samples-all_positions)**2)*all_weights).sum(1)

        optimizer = torch.optim.LBFGS([solved_positions], history_size=10,
                                      line_search_fn="strong_wolfe")
        mse = nn.MSELoss()
        optimizer.zero_grad()

        def closure():
            kde_pred = ((1/(math.sqrt(2*math.pi)*self.bw))*torch.exp(-1 /
                        (2*self.bw**2)*(samples-solved_positions)**2)*positions_weights).sum(1)
            loss = mse(kde_pred, kde_true)
            loss.backward()

            return loss

        optimizer.step(closure=closure)

        # TODO:为什么出现优化结果nan???
        if np.isnan(solved_positions[0, :].cpu().detach().numpy()).any():
            return list(init_positions), list(init_weights)
        else:
            return list(solved_positions[0, :].cpu().detach().numpy()), list(positions_weights[0, :].cpu().detach().numpy())


class MidKdeApproximation(KdeApproximation):
    '''
    sort and set middle point 
    '''

    def __init__(self, pool_positions: List[float], kernel: str = "gaussian", bw: int = 0.01, pool_size: int = 100) -> None:
        super().__init__(pool_positions)
        self.kernel = kernel
        self.bw = bw
        self.pool_size = pool_size

        self.logger = Logger()

    def update_positions(self, append_positions: List[float], append_weights: List[int]) -> None:
        all_positions = self.pool_positions
        all_positions.extend(append_positions)
        mid_positions, _ = self.get_mid(all_positions)

        self.pool_positions = mid_positions

    def get_mid(self, all_positions: List[float]) -> Tuple[List[float], List[float]]:
        positions_sorted = sorted(all_positions)
        if len(positions_sorted) % 2 == 1:
            positions_sorted.append(positions_sorted[-1])

        mid_positions = list(
            np.sum([positions_sorted[::2], positions_sorted[1::2]], 0)/2)

        return mid_positions, None


class BasicFWApproximation(KdeApproximation):
    """Implement Basic Frank-Wolfe Algorithm
    """

    def __init__(self, pool_positions: List[float], pool_weights: List[float], kernel: str = "gaussian", bw: int = 0.01) -> None:
        super().__init__(pool_positions, pool_weights)
        self.kernel = kernel
        self.bw = bw
        self.pool_size = 100
        self.logger = Logger()

    def update_positions(self, append_positions: List[float], append_weights: List[int]) -> None:
        all_positions = self.pool_positions
        all_positions.extend(append_positions)
        all_weights = self.pool_weights
        all_weights.extend(append_weights)

        self.build_kde(all_positions, all_weights)
       # 控制采样多少个samples
        samples = self.kde.sample(self.sample_num)

        solved_positions, positions_weights = self.optimize_solved_positions_bfw(
            all_positions, all_weights, samples)

        self.pool_positions = solved_positions
        self.pool_weights = positions_weights

    def optimize_solved_positions_bfw(self, all_positions: List[float], all_weights: List[int], samples: List[float]):
        """Stochstic Frank-Wolfe Algorithm

        Args:
            positions (List[float]): pool里面所有positions位置
            samples (List[float]): 从已知分布采样的一批点
        """

        N = self.pool_size
        M = self.sample_num

        self.build_kde(all_positions, all_weights)
        x_pdf = self.evaluate(samples)

        # 待优化的N个点的初始化,这种初始化方式使用采样，对比均匀采样???
        # solved_positions = np.array(random.sample(all_positions, N))
        # solved_positions = solved_positions.reshape([-1, 1]
        init_positions, init_weights = self.cluster(
            all_positions, all_weights, N)
        solved_positions = init_positions.reshape([-1, 1])
        positions_weights = init_weights
        positions_pmf = np.ones_like(solved_positions)

        self.build_kde(solved_positions, positions_weights,
                       self.kernel, self.bw)
        y_pdf_bfw = self.evaluate(samples)
        js_km_bfw = jensenshannon(x_pdf, y_pdf_bfw)
        self.logger.info(f"cluster initiation,bfw js:{js_km_bfw}")

        K = 10  # 迭代次数
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d-%H-%M-%S')
        writer = SummaryWriter(f'./logs/{time_str}')
        for k in range(K):
            err = math.inf
            w_k = 2/(k+2)
            # step1:Resolution of the subproblems
            solved_positions_ba = self.solve_subproblem(
                all_positions, all_weights, solved_positions, positions_weights, positions_pmf, samples)

            solved_positions = np.concatenate(
                [solved_positions, solved_positions_ba], axis=1)
            positions_pmf = np.concatenate(
                [(1-w_k)*positions_pmf, w_k*np.ones_like(solved_positions_ba)], axis=1)

            # evaluate
            # sample_positions from solved_positions
            sample_positions = []
            for row in range(solved_positions.shape[0]):
                dist = rv_discrete(
                    values=(solved_positions[row], positions_pmf[row]))
                # sample_positions.append(dist.rvs())
                sample_positions.append(dist.mean())

            self.build_kde(sample_positions, positions_weights,
                           self.kernel, self.bw)
            y_pdf_bfw = self.evaluate(samples)
            js_km_bfw = jensenshannon(x_pdf, y_pdf_bfw)
            self.logger.info(f"iteration {k},bfw js:{js_km_bfw}")

            writer.add_scalar("bfw_js_cluster", js_km_bfw, global_step=k)

        return sample_positions, positions_weights

    def solve_subproblem(self, all_positions: List[float], all_weights: List[int], solved_positions: List[float], positions_weights: List[float], positions_pmf: List[float], samples: List[float], multiVar: int = 0):
        N = self.pool_size
        solved_positions_new = np.zeros([N, 1])

        positions_weights_expand = np.expand_dims(positions_weights, axis=0)
        positions_weights_expand = np.expand_dims(
            positions_weights_expand, axis=2)
        all_weights_expand = np.expand_dims(all_weights, axis=0)
        positions_expand = np.expand_dims(
            solved_positions, axis=0)  # 为了广播运算，扩展维度
        y = (positions_weights_expand*np.exp(-1/(2*self.bw**2)*(samples.reshape(
            [-1, 1, 1])-positions_expand)**2)*positions_pmf).sum(axis=(1, 2))  # 利用期望求y_k
        kde_true = (all_weights_expand*1/N*np.exp(-1/(2*self.bw**2)
                    * (samples-all_positions)**2)).sum(1)

        sample_positions = []
        for row in range(solved_positions.shape[0]):
            dist = rv_discrete(
                values=(solved_positions[row], positions_pmf[row]))
            sample_positions.append(dist.rvs())
        if multiVar == 0:
            def obj_fun(x: float):  # 单个变量优化
                g_x = np.exp(-1/(2*self.bw**2)*(x-samples)**2)
                obj = 2*np.dot(y-kde_true, g_x)

                return obj

            for i, x in enumerate(solved_positions):  # 这里初始化值用什么？？？
                # 调用优化工具解一元优化问题  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
                # obj_fun = fun(positions_weights[i])
                solution = optimize.minimize(obj_fun, x, method='BFGS')
                solved_positions_new[i] = solution.x[0]
        else:
            def fun2(x: List[float]):  # 多变量优化
                g_x = np.exp(-1/(2*self.bw**2)*(x-samples)**2)
                obj = 2*np.dot(y-kde_true, g_x).sum()

                return obj

            solution = optimize.minimize(
                fun2, solved_positions, method='BFGS')  # 调用优化工具解一元优化问题
            solved_positions_new = solution.x

        return solved_positions_new


class SFWApproximation(KdeApproximation):
    """Implement Stochastic Frank-Wolfe Algorithm
    """

    def __init__(self, pool_positions: List[float], kernel: str = "gaussian", bw: int = 0.01) -> None:
        super().__init__(pool_positions)
        self.kernel = kernel
        self.bw = bw

        self.logger = Logger()

    def solve_subproblem(self, all_positions: List[float], solved_positions: List[float], samples: List[float], multiVar: int = 0):
        N = len(solved_positions)
        solved_positions_new = solved_positions
        y = 1/N*np.exp(-1/(2*self.bw**2)*(samples-solved_positions)**2).sum(1)
        kde_true = 1/N*np.exp(-1/(2*self.bw**2) *
                              (samples-all_positions)**2).sum(1)

        if multiVar == 0:
            def fun(x: float):  # 单个变量优化
                g_x = np.exp(-1/(2*self.bw**2)*(x-samples)**2)
                obj = 2*np.dot(y-kde_true, g_x)

                return obj

            for i, x in enumerate(solved_positions):
                # solution = optimize.minimize(fun, x, method='BFGS')  # 调用优化工具解一元优化问题  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
                solution = optimize.minimize(fun, x, method='CG')
                solved_positions_new[i] = solution.x[0]
        else:
            def fun2(x: List[float]):  # 多变量优化
                g_x = np.exp(-1/(2*self.bw**2)*(x-samples)**2)
                obj = 2*np.dot(y-kde_true, g_x).sum()

                return obj

            solution = optimize.minimize(
                fun2, solved_positions, method='BFGS')  # 调用优化工具解一元优化问题
            solved_positions_new = solution.x

        return solved_positions_new

    def objective(self, all_positions: List[float], solved_positions: List[float], samples: List[float]):
        N = len(solved_positions)
        kde_true = 1/N*np.exp(-1/(2*self.bw**2) *
                              (samples-all_positions)**2).sum(1)
        kde_pred = 1/N*np.exp(-1/(2*self.bw**2) *
                              (samples-solved_positions)**2).sum(1)

        mse = np.square(kde_true-kde_pred).sum()

        return mse

    def update_positions(self, append_positions: List[float], append_weights: List[int]) -> None:
        all_positions = self.pool_positions
        all_positions.extend(append_positions)

        self.build_kde(all_positions, None, self.kernel, self.bw)

       # 控制采样多少个samples
        x = self.kde.sample(self.sample_num)

        solved_positions, _ = self.optimize_solved_positions_sfw(
            all_positions, x)

        self.pool_positions = solved_positions

    def optimize_solved_positions_sfw(self, all_positions: List[float], samples: List[float]):
        """Stochstic Frank-Wolfe Algorithm

        Args:
            positions (List[float]): pool里面所有positions位置
            samples (List[float]): 从已知分布采样的一批点
        """
        # N = len(all_positions)//2 #1000
        N = 100
        M = self.sample_num

        x_pdf = self.evaluate(all_positions)

        # 待优化的N个点的初始化,这种初始化方式使用采样，对比均匀采样???
        solved_positions = random.sample(all_positions, N)

        self.build_kde(solved_positions, None, self.kernel, self.bw)
        y_pdf_sfw = self.evaluate(all_positions)
        js_km_sfw = jensenshannon(x_pdf, y_pdf_sfw)
        self.logger.info(f"initiation,sfw js:{js_km_sfw}")

        K = 100  # 迭代次数
        for k in range(K):
            err = math.inf
            # step1:Resolution of the subproblems
            solved_positions_ba = self.solve_subproblem(
                all_positions, solved_positions, samples)

            # step2:Update
            n = 100
            for j in range(n):  # 模拟伯努力分布采样次数
                solved_positions_hat = solved_positions
                for i in range(N):
                    if random.random() < 2/(k+2):  # w_k = 2/(k+2)
                        solved_positions_hat[i] = solved_positions_ba[i]

                err_new = self.objective(
                    all_positions, solved_positions_hat, samples)
                if err_new < err:
                    solved_positions_next = solved_positions_hat
                    err = err_new

            solved_positions = solved_positions_next

            # evaluate
            self.build_kde(solved_positions, None, self.kernel, self.bw)
            y_pdf_sfw = self.evaluate(all_positions)
            js_km_sfw = jensenshannon(x_pdf, y_pdf_sfw)
            self.logger.info(f"iteration {k},sfw js:{js_km_sfw}")

        return solved_positions, None


class SGDKdeApproximation(KdeApproximation):
    '''
    gradient descent
    '''

    def __init__(self, pool_positions: List[float], pool_weights: List[float], kernel: str = "gaussian", bw: int = 0.01) -> None:
        super().__init__(pool_positions, pool_weights)
        self.kernel = kernel
        self.bw = bw
        self.pool_size = 100
        self.sample_num = 10
        self.logger = Logger()

    def update_positions(self, append_positions: List[float], append_weights: List[float]) -> None:
        all_positions = self.pool_positions
        all_positions.extend(append_positions)
        all_weights = self.pool_weights
        all_weights.extend(append_weights)

        self.build_kde(all_positions, all_weights, self.kernel, self.bw)
        # 控制采样多少个samples
        samples = self.kde.sample(self.sample_num)
        samples = sorted(samples)

        solved_positions, positions_weights = self.optimize_pts_sgd(
            all_positions, all_weights, samples)

        self.pool_positions = solved_positions
        self.pool_weights = positions_weights

    def optimize_pts_sgd(self, all_positions: List[float], all_weights: List[float], samples: List[float]) -> tuple[List[float], List[float]]:
        N = self.pool_size
        M = self.sample_num
        batch_size = self.sample_num
        total = sum(all_weights)

        init_method = "cluster"
        # TODO:初始化方法
        if init_method == "cluster":
            init_positions, init_weights = self.cluster(
                all_positions, all_weights, N)
        elif init_method == "xavier":
            w = torch.empty(1, N)
            init_positions = torch.nn.init.xavier_normal_(w)
            init_weights = torch.nn.init.xavier_normal_(w)
        elif init_method == "mix":
            init_positions, init_weights = self.cluster(
                all_positions, all_weights, N)
            w = torch.empty(1, N)
            init_positions_xavier = torch.nn.init.xavier_normal_(w)

        x_pdf = self.evaluate(samples)
        self.build_kde(init_positions, init_weights,
                       self.kernel, self.bw)  # 使用聚类得到的结果构建kde
        y_pdf = self.evaluate(samples)
        js_km_cluster = jensenshannon(x_pdf, y_pdf)
        self.logger.info(f"cluster initialization js:{js_km_cluster}")

        solved_positions = torch.Tensor(init_positions).reshape(
            1, N).repeat(batch_size, 1).cuda()
        positions_weights = torch.IntTensor(init_weights).reshape(
            1, N).repeat(batch_size, 1).cuda()
        solved_positions.requires_grad = True  # positions to be optimized

        samples_gpu = torch.Tensor(samples).cuda().reshape(M, 1)
        all_positions = torch.Tensor(all_positions).cuda().repeat(M, 1)
        all_weights = torch.IntTensor(all_weights).cuda().repeat(M, 1)

        kde_true = ((1/(math.sqrt(2*math.pi)*self.bw))*torch.exp(-1 /
                                                                 (2*self.bw**2)*(samples_gpu-all_positions)**2)*(all_weights/total)).sum(1)

        epoches = 5000
        optimizer = torch.optim.Adam(
            [solved_positions], lr=0.00001)

        mse = nn.MSELoss()
        # TODO：试试kl散度作为loss
        kl = torch.nn.KLDivLoss(reduction='batchmean')
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d-%H-%M-%S')
        base_dir = os.path.dirname(__file__)
        writer = SummaryWriter(f'{base_dir}/logs/{time_str}')

        positions_data = positionsDataset(samples_gpu, kde_true)
        positions_loader = DataLoader(
            positions_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        for epoch in tqdm(range(epoches)):
            for train_x, train_y in positions_loader:
                kde_pred = ((1/(math.sqrt(2*math.pi)*self.bw))*torch.exp(-1 / (2*self.bw**2)
                            * (train_x-solved_positions)**2)*(positions_weights/total)).sum(1)

                mean_dist = 0.5*(train_y+kde_pred)
                loss = 0.5*kl(torch.log(mean_dist), train_y) + \
                    0.5*kl(torch.log(mean_dist), kde_pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            solved_positions_cpu, positions_weights_cpu = list(solved_positions[0, :].cpu(
            ).detach().numpy()), list(positions_weights[0, :].cpu().detach().numpy())
            self.build_kde(solved_positions_cpu, positions_weights_cpu,
                           self.kernel, self.bw)  # 使用优化得到的结果构建kde

            y_pdf_sgd = self.evaluate(samples)
            js_km_sgd = jensenshannon(x_pdf, y_pdf_sgd)

            if epoch % 100 == 0:
                writer.add_scalar("loss", loss, global_step=epoch)
                writer.add_scalar("sgd_js", js_km_sgd, global_step=epoch)

        self.build_kde(solved_positions_cpu, positions_weights_cpu,
                       self.kernel, self.bw)  # 使用优化得到的结果构建kde
        y_pdf_sgd = self.evaluate(samples)
        js_km_sgd = jensenshannon(x_pdf, y_pdf_sgd)
        self.logger.info(f"sgd optim js:{js_km_sgd}")

        with open(f"{base_dir}/logs/js.txt", "a") as f:
            f.write(f"{js_km_cluster},{js_km_sgd}\n")

        plt.plot(samples, y_pdf, color='r', linestyle=':', label='cluster')
        plt.plot(samples, y_pdf_sgd, color='g', linestyle='-', label='sgd')
        plt.legend()
        plt.savefig("samples.jpg")

        return solved_positions_cpu, positions_weights_cpu


class QuadKdeApproximation(KdeApproximation):
    '''
    Quadratic optimization problems
    '''

    def __init__(self, pool_positions: List[float], pool_weights: List[float], kernel: str = "epanechnikov", bw: int = 0.01) -> None:

        super().__init__(pool_positions, pool_weights)
        self.kernel = kernel
        self.bw = bw
        self.pool_size = 100
        self.sample_num = 1000
        self.logger = Logger()

    def update_positions(self, append_positions: List[float], append_weights: List[float]) -> None:
        all_positions = self.pool_positions
        all_positions.extend(append_positions)
        all_weights = self.pool_weights
        all_weights.extend(append_weights)

        init_positions, init_weights = self.cluster(
                all_positions, all_weights, self.pool_size)

        self.build_kde(all_positions, all_weights, self.kernel, self.bw)
        x_pdf = self.evaluate(all_positions)
        self.build_kde(init_positions, init_weights,
                       self.kernel, self.bw)  # 使用聚类得到的结果构建kde
        y_pdf = self.evaluate(all_positions)
        js_km_cluster = jensenshannon(x_pdf, y_pdf)
        self.logger.info(f"cluster initialization js:{js_km_cluster}")#0.0001708098102390352

        self.kernel="gaussian"
        self.build_kde(all_positions, all_weights, self.kernel, self.bw)
        x_pdf = self.evaluate(all_positions)
        self.build_kde(init_positions, init_weights,
                       self.kernel, self.bw)  # 使用聚类得到的结果构建kde
        y_pdf = self.evaluate(all_positions)
        js_km_cluster = jensenshannon(x_pdf, y_pdf)
        self.logger.info(f"cluster initialization js:{js_km_cluster}")#0.0001708098102390352

        self.build_kde(all_positions, all_weights, self.kernel, self.bw)
        # 控制采样多少个samples
        samples = self.kde.sample(self.sample_num)
        samples = sorted(samples)

        solved_positions, positions_weights = self.optimize_pts_quad(
            all_positions, all_weights, samples)

        self.pool_positions = solved_positions
        self.pool_weights = positions_weights

    def optimize_pts_quad(self, all_positions: List[float], all_weights: List[float], samples: List[float]) -> tuple[List[float], List[float]]:
        N = self.pool_size
        M = self.sample_num
        batch_size = self.sample_num
        total = sum(all_weights)

        x_pdf = self.evaluate(samples)
        
        init_method = "cluster"
        # TODO:初始化方法
        if init_method == "cluster":
            init_positions, init_weights = self.cluster(
                all_positions, all_weights, N)


        samples = np.array(samples)  # [M,1]
        t = ((1/(math.sqrt(2*math.pi)*self.bw))*np.exp(-1 / (2*self.bw**2)
             * (samples-all_positions)**2)*all_weights).sum(1)
        t = np.expand_dims(t, axis=1)  # [M,1]

        init_positions = np.expand_dims(init_positions, axis=0)  # [1,N]
        M = np.exp(-1/(2*self.bw**2)*(samples-init_positions)**2)
        P = 2*np.matmul(np.transpose(M), M)
        q = -2*np.matmul(np.transpose(t), M)[0]

        # cvxopt.solvers.qp(P,q,G,h,A,b)
        P = matrix(P)
        q = matrix(q)
        G = matrix(- np.eye(N))
        h = matrix(np.zeros(N))
        A = matrix(np.ones(N)).T
        b = matrix(1.0)
        # quadprog(H,f,A,b,Aeq,beq)
        sio.savemat("optimize.mat", {"H": P, "f": q,
                    "A": G, "b": h, "Aeq": A, "beq": b,'x0':init_weights})

        sol = solvers.qp(P, q, G, h, A, b)
        solved_weights = np.array(sol['x'])
        weights = np.zeros(N)
        weights[45] =1 
        self.build_kde(init_positions,weights ,
                       self.kernel, self.bw)  # 使用聚类得到的结果构建kde
        y_pdf_quad = self.evaluate(samples)
        js_km_quad = jensenshannon(x_pdf, y_pdf_quad)
        self.logger.info(f"quad programming js:{js_km_quad}")

        return init_positions, solved_weights


class AMKdeApproximation(KdeApproximation):
    '''
    alternating minimization
    '''

    def __init__(self, pool_positions: List[float], pool_weights: List[float], kernel: str = "gaussian", bw: int = 0.01) -> None:
        super().__init__(pool_positions, pool_weights)
        self.kernel = kernel
        self.bw = bw
        self.pool_size = 100
        self.sample_num = 100
        self.logger = Logger()

    def update_positions(self, append_positions: List[float], append_weights: List[float]) -> None:
        all_positions = self.pool_positions
        all_positions.extend(append_positions)
        all_weights = self.pool_weights
        all_weights.extend(append_weights)

        self.build_kde(all_positions, all_weights, self.kernel, self.bw)
        # 控制采样多少个samples
        samples = self.kde.sample(self.sample_num)
        samples = sorted(samples)

        solved_positions, positions_weights = self.optimize_pts_am(
            all_positions, all_weights, samples)

        self.pool_positions = solved_positions
        self.pool_weights = positions_weights

    def optimize_pts_am(self, all_positions: List[float], all_weights: List[float], samples: List[float]) -> tuple[List[float], List[float]]:
        N = self.pool_size
        M = self.sample_num
        batch_size = self.sample_num
        total = sum(all_weights)

        init_method = "cluster"
        # TODO:初始化方法
        if init_method == "cluster":
            init_positions, init_weights = self.cluster(
                all_positions, all_weights, N)

        x_pdf = self.evaluate(samples)
        self.build_kde(init_positions, init_weights,
                       self.kernel, self.bw)  # 使用聚类得到的结果构建kde
        y_pdf = self.evaluate(samples)
        js_km_cluster = jensenshannon(x_pdf, y_pdf)
        self.logger.info(f"cluster initialization js:{js_km_cluster}")

        solved_positions = torch.Tensor(init_positions).reshape(
            1, N).repeat(batch_size, 1).cuda()
        positions_weights = torch.Tensor(init_weights).reshape(
            1, N).repeat(batch_size, 1).cuda()
        solved_positions.requires_grad = True  # positions to be optimized

        samples_gpu = torch.Tensor(samples).cuda().reshape(M, 1)
        all_positions = torch.Tensor(all_positions).cuda().repeat(M, 1)
        all_weights = torch.Tensor(all_weights).cuda().repeat(M, 1)

        kde_true = ((1/(math.sqrt(2*math.pi)*self.bw))*torch.exp(-1 /
                                                                 (2*self.bw**2)*(samples_gpu-all_positions)**2)*(all_weights/total)).sum(1)

        epoches = 5000
        optimizer = torch.optim.Adam(
            [solved_positions], lr=0.00001)

        mse = nn.MSELoss()
        # TODO：试试kl散度作为loss
        kl = torch.nn.KLDivLoss(reduction='batchmean')
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d-%H-%M-%S')
        base_dir = os.path.dirname(__file__)
        writer = SummaryWriter(f'{base_dir}/logs/{time_str}')

        positions_data = positionsDataset(samples_gpu, kde_true)
        positions_loader = DataLoader(
            positions_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        for epoch in tqdm(range(epoches)):
            for train_x, train_y in positions_loader:
                kde_pred = ((1/(math.sqrt(2*math.pi)*self.bw))*torch.exp(-1 / (2*self.bw**2)
                            * (train_x-solved_positions)**2)*(positions_weights/total)).sum(1)

                mean_dist = 0.5*(train_y+kde_pred)
                loss = 0.5*kl(torch.log(mean_dist), train_y) + \
                    0.5*kl(torch.log(mean_dist), kde_pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            solved_positions_cpu, positions_weights_cpu = list(solved_positions[0, :].cpu(
            ).detach().numpy()), list(positions_weights[0, :].cpu().detach().numpy())
            self.build_kde(solved_positions_cpu, positions_weights_cpu,
                           self.kernel, self.bw)  # 使用优化得到的结果构建kde

            y_pdf_am = self.evaluate(samples)
            js_km_am = jensenshannon(x_pdf, y_pdf_am)

            if epoch % 100 == 0:
                writer.add_scalar("loss", loss, global_step=epoch)
                writer.add_scalar("am_js", js_km_am, global_step=epoch)

        self.build_kde(solved_positions_cpu, positions_weights_cpu,
                       self.kernel, self.bw)  # 使用优化得到的结果构建kde
        y_pdf_am = self.evaluate(samples)
        js_km_am = jensenshannon(x_pdf, y_pdf_am)
        self.logger.info(f"am optim js:{js_km_am}")

        with open(f"{base_dir}/logs/js.txt", "a") as f:
            f.write(f"{js_km_cluster},{js_km_am}\n")

        plt.plot(samples, y_pdf, color='r', linestyle=':', label='cluster')
        plt.plot(samples, y_pdf_am, color='g', linestyle='-', label='am')
        plt.legend()
        plt.savefig("samples.jpg")

        return solved_positions_cpu, positions_weights_cpu


class KdeFactory():
    def get_kde_approximation(self, method: str) -> KdeApproximation:

        kde_approximation = {
            "base": KdeApproximation,
            "cluster": ClusterKdeApproximation,
            "mid": MidKdeApproximation,
            "lbfgs": LBFGSKdeApproximation,
            "sfw": SFWApproximation,
            "bfw": BasicFWApproximation,
            "sgd": SGDKdeApproximation,
            "quad": QuadKdeApproximation
        }

        return kde_approximation[method]


class positionsDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    # 根据索引获取data和label
    def __getitem__(self, index):
        return self.data[index], self.label[index]  # 以元组的形式返回

    # 获取数据集的大小
    def __len__(self):
        return len(self.data)
