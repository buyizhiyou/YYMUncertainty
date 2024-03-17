#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_kde_approximation.py
@Time    :   2022/12/06 11:43:04
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''
import sys
sys.path.append("../")
import pytest
import numpy as np 
import random 
from kde_factory import KdeFactory
from utils import read_array
from KDEpy import BaseKDE, NaiveKDE, TreeKDE,FFTKDE
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt 

# nums0 = list(read_array("../process_data/cifar10_activation/nums0.pkl"))
# nums1 = list(read_array("../process_data/cifar10_activation/nums1.pkl"))
activation0 = read_array("../process_data/cifar10_activation/features.3_label0_activation.pkl")#(5000, 64, 32, 32)
activation0 = activation0.reshape(activation0.shape[0],-1)#(5000, 65536)
activation1 = read_array("../process_data/cifar10_activation/features.3_label1_activation.pkl")#(5000, 64, 32, 32)
activation1 = activation1.reshape(activation1.shape[0],-1)#(5000, 65536)
# random.seed(1)
idx0 = random.randint(0,activation0.shape[1])
idx1 = random.randint(0,activation1.shape[1])
# print(f'select neuron {idx0}...')
# print(f"select neuron {idx1}...")
nums0 = list(activation0[:,idx0])
nums1 = list(activation0[:,idx1])

factory = KdeFactory()

class TestKde():

    def test_cluster_kde_approximation(self):
        weights = [1]*len(nums0)
        cluster_kde_approximation = factory.get_kde_approximation("cluster")(nums0,weights)
        append_weights = [1]*len(nums1)
        cluster_kde_approximation.update_positions(nums1,append_weights = [1]*len(nums1))

        kde = cluster_kde_approximation.build_kde()

    def test_mid_kde_approximation(self):
        weights = [1]*len(nums0)
        mid_kde_approximation = factory.get_kde_approximation("mid")(nums0,weights)
        mid_kde_approximation.update_positions(nums1,append_weights = [1]*len(nums1))

    def test_optimize_kde_approximation(self):
        for i in range(100):
            random.seed(i)
            idx0 = random.randint(0,activation0.shape[1])
            idx1 = random.randint(0,activation1.shape[1])
            print(f'select neuron {idx0}...')
            print(f"select neuron {idx1}...")
            nums0 = list(activation0[:,idx0])
            nums1 = list(activation0[:,idx1])
            if 0.0 in nums0 or 0.0 in nums1:
                import pdb;pdb.set_trace()
                pass
            weights = [1]*len(nums0)
            optimize_kde_approximation = factory.get_kde_approximation("sgd")(nums0,weights)
            optimize_kde_approximation.update_positions(nums1,append_weights = [1]*len(nums1))

    
    def test_sklearn(self):
        nums0.extend(nums1)
        positions = np.array(nums0)
        kde = KernelDensity(kernel="gaussian", bandwidth=0.01).fit(positions.reshape(-1, 1))
        x_pdf = np.exp(kde.score_samples(positions.reshape(-1, 1)))
        plt.plot(positions,x_pdf)
        plt.savefig("kde.png")


    def test_kdepy(self):
        estimator = NaiveKDE(kernel='gaussian', bw=0.01)
        global nums0
        nums0.extend(nums1)
        nums0 = sorted(nums0)
        positions = np.array(nums0)

        x_pdf = estimator.fit(positions).evaluate(positions)
        # x,x_pdf = estimator.fit(positions).evaluate(2**10)
        plt.plot(positions,x_pdf)
        plt.savefig("kde.png")

if __name__ == '__main__':
    # pytest.main(
    #     ['-s', "./test_kde_approximation.py::TestKde::test_optimize_kde_approximation"])
    Test = TestKde()
    Test.test_optimize_kde_approximation()
