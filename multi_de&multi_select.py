#!/usr/bin/env python
# -*-coding:utf-8-*-
"""
Using genetic algorithm to optimize a three-dimensional decision variable

The individual class is defined to generate the individual and calculate
the fitness which is 1 divided by the objective function. Then, the indi-
viduals are added to the population attribute of the genetic algorithm.
Finally, perform selection, crossover and mutation operation on the popu-
lation to obtain optimal decision variable.

Input:
-------------------------------------------------------------------------
sizepop : int
 population size
vardim : int
      dimension of decision variable
bound : list
     The boundary of the decision variable in each dimension
MAXGEN : int
      Largest evolutionary algebra
param : list
     crossover rate, mutation rate, cross coefficient

There are also have some hyperparameter in different genetic operators.Y-
ou can see details in the function docstring

Output:
-------------------------------------------------------------------------
result : array
      optimal decision variables
"""
import os
import sys
import copy
import math
import time

import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import matplotlib.pyplot as plt


__author__ = 'ru'
sys.path.append(os.path.dirname(__file__))
# 得到当前程序所在的路径
__dirname__ = os.path.dirname(__file__)


def get_path(path):
    """Splice to get the data storage path"""
    return os.path.join(__dirname__, path)


def caculate_concentration(x, y,
                           leaking_point_x=50.,
                           leaking_point_y=25.,
                           leaking_q=10434.78):
    diffusion_coefficient_y = 0.04 * (x - leaking_point_x) *\
                              ((1 + 0.0001 * (x - leaking_point_x)) **
                               (- 1 / 2))
    diffusion_coefficient_z = 0.016 * (x - leaking_point_x) *\
                              ((1 + 0.0003 * (x - leaking_point_x)) **
                               (- 1 / 2))
    concentation = (leaking_q / (2 * math.pi * 2 * diffusion_coefficient_y *
                                 diffusion_coefficient_z)) *\
                   math.exp(- 1 / 2 * ((y - leaking_point_y) /
                                       diffusion_coefficient_y) ** 2)
    if float(concentation.real) > 0:
        return concentation
    else:
        return 0


def generate_data(x_interval=200, y_interval=250):
    data = []
    for i in range(1 + int(1000 / x_interval)):
        for j in range(1 + int(1000 / y_interval)):
            x = i * x_interval + norm.rvs(scale=5)
            y = -500 + j * y_interval + norm.rvs(scale=5)
            data.append((x, y, caculate_concentration(x, y)))
    return data


global data
data = generate_data()


def likelihood(iterator_x, iterator_y, iterator_q, scale=0.01):
    total = 0
    for i in range(len(data)):
        x = data[i][0]
        y = data[i][1]
        measure_con = data[i][2]
        total += (measure_con -
                  caculate_concentration(x, y, leaking_point_x=iterator_x,
                                         leaking_point_y=iterator_y,
                                         leaking_q=iterator_q)) ** 2
    plikelihood = - total / (2 * scale * scale)
    return plikelihood


class GAIndividual:

    def __init__(self, vardim, bound):
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.
        self.chrom = np.zeros(self.vardim)
        self.site = 0

    def generate(self, count='a'):
        if count == 'a':
            len = self.vardim
            rnd = np.random.random(size=len)
            for i in range(len):
                self.chrom[i] = (self.bound[i][0] +
                                 (self.bound[i][1] -
                                 self.bound[i][0]) *
                                 rnd[i])
        else:
            rnd = np.random.random()
            # 简单注释                                                              第一维分为几块   第一维分为几块
            self.chrom[0] = self.bound[0][0] + ((self.bound[0][1] - self.bound[0][0]) / 5) * (count % 5 + rnd)
            rnd = np.random.random()
            # 简单注释                                                              第二维分为几块           二维分为几块   第二维分为几块
            self.chrom[1] = self.bound[1][0] + ((self.bound[1][1] - self.bound[1][0]) / 5)* ((count - (count // 25) * 25) // 3 + rnd)
            rnd = np.random.random()
            # 简单注释                                                              第三维分为几块    二维分为几块
            self.chrom[2] = self.bound[2][0] + ((self.bound[2][1] - self.bound[2][0]) / 4)* (count // 25 + rnd)
            self.site = int(count)


    def calculate_fitness(self):
        self.fitness = 1. / (- likelihood(self.chrom[0], self.chrom[1],
                                          self.chrom[2]))

class GeneticAlgorithm:

    def __init__(self, sizepop, vardim, bound, MAXGEN, params, cross_operator_prop):
        """
        :param sizepop:  population size
        :param vardim:  dimension of variable
        :param bound:  boundaries of variable
        :param MAXGEN:  termination condition
        :param params: crossover rate, mutation rate, alpha
        :return:
        """
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.params = params
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.cross_operator_prop = cross_operator_prop

    def initialize(self):
        """Initialize the individual"""
        individual_count = 0
        # individual_count = 'a'
        for _ in range(self.sizepop):
            individual = GAIndividual(self.vardim, self.bound)
            individual.generate(count=individual_count)
            self.population.append(individual)
            individual_count += 1
        individual_count = 0

    def evaluate(self):
        """Calculating individual fitness in a population"""
        for i in range(self.sizepop):
            self.population[i].calculate_fitness()
            self.fitness[i] = self.population[i].fitness

    def solve(self):
        """Evolution process of genetic algorithm"""
        start = time.time()
        self.t = 0
        self.initialize()
        self.evaluate()

        best = np.max(self.fitness)
        best_index = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[best_index])
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] = self.best.fitness
        self.trace[self.t, 1] = self.avefitness

        while (self.t < self.MAXGEN - 1):
            self.t += 1
            F1 = 0.3 + ((self.MAXGEN - self.t + 1) / self.MAXGEN) * 0.3
            F = 0.6 - ((self.MAXGEN - self.t + 1) / self.MAXGEN) * 0.3
            prop = 0.4 + ((self.MAXGEN - self.t + 1) / self.MAXGEN) * 0.3
            self.cross_operator_prop = 0.5 + ((self.MAXGEN - self.t + 1) / self.MAXGEN) * 0.2
            n = 70 - self.t
            m = int(n * (0.25 + ((self.MAXGEN - self.t + 1) / self.MAXGEN) * 0.25))
            # m = int(0.5 * n)
            self.multi_selection_operation(prop=prop, n=n, m=m)
            self.multi_de_operation(F=F, F1=F1, CR=self.params[0], cross_operator_prop=0.7)
            self.mutation_operation()  # 突变
            self.evaluate()

            best = np.max(self.fitness)
            best_index = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[best_index])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t][0] = self.best.fitness
            self.trace[self.t][1] = self.avefitness
     
        print("Optimal function value is: %f;" % self.trace[self.t, 0])
        print("Optimal solution is:")
        print(self.best.chrom)
        cost = time.time() - start
        print("Time cost:", cost)
        return self.best.chrom

    def multi_selection_operation(self, prop=0.5, n=10, m=3):
        newpop = []
        total_fitness = np.sum(self.fitness)
        accu_fitness = np.zeros((self.sizepop, 1))

        sum = 0.
        for i in range(self.sizepop):
            accu_fitness[i] = sum + self.fitness[i] / total_fitness
            sum = accu_fitness[i]
        
        roulette_num = int(self.sizepop * prop)
        # 轮盘赌注选择个体
        for i in range(roulette_num):
            r = np.random.random()
            idx = 0
            for j in range(self.sizepop - 1):
                if j == 0 and r < accu_fitness[j]:
                    idx = 0
                    break
                elif r >= accu_fitness[j] and r < accu_fitness[j + 1]:
                    idx = j + 1
                    break
            newpop.append(self.population[idx])
        #  锦标赛选择个体
        count = roulette_num
        judge = True
        while (judge):
            index = np.random.choice(self.sizepop, n, replace=False)  # 不重复选择
            temp_fitness = self.fitness[index]
            for _ in range(0, m):  # 一次竞赛选出m个优胜者
                best_index = np.argmax(temp_fitness)
                newpop.append(self.population[index[best_index]])
                count += 1
                if count == self.sizepop:
                    judge = False
                    break
                temp_fitness = np.delete(temp_fitness, [best_index])  # 去除被选择索引
                index = np.delete(index, [best_index])
        self.population = newpop

    def multi_de_operation(self, F=0.5, F1=0.5, CR=0.9, cross_operator_prop=0.3):
        newpop = []
        for i in range(0, self.sizepop):
            if i < int(self.sizepop * cross_operator_prop):  # DE/rand/1
                index = np.random.choice(self.sizepop, 3, replace=False)
                parents = np.array(self.population)[index]
                FF = F
            else:  # DE/best/1
                index = np.random.choice(self.sizepop, 2, replace=False)
                parents = np.array(self.population)[index]
                parents = np.insert(parents, 0, self.best, 0)  # 保证统一的计算顺序
                FF = F1
            u = np.random.random()
            y = copy.deepcopy(self.population[i])
            if u <= CR:  # 判定是否进行交叉
                cross_pos = np.random.randint(0, self.vardim - 1)
                for j in range(cross_pos, self.vardim):
                    y.chrom[j] = parents[0].chrom[j] + FF *\
                                 (parents[1].chrom[j] - parents[2].chrom[j])
                    if y.chrom[j] > self.bound[j][1]:
                        y.chrom[j] = self.best.chrom[j]  # 防止差分过上限
                    elif y.chrom[j] < self.bound[j][0]:
                        y.chrom[j] = self.best.chrom[j]  # 防止差分过下限
            newpop.append(y)
        self.population = newpop

    def mutation_operation(self):
        """This function  generates offrsing by mutation"""
        newpop = []
        for i in range(self.sizepop):
            newpop.append(copy.deepcopy(self.population[i]))
            r = np.random.random()
            if r < self.params[1]:
                mutate_pos = np.random.randint(0, self.vardim - 1)
                theta = np.random.random()
                if theta > 0.5:
                    newpop[i].chrom[mutate_pos] = (newpop[i].chrom[mutate_pos] -
                                                   (newpop[i].chrom[mutate_pos] -
                                                   self.bound[mutate_pos][0]) *
                                                   (1 - np.random.random() **
                                                   (1 - self.t / self.MAXGEN)))
                else:
                    newpop[i].chrom[mutate_pos] = (newpop[i].chrom[mutate_pos] +
                                                   (self.bound[mutate_pos][1] -
                                                   newpop[i].chrom[mutate_pos]) *
                                                   (1 - np.random.random() **
                                                   (1 - self.t / self.MAXGEN)))
        self.population = newpop

if __name__ == '__main__':
    bound = [[0, 1000], [-500, 500], [0, 20000]]
    final_result = []
    for i in range(100):
        demo = GeneticAlgorithm(100, 3, bound, 30, [0.9, 0.1, 0.7], [0.3, 0.6])
        result = demo.solve()
        path = get_path(r"tmp\detials\ " + str(i+1) + "_" +  "best_trace.txt")
        np.savetxt(path, demo.trace[:, 0])
        path = get_path(r"tmp\detials\ " + str(i+1) + "_" +  "aver_trace.txt")
        np.savetxt(path, demo.trace[:, 1])
        path = get_path(r"tmp\detials\ " + str(i+1) + "_" +  "population.txt")
        coordinate = []
        for i in range(100):
            coordinate.append(demo.population[i].chrom)
        np.savetxt(path, coordinate)
        final_result.append([demo.best.chrom[0], demo.best.chrom[1], demo.best.chrom[2], demo.trace[-1, 0]])
    final_result = np.array(final_result)
    np.savetxt(get_path("tmp\\ " + "_" + "all_result.txt"), final_result)
    minmum = np.min(final_result[:, 3])
    maxmum = np.max(final_result[:, 3])
    aver = np.mean(final_result[:, 3])
    std = np.std(final_result[:, 3])
    np.savetxt(get_path("tmp\\ " + "_" + "result.txt"), [minmum, maxmum, aver, std])
