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
import copy
import math
import time

import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import matplotlib.pyplot as plt


__author__ = 'ru'


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

    def generate(self):
        len = self.vardim
        rnd = np.random.random(size=len)
        for i in range(len):
            self.chrom[i] = (self.bound[i][0] +
                             (self.bound[i][1] -
                             self.bound[i][0]) *
                             rnd[i])

    def calculate_fitness(self):
        self.fitness = 1. / (- likelihood(self.chrom[0], self.chrom[1],
                                          self.chrom[2]))


class GeneticAlgorithm:

    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
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

    def initialize(self):
        """Initialize the individual"""
        for i in range(self.sizepop):
            individual = GAIndividual(self.vardim, self.bound)
            individual.generate()
            self.population.append(individual)

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
            self.selection_operation()  # 轮盘赌注选择
            # self.linear_ranking_operation()  # 线性排序选择
            # self.exponential_ranking_operation()  # 指数排序选择
            # self.tournament_selection_operation()  # 锦标赛选择
            # self.crossover_operation()  # 线性交叉
            # self.simulated_binary_crossover_operation()  # 模拟二进制交叉SBX
            # self.differential_evolution_operation()  # 差分进化DE（有三种类型可选）
            # self.blend_crossover_operation()  # 混合交叉BLX-α
            self.center_of_mass_crossover_operation()  # 质心交叉CMX
            # self.simplex_crossover_operation()  # 一种简单型交叉SPX
            self.simplex_crossover_operation1()  # 另一种简单型交叉SPX（包含了一点选择）
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

    def selection_operation(self):
        """
        this function select the next generation based on the
        propotion of each individual
        :return:
        """
        newpop = []
        total_fitness = np.sum(self.fitness)
        accu_fitness = np.zeros((self.sizepop, 1))

        sum = 0.
        for i in range(self.sizepop):
            accu_fitness[i] = sum + self.fitness[i] / total_fitness
            sum = accu_fitness[i]

        for i in range(self.sizepop):
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
        self.population = newpop

    def linear_ranking_operation(self, low=1, high=100):
        """
        this function select the next generation based on the
        propotion of each individual
        ------------------------------------------------------
        parameter
        low : float
           The probability that the minimum fitness individual
           is selected
        high : float
            The probability that the maximum fitness individual
            is selected
        -------------------------------------------------------
        """
        newpop = []
        index = np.argsort(self.fitness)
        temp_fitness = np.zeros((self.sizepop))
        for i in range(0, self.sizepop):
            temp_fitness[index[i]] = i + 1  # 适应度排序值
            temp_fitness[index[i]] = 1 / self.sizepop *\
                                     (low + (high - low) *
                                      (temp_fitness[index[i]] -
                                       1) / (self.sizepop - 1))  # 排序概率
        total_fitness = np.sum(temp_fitness)
        accu_fitness = np.zeros((self.sizepop, 1))

        sum = 0.
        for i in range(self.sizepop):
            accu_fitness[i] = sum + temp_fitness[i] / total_fitness
            sum = accu_fitness[i]

        for i in range(self.sizepop):
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
        self.population = newpop

    def exponential_ranking_operation(self, c=0.5):
        """
        this function select the next generation based on the
        propotion of each individual
        -----------------------------------------------------
        parameter
        c : float
         exponential which is between zero and one
        -----------------------------------------------------
        """
        newpop = []
        index = np.argsort(self.fitness)
        temp_fitness = np.zeros((self.sizepop))
        for i in range(0, self.sizepop):
            temp_fitness[index[i]] = i + 1  # 适应度排序值
            temp_fitness[index[i]] = np.power(c, self.sizepop -
                                              temp_fitness[index[i]])  # 个体指数值
        total_fitness = np.sum(temp_fitness)
        accu_fitness = np.zeros((self.sizepop, 1))

        sum = 0.
        for i in range(self.sizepop):
            accu_fitness[i] = sum + temp_fitness[i] / total_fitness
            sum = accu_fitness[i]

        for i in range(self.sizepop):
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
        self.population = newpop

    def tournament_selection_operation(self, n=5, m=1):
        """
        this function select the next generation based on the
        propotion of each individual
        ------------------------------------------------------
        parameter
        n : int
         the number of individuals participating in one com-
         petition
        m : int
         The number of winners in a competition which must less
         than n
        -------------------------------------------------------
        """
        newpop = []
        for i in range(self.sizepop):
            index = np.random.choice(self.sizepop, n, replace=False)  # 不重复选择
            temp_fitness = self.fitness[index]
            for j in range(0, m):  # 一次竞赛选出m个优胜者
                best_index = np.argmax(temp_fitness)
                newpop.append(self.population[index[best_index]])
                temp_fitness = np.delete(temp_fitness, [best_index])  # 去除被选择索引
                index = np.delete(index, [best_index])
        self.population = newpop

    def crossover_operation(self):
        """This function  generates offrsing by cross"""
        newpop = []
        for i in range(0, self.sizepop, 2):
            idx1 = np.random.randint(0, self.sizepop - 1)
            idx2 = np.random.randint(0, self.sizepop - 1)
            while idx1 == idx2:
                idx2 = np.random.randint(0, self.sizepop - 1)

            newpop.append(copy.deepcopy(self.population[idx1]))
            newpop.append(copy.deepcopy(self.population[idx2]))
            r = np.random.random()
            if r < self.params[0]:
                cross_pos = np.random.randint(0, self.vardim - 1)
                for j in range(cross_pos, self.vardim):
                    temp = newpop[i].chrom[j]
                    newpop[i].chrom[j] = (newpop[i].chrom[j] *
                                          self.params[2] +
                                          (1 - self.params[2]) *
                                          newpop[i + 1].chrom[j])
                    newpop[i + 1].chrom[j] = (newpop[i + 1].chrom[j] *
                                              self.params[2] +
                                              (1 - self.params[2]) *
                                              temp)
        self.population = newpop

    def simulated_binary_crossover_operation(self, rho=1):
        """
        This function  generates offrsing by SBX
        ------------------------------------------------------
        parameter
        rho : int
           Hyperparameter which is one or two
        -------------------------------------------------------
        """
        newpop = []
        for i in range(0, self.sizepop, 2):
            idx1 = np.random.randint(0, self.sizepop - 1)
            idx2 = np.random.randint(0, self.sizepop - 1)
            while idx1 == idx2:
                idx2 = np.random.randint(0, self.sizepop - 1)

            newpop.append(copy.deepcopy(self.population[idx1]))
            newpop.append(copy.deepcopy(self.population[idx2]))
            r = np.random.random()
            if r < self.params[0]:
                cross_pos = np.random.randint(0, self.vardim - 1)
                for j in range(cross_pos, self.vardim):
                    temp = newpop[i].chrom[j]
                    u = np.random.random()
                    if u <= 0.5:
                        u = np.power((2 * u), (1 / (rho + 1)))
                    else:
                        u = np.power((1 / (2 * (1 - u))), (rho + 1))
                    newpop[i].chrom[j] = 0.5 * (newpop[i].chrom[j] * (1 + u) +
                                                (1 - u) * newpop[i + 1].
                                                chrom[j])
                    newpop[i + 1].chrom[j] = 0.5 * (temp * (1 - u) + (1 + u) *
                                                    newpop[i + 1].chrom[j])
        self.population = newpop

    def differential_evolution_operation(self, F=0.5, CR=1, option=1):
        """
        This function  generates offrsing by DE
        ------------------------------------------------------
        parameter
        F : float
           Scaling factor which is between 0 and 1
        CR : float
          Cross probability
        option : int
              1 represents DE/rand/1
              2 represents DE/current-to-best/1
              3 represents DE/best/1
        -------------------------------------------------------
        """
        newpop = []
        for i in range(0, self.sizepop):
            if option == 1:  # DE/rand/1
                index = np.random.choice(self.sizepop, 3, replace=False)
                parents = np.array(self.population)[index]
            elif option == 2:  # DE/current-to-best/1
                choice_list = list(range(0, self.sizepop))
                choice_list.remove(i)  # 除去该个体的索引，保证不重复选择
                index = np.random.choice(choice_list, 2, replace=False)
                parents = np.array(self.population)[index]
                parents = np.insert(parents, 0, self.best, 0)  # 保证统一的计算顺序
            else:  # DE/best/1
                index = np.random.choice(self.sizepop, 2, replace=False)
                parents = np.array(self.population)[index]
                parents = np.insert(parents, 0, self.best, 0)  # 保证统一的计算顺序
            u = np.random.random()
            y = copy.deepcopy(self.population[i])
            if u <= CR:  # 判定是否进行交叉
                cross_pos = np.random.randint(0, self.vardim - 1)
                for j in range(cross_pos, self.vardim):
                    y.chrom[j] = parents[0].chrom[j] + F *\
                                 (parents[1].chrom[j] - parents[2].chrom[j])
                    if option == 2:
                        y.chrom[j] = y.chrom[j] + F * (self.best.chrom[j] -
                                                       parents[0].chrom[j])
                    if y.chrom[j] > self.bound[j][1]:
                        y.chrom[j] = self.bound[j][1]  # 防止差分过上限
                    elif y.chrom[j] < self.bound[j][0]:
                        y.chrom[j] = self.bound[j][0]  # 防止差分过下限
            newpop.append(y)
        self.population = newpop

    def blend_crossover_operation(self, alph=0.5):
        """
        This function  generates offrsing by LBX-α
        ------------------------------------------------------
        parameter
        alph : int
           Hyperparameter
        -------------------------------------------------------
        """
        newpop = []
        for i in range(0, self.sizepop, 2):
            index = np.random.choice(self.sizepop, 2, replace=False)
            parents = np.array(self.population)[index]
            center = np.zeros((self.vardim))
            for k in range(0, self.vardim):  # 计算父代中心
                for j in range(0, 2):
                    center[k] = center[k] + parents[j].chrom[k]
                center[k] = center[k] / 2
            low = np.zeros((self.vardim))
            high = np.zeros((self.vardim))
            for j in range(0, self.vardim):  # 计算每个维度的搜索上下限
                d = np.abs(parents[0].chrom[j] - parents[1].chrom[j])
                low[j] = center[j] - d / 2 - alph * d
                high[j] = center[j] - d / 2 - alph * d
            for j in range(0, 2):
                y = copy.deepcopy(parents[j])
                r = np.random.random()
                if r < self.params[0]:
                    cross_pos = np.random.randint(0, self.vardim - 1)
                    for k in range(cross_pos, self.vardim):  # 上下限范围内均匀产生随机值
                        y.chrom[k] = (np.random.uniform(low[k], high[k], 1))[0]
                        if y.chrom[k] > self.bound[k][1]:
                            y.chrom[k] = self.bound[k][1]
                        elif y.chrom[k] < self.bound[k][0]:
                            y.chrom[k] = self.bound[k][0]
                newpop.append(y)
        self.population = newpop

    def center_of_mass_crossover_operation(self, n=5, alph=0.3):
        """
        This function  generates offrsing by CMX
        ------------------------------------------------------
        parameter
        n : int
         The number of parents
        alph : float
            The spatial expansion factor
        -------------------------------------------------------
        """
        newpop = []
        for i in range(0, self.sizepop, n):
            index = np.random.choice(self.sizepop, n, replace=False)
            parents = np.array(self.population)[index]
            center = np.zeros((1, self.vardim))
            for k in range(0, self.vardim):  # 计算多父代中心
                for j in range(0, n):
                    center[0][k] = center[0][k] + parents[j].chrom[k]
                center[0][k] = center[0][k] / n
            vir_parents = np.zeros((n, self.vardim))
            for j in range(0, n):  # 计算虚拟父代
                for k in range(0, self.vardim):
                    vir_parents[j][k] = 2 * center[0][k] - parents[j].chrom[k]
            for j in range(0, n):  # 每对父代采用BLX-α算子交叉
                y = copy.deepcopy(self.population[i + j])
                r = np.random.random()
                if r < self.params[0]:
                    cross_pos = np.random.randint(0, self.vardim - 1)
                    for k in range(cross_pos, self.vardim):  # 上下限范围内均匀产生随机值
                        d = np.abs(parents[j].chrom[k] - vir_parents[j][k])
                        low = center[0][k] - d / 2 - alph * d
                        high = center[0][k] + d / 2 + alph * d
                        y.chrom[k] = (np.random.uniform(low, high, 1))[0]
                        if y.chrom[k] > self.bound[k][1]:
                            y.chrom[k] = self.bound[k][1]
                        elif y.chrom[k] < self.bound[k][0]:
                            y.chrom[k] = self.bound[k][0]
                newpop.append(y)
        self.population = newpop

    def simplex_crossover_operation(self, n=5):
        """
        This function  generates offrsing by SPX
        ------------------------------------------------------
        parameter
        n : int
         The number of parents
        -------------------------------------------------------
        """
        eps = np.sqrt(n + 2)
        newpop = []
        for i in range(0, self.sizepop):
            index = np.random.choice(self.sizepop, n + 1, replace=False)
            parents = np.array(self.population)[index]
            gravity_center = np.zeros((self.vardim))
            for k in range(0, self.vardim):  # 计算多父代重心
                for j in range(0, n):
                    gravity_center[k] = gravity_center[k] + parents[j].chrom[k]
            y = copy.deepcopy(self.population[i])
            r1 = np.random.random()
            if r1 < self.params[0]:
                r = np.ones((n + 1))  # 过渡参数
                c = np.zeros((n + 1, self.vardim))  # 过渡值
                temp = np.zeros((n + 1, self.vardim))  # 过渡值
                for j in range(0, n + 1):  # 选中父代迭代产生一个子代
                    if j != n:
                        u = np.random.random()
                        r[j] = np.power(u, 1 / (j + 1))
                    for k in range(0, self.vardim):
                        temp[j][k] = gravity_center[k] + eps *\
                                     (parents[j].chrom[k] - gravity_center[k])
                    if j != 0:
                        for k in range(0, self.vardim):
                            c[j][k] = r[j - 1] * (temp[j][k] -
                                                  temp[j - 1][k] + c[j - 1][k])
                for k in range(0, self.vardim):  # 上下限范围内产生子代
                    y.chrom[k] = temp[n][k] + c[n][k]
                    if y.chrom[k] > self.bound[k][1]:
                        y.chrom[k] = self.bound[k][1]
                    elif y.chrom[k] < self.bound[k][0]:
                        y.chrom[k] = self.bound[k][0]
            newpop.append(y)
        self.population = newpop

    def simplex_crossover_operation1(self, n=5):
        """
        This function  generates offrsing by CMX
        ------------------------------------------------------
        parameter
        n : int
         The number of parents
        -------------------------------------------------------
        """
        new = []
        index = np.argsort(self.fitness)
        for i in range(0, self.sizepop // n):  # 产生sizepop/n个子代
            parents = np.array(self.population)[i * n:i * n + n]  # n个父代为一组
            temp_fitness = self.fitness[i * n:i * n + n]
            temp_fitness = temp_fitness.reshape(-1)  # 转换为一维便于计算
            group_index = np.argsort(temp_fitness)  # 组中父代适应度从小到大排序索引
            max_fitness = temp_fitness[group_index[-1]]  # 组中最大适应度值
            max_parent = parents[group_index[-1]]  # 组中最大适应度个体
            min_fitness = temp_fitness[group_index[0]]  # 组中最小适应度值
            min_parent = parents[group_index[0]]  # 组中最小适应度个体
            center = np.zeros((self.vardim))
            center_order = np.delete(group_index, 0)  # 除最小适应度个体外索引
            for k in range(0, self.vardim):  # 除最小适应度个体外的组群中心
                for j in center_order:
                    center[k] = center[k] + parents[j].chrom[k]
                center[k] = center[k] / (n - 1)
            new_r = copy.deepcopy(self.population[i * n])
            new_e = copy.deepcopy(self.population[i * n])
            new_i = copy.deepcopy(self.population[i * n])
            new_o = copy.deepcopy(self.population[i * n])
            for k in range(0, self.vardim):  # 需要考虑上下限约束情况
                new_r.chrom[k] = 2 * center[k] - min_parent.chrom[k]
                if new_r.chrom[k] > self.bound[k][1]:
                    new_r.chrom[k] = self.bound[k][1]
                elif new_r.chrom[k] < self.bound[k][0]:
                    new_r.chrom[k] = self.bound[k][0]
            new_r_fitness = 1 / (- likelihood(new_r.chrom[0],
                                     new_r.chrom[1], new_r.chrom[2]))
            if new_r_fitness > max_fitness:
                for k in range(0, self.vardim):
                    new_e.chrom[k] = 2 * new_r.chrom[k] - center[k]
                    if new_e.chrom[k] > self.bound[k][1]:
                        new_e.chrom[k] = self.bound[k][1]
                    elif new_e.chrom[k] < self.bound[k][0]:
                        new_e.chrom[k] = self.bound[k][0]
                new_e_fitness = 1 / (- likelihood(new_e.chrom[0],
                                     new_e.chrom[1], new_e.chrom[2]))
                if new_e_fitness > new_r_fitness:
                    new.append(new_e)
                else:
                    new.append(new_r)
            elif min_fitness <= new_r_fitness and new_r_fitness <= max_fitness:
                new.append(new_r)
            else:
                for k in range(0, self.vardim):
                    new_i.chrom[k] = (min_parent.chrom[k] + center[k]) / 2
                new_i_fitness = 1 / (- likelihood(new_i.chrom[0],
                                     new_i.chrom[1], new_i.chrom[2]))
                if new_i_fitness > min_fitness:
                    new.append(new_i)
                else:
                    for k in range(0, self.vardim):
                        new_o.chrom[k] = (min_parent.chrom[k] +
                                          max_parent.chrom[k]) / 2
                    new.append(new_o)
        for i in range(0, self.sizepop // n):  # 取代上一代种最差的sizepop/n个个体
            self.population[index[i][0]] = new[i]

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
    demo = GeneticAlgorithm(100, 3, bound, 500, [0.6, 0.7, 0.7])
    result = demo.solve()
    print(f"Output result is {result}")
