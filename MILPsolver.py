# -*- coding: utf-8 -*-
# TSP-D 子问题1 (多目标MILP) —— 按 IPforTSP_rr.py 的代码风格改写
# 节点编号：0 = 起点仓库, 1..c = 客户, c+1 = 虚拟终点仓库
# "提出虚拟点n+1，dismat需要加一个维度, 共c+2维" (与你原TSP注释一致)
# 公式对应：目标(1)(2)、节点风险(3)、约束(5)~(38)（论文无编号4）
from gurobipy import *
import re
import numpy as np


class MILPConstrain:
    def __init__(self, holdinglist):
        self.model = None
        num = holdinglist[-1] + 1                                   # 列表需要造大, 共 c+2 维
        # X[i][j] 骑手弧
        self.X = [[0] * num for _ in range(num)]
        # Y[i][j][k] UAV行程: i发射, j服务, k回收 (三维占位)
        self.Y = [[[0] * num for _ in range(num)] for _ in range(num)]
        # U[i] 客户i在骑手路径中的位次
        self.U = [0] * num
        # P[i][j] 骑手路径中 i 是否先于 j
        self.P = [[0] * num for _ in range(num)]
        # T[i][k] i-k之间是否启用换电中转站
        self.T = [[0] * num for _ in range(num)]
        # Z[jp][j] UAV单次行程中 jp 先于 j 被服务 (对应论文 Z_j'j)
        self.Z = [[0] * num for _ in range(num)]
        # t[i] 骑手到达时刻 / tp[i] UAV到达时刻
        self.t = [0] * num
        self.tp = [0] * num
        # 两个目标表达式
        self.OF1 = None
        self.OF2 = None

    def build_model(self, holdinglist, C_prime, tau, tau_r, D,
                    s_L=0.05, s_R=0.05, e=2, c=None, M=None,
                    weight_OF1=1, weight_OF2=0, timelimit=3600,
                    solve_model=False):
        # holdinglist: [0,1,...,c,c+1]   C_prime: UAV可服务客户
        # tau:   骑手时间矩阵 (欧氏/骑手速度)  tau_r: UAV时间矩阵 (曼哈顿/UAV速度)
        # D:     节点风险字典 {客户j: D_j}, 由公式(3)在模型外算好传入
        self.model = Model("MILPforTSPD")
        cN = holdinglist[-1]
        C = holdinglist[1:-1]                    # 全部客户
        N_O = holdinglist[:-1]                   # 可发射节点 0..c
        N_D = holdinglist[1:]                    # 可回收节点 1..c+1
        if c is None:
            c = len(C)
        if M is None:
            M = 10 * max(max(row) for row in tau) + \
                10 * max(max(row) for row in tau_r) + 1000
        print("M的取值lixiao",M)
        C_prime = list(C_prime)
        C_not = [j for j in C if j not in C_prime]

        # ================= 构建决策变量 =================
        for i in N_O:                            # 骑手弧 x_ij
            for j in N_D:
                if i != j:
                    self.X[i][j] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY,
                                                     name="X_" + str(i) + "_" + str(j))
        for i in N_O:                            # UAV行程 y_ijk
            for j in C_prime:
                for k in N_D:
                    if i != j and i != k and j != k:
                        self.Y[i][j][k] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY,
                                                            name="Y_" + str(i) + "_" + str(j) + "_" + str(k))
        for i in C:                              # 位次 u_i
            self.U[i] = self.model.addVar(lb=1, ub=c, vtype=GRB.INTEGER, name="U_" + str(i))
        for i in C:                              # 先后关系 p_ij
            for j in C:
                if i != j:
                    self.P[i][j] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY,
                                                     name="P_" + str(i) + "_" + str(j))
        for i in N_O:                            # 换电站 T_ik
            for k in N_D:
                if i != k:
                    self.T[i][k] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY,
                                                     name="T_" + str(i) + "_" + str(k))
        for jp in C_prime:                       # 服务顺序 Z_j'j
            for j in C_prime:
                if jp != j:
                    self.Z[jp][j] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY,
                                                      name="Z_" + str(jp) + "_" + str(j))
        for i in holdinglist:                    # 到达时刻
            self.t[i] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t_" + str(i))
            self.tp[i] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="tp_" + str(i))

        # ================= 目标函数 =================
        self.OF1 = LinExpr()                     # 公式(1): 骑手效率 (min)
        for i in N_O:
            for j in N_D:
                if i != j and tau[i][j] != 0:
                    self.OF1.addTerms(tau[i][j], self.X[i][j])
        self.OF2 = LinExpr()                     # 公式(2): UAV承担风险 (max)
        for j in C_prime:
            for i in N_O:
                for k in N_D:
                    if i != j and i != k and j != k:
                        self.OF2.addTerms(D[j], self.Y[i][j][k])
        obj = weight_OF1 * self.OF1 - weight_OF2 * self.OF2   # 加权法: min w1*OF1 - w2*OF2
        self.model.setObjective(obj, GRB.MINIMIZE)
        #伊普西龙约束
        expr = LinExpr()
        for i in N_O:
            for j in C_prime:
                for k in N_D:
                    if i != j and i != k and j != k:
                        expr.addTerms(D[j], self.Y[i][j][k])

        self.model.addConstr(expr >= 190600, name="sum_Dj_yijk")

        # ============ (i) 路径可行性 约束(5)~(11) ============
        # 5: C'中客户由骑手或UAV二选一
        for j in C_prime:
            expr = LinExpr()
            for i in N_O:
                if i != j:
                    expr.addTerms(1, self.X[i][j])
            for i in N_O:
                for k in N_D:
                    if i != j and i != k and j != k:
                        expr.addTerms(1, self.Y[i][j][k])
            self.model.addConstr(expr == 1, "c5_" + str(j))
        # 6: C\C'中客户只能由骑手服务
        for j in C_not:
            expr = LinExpr()
            for i in N_O:
                if i != j:
                    expr.addTerms(1, self.X[i][j])
            self.model.addConstr(expr == 1, "c6_" + str(j))
        # 7: 骑手从起点仓库出发
        expr = LinExpr()
        for j in N_D:
            if j != 0:
                expr.addTerms(1, self.X[0][j])
        self.model.addConstr(expr == 1, "c7")
        # 8: 骑手最终回到终点仓库
        expr = LinExpr()
        for i in N_O:
            if i != cN:
                expr.addTerms(1, self.X[i][cN])
        self.model.addConstr(expr == 1, "c8")
        # 9: 来源=去路 (流守恒)
        for j in C:
            expr = LinExpr()
            for i in N_O:
                if i != j:
                    expr.addTerms(1, self.X[i][j])
            for k in N_D:
                if k != j:
                    expr.addTerms(-1, self.X[j][k])
            self.model.addConstr(expr == 0, "c9_" + str(j))
        # 10: UAV在客户i发射 => 骑手必须到访i (论文下标 ∀i∈C)
        for i in C:
            for j in C_prime:
                for k in N_D:
                    if i != j and i != k and j != k:
                        expr = LinExpr()
                        for h in N_O:
                            if h != i:
                                expr.addTerms(1, self.X[h][i])
                        expr.addTerms(-1, self.Y[i][j][k])
                        self.model.addConstr(expr >= 0, "c10_" + str(i) + "_" + str(j) + "_" + str(k))
        for i in N_O:  # 不再限于 i∈C
            for j in C_prime:
                for k in N_D:
                    if i != j and i != k and j != k:
                        expr = LinExpr()
                        for h in N_O:
                            if h != k:
                                expr.addTerms(1, self.X[h][k])
                        expr.addTerms(-1, self.Y[i][j][k])
                        self.model.addConstr(expr >= 0,
                                             "c11fix_" + str(i) + "_" + str(j) + "_" + str(k))
        # 11: UAV在k回收 => 骑手必须到访k
        for i in N_O:
            for j in C_prime:
                for k in N_D:
                    if i != j and i != k and j != k:
                        expr = LinExpr()
                        for l in N_O:
                            if l != k:
                                expr.addTerms(1, self.X[l][k])
                        expr.addTerms(-1, self.Y[i][j][k])
                        self.model.addConstr(expr >= 0, "c11_" + str(i) + "_" + str(j) + "_" + str(k))

        # ============ (ii) 变量定义相关 约束(12)~(24) ============
        # 12/13: T_ik定义——行程i→k服务2个客户时激活
        for i in N_O:
            for k in N_D:
                if i != k:
                    expr = LinExpr()
                    for j in C_prime:
                        if j != i and j != k:
                            expr.addTerms(1, self.Y[i][j][k])
                    self.model.addConstr(expr >= 2 * self.T[i][k], "c12_" + str(i) + "_" + str(k))
                    self.model.addConstr(expr <= 1 + self.T[i][k], "c13_" + str(i) + "_" + str(k))
        # 14: 每客户在Z关系中最多出现一次
        for j in C_prime:
            expr = LinExpr()
            for jp in C_prime:
                if jp != j:
                    expr.addTerms(1, self.Z[jp][j])
                    expr.addTerms(1, self.Z[j][jp])
            self.model.addConstr(expr <= 1, "c14_" + str(j))
        # 15: 同一行程服务j,j' => 必须指定先后
        for i in N_O:
            for j in C_prime:
                for jp in C_prime:
                    if jp == j:
                        continue
                    for k in N_D:
                        if i != j and i != k and j != k and i != jp and jp != k:
                            expr = LinExpr()
                            expr.addTerms(1, self.Z[jp][j])
                            expr.addTerms(1, self.Z[j][jp])
                            expr.addTerms(-1, self.Y[i][j][k])
                            expr.addTerms(-1, self.Y[i][jp][k])
                            self.model.addConstr(expr >= -1, "c15_" + str(i) + "_" + str(j) + "_" + str(jp) + "_" + str(k))
        # 16: Z关系必须依附实际UAV行程
        for j in C_prime:
            expr = LinExpr()
            for jp in C_prime:
                if jp != j:
                    expr.addTerms(1, self.Z[jp][j])
                    expr.addTerms(1, self.Z[j][jp])
            for i in N_O:
                for k in N_D:
                    if i != j and i != k and j != k:
                        expr.addTerms(-1, self.Y[i][j][k])
            self.model.addConstr(expr <= 0, "c16_" + str(j))
        # 17: 单客户行程(T_ik=0)不允许Z关系
        for i in N_O:
            for j in C_prime:
                for k in N_D:
                    if i != j and i != k and j != k:
                        expr = LinExpr()
                        for jp in C_prime:
                            if jp != j:
                                expr.addTerms(1, self.Z[jp][j])
                                expr.addTerms(1, self.Z[j][jp])
                        expr.addTerms(1, self.Y[i][j][k])
                        expr.addTerms(-1, self.T[i][k])
                        self.model.addConstr(expr <= 1, "c17_" + str(i) + "_" + str(j) + "_" + str(k))
        # 18: Z与y/T的一致性
        for i in N_O:
            for j in C_prime:
                for jp in C_prime:
                    if jp == j:
                        continue
                    for k in N_D:
                        if i != j and i != k and j != k and i != jp and jp != k:
                            expr = LinExpr()
                            expr.addTerms(1, self.Z[jp][j])
                            expr.addTerms(1, self.Z[j][jp])
                            expr.addTerms(-1, self.Y[i][j][k])
                            expr.addTerms(1, self.Y[i][jp][k])
                            expr.addTerms(1, self.T[i][k])
                            self.model.addConstr(expr <= 2, "c18_" + str(i) + "_" + str(j) + "_" + str(jp) + "_" + str(k))
        # 19: UAV两客户访问顺序取时间更短者
        for i in N_O:
            for j in C_prime:
                for jp in C_prime:
                    if jp == j:
                        continue
                    for k in N_D:
                        if i != j and i != k and j != k and i != jp and jp != k:
                            lhs = tau_r[i][jp] + tau_r[jp][j] + tau_r[j][k]
                            rhs = tau_r[i][j] + tau_r[j][jp] + tau_r[jp][k] + M * (2 - self.Y[i][j][k] - self.Z[jp][j])
                            self.model.addConstr(lhs <= rhs, "c19_" + str(i) + "_" + str(j) + "_" + str(jp) + "_" + str(k))
        # 20: x_ij=1 => u_j ≥ u_i+1
        # 22/23: p_ij 与 u 的耦合
        # 24: p_ij + p_ji = 1
        for i in C:
            for j in C:#这里j可能是in ND
                if i == j:
                    continue
                expr = LinExpr()
                expr.addTerms(1, self.U[i])
                expr.addTerms(-1, self.U[j])
                expr.addTerms(c + 2, self.X[i][j])
                self.model.addConstr(expr <= c + 1, "c20_" + str(i) + "_" + str(j))
                expr2 = LinExpr()
                expr2.addTerms(1, self.U[i])
                expr2.addTerms(-1, self.U[j])
                expr2.addTerms(c + 2, self.P[i][j])
                self.model.addConstr(expr2 >= 1, "c22_" + str(i) + "_" + str(j))
                self.model.addConstr(expr2 <= c + 1, "c23_" + str(i) + "_" + str(j))
                expr3 = LinExpr()
                expr3.addTerms(1, self.P[i][j])
                expr3.addTerms(1, self.P[j][i])
                self.model.addConstr(expr3 == 1, "c24_" + str(i) + "_" + str(j))
        # 21: UAV行程i→k => u_k ≥ u_i+1
        for i in C:
            for k in C:
                if i == k:
                    continue
                expr = LinExpr()
                expr.addTerms(1, self.U[k])
                expr.addTerms(-1, self.U[i])
                expr.addTerms(c + 2, self.T[i][k])
                for j in C_prime:
                    if j != i and j != k:
                        expr.addTerms(-(c + 2), self.Y[i][j][k])
                self.model.addConstr(expr >= 1 - (c + 2), "c21_" + str(i) + "_" + str(k))

        # ============ (iii) 时间约束 (25)~(35) ============
        # 25/26: UAV从i单客户发射 => t_i = t'_i
        for i in N_O:
            exprT = LinExpr()
            for k in N_D:
                if k != i:
                    exprT.addTerms(1, self.T[i][k])
            exprY = LinExpr()
            for j in C_prime:
                for k in N_D:
                    if i != j and i != k and j != k:
                        exprY.addTerms(1, self.Y[i][j][k])
            self.model.addConstr(self.t[i] - self.tp[i] - M * exprT + M * exprY <= M, "c25_" + str(i))
            self.model.addConstr(self.t[i] - self.tp[i] + M * exprT - M * exprY >= -M, "c26_" + str(i))
        # 27/28: UAV在k单客户回收 => t_k = t'_k
        for k in N_D:
            exprT = LinExpr()
            for i in N_O:
                if i != k:
                    exprT.addTerms(1, self.T[i][k])
            exprY = LinExpr()
            for i in N_O:
                for j in C_prime:
                    if i != j and i != k and j != k:
                        exprY.addTerms(1, self.Y[i][j][k])
            self.model.addConstr(self.t[k] - self.tp[k] - M * exprT + M * exprY <= M, "c27_" + str(k))
            self.model.addConstr(self.t[k] - self.tp[k] + M * exprT - M * exprY >= -M, "c28_" + str(k))
        # 29: 单客户行程第一段
        for i in N_O:
            for j in C_prime:
                if i == j:
                    continue
                exprY = LinExpr()
                for k in N_D:
                    if k != j and k != i:
                        exprY.addTerms(1, self.Y[i][j][k])
                exprT = LinExpr()
                for k in N_D:
                    if k != i:
                        exprT.addTerms(1, self.T[i][k])
                self.model.addConstr(self.tp[j] - self.tp[i] - M * exprY + M * exprT >= tau_r[i][j] - M,
                                     "c29_" + str(i) + "_" + str(j))
        # 30: 单客户行程第二段
        for j in C_prime:
            for k in N_D:
                if j == k:
                    continue
                exprY = LinExpr()
                for i in N_O:
                    if i != j and i != k:
                        exprY.addTerms(1, self.Y[i][j][k])
                exprT = LinExpr()
                for i in N_O:
                    if i != k:
                        exprT.addTerms(1, self.T[i][k])
                self.model.addConstr(self.tp[k] - self.tp[j] - M * exprY + M * exprT >= tau_r[j][k] - M,
                                     "c30_" + str(j) + "_" + str(k))
        # 31: 双客户行程第一段 (j'为第一站)
        for i in N_O:
            for jp in C_prime:
                if i == jp:
                    continue
                exprZ = LinExpr()
                for j in C_prime:
                    if j != jp:
                        exprZ.addTerms(1, self.Z[jp][j])
                exprY = LinExpr()
                for k in N_D:
                    if k != jp and k != i:
                        exprY.addTerms(1, self.Y[i][jp][k])
                self.model.addConstr(self.tp[jp] - self.tp[i] - M * exprZ - M * exprY >= tau_r[i][jp] - 2 * M,
                                     "c31_" + str(i) + "_" + str(jp))
        # 32: UAV两客户间衔接
        for jp in C_prime:
            for j in C_prime:
                if jp == j:
                    continue
                self.model.addConstr(self.tp[j] - self.tp[jp] - M * self.Z[jp][j] >= tau_r[jp][j] - M,
                                     "c32_" + str(jp) + "_" + str(j))
        # 33: 双客户行程第二段 (j为第二站)
        for j in C_prime:
            for k in N_D:
                if j == k:
                    continue
                exprZ = LinExpr()
                for jp in C_prime:
                    if jp != j:
                        exprZ.addTerms(1, self.Z[jp][j])
                exprY = LinExpr()
                for i in N_O:
                    if i != j and i != k:
                        exprY.addTerms(1, self.Y[i][j][k])
                self.model.addConstr(self.tp[k] - self.tp[j] - M * exprZ - M * exprY >= tau_r[j][k] - 2 * M,
                                     "c33_" + str(j) + "_" + str(k))
        # 34: 骑手到达时刻计入发射/回收耗时 s_L, s_R
        for h in N_O:
            for k in N_D:
                if h == k:
                    continue
                exprYL = LinExpr()               # 在h发射的行程
                for j in C_prime:
                    for m in N_D:
                        if h != j and h != m and j != m:
                            exprYL.addTerms(1, self.Y[h][j][m])
                exprTL = LinExpr()               # h发出的换电行程
                for m in N_D:
                    if m != h:
                        exprTL.addTerms(1, self.T[h][m])
                exprYR = LinExpr()               # 在k回收的行程
                for i in N_O:
                    for j in C_prime:
                        if i != j and i != k and j != k:
                            exprYR.addTerms(1, self.Y[i][j][k])
                exprTR = LinExpr()               # 回到k的换电行程
                for i in N_O:
                    if i != k:
                        exprTR.addTerms(1, self.T[i][k])
                self.model.addConstr(
                    self.t[k] - self.t[h] - s_L * exprYL + s_L * exprTL
                    - s_R * exprYR + s_R * exprTR - M * self.X[h][k] >= tau[h][k] - M,
                    "c34_" + str(h) + "_" + str(k))
        # 35: 连续飞, 后一次launch一定晚于前一次recovery
        for i in C:
            for l in C:
                if i == l:
                    continue
                for k in C:
                    if k == i:
                        continue
                    exprY1 = LinExpr()
                    for j in C_prime:
                        if j != i and j != k:
                            exprY1.addTerms(1, self.Y[i][j][k])
                    exprY2 = LinExpr()
                    for m in C_prime:
                        for n in N_D:
                            if m != l and m != k and n != l and m != n:
                                exprY2.addTerms(1, self.Y[l][m][n])
                    exprT2 = LinExpr()
                    for n in N_D:
                        if n != l:
                            exprT2.addTerms(1, self.T[l][n])
                    self.model.addConstr(
                        self.tp[l] - self.tp[k] - M * exprY1 + M * self.T[i][k]
                        - M * exprY2 + M * exprT2 - M * self.P[i][l] >= -3 * M,
                        "c35_" + str(i) + "_" + str(l) + "_" + str(k))
        for l in C:
            for k in N_D:
                if l == k: continue
                exprY1 = LinExpr()  # Σ_j y[0][j][k]
                for j in C_prime:
                    if j != 0 and j != k: exprY1.addTerms(1, self.Y[0][j][k])
                exprY2 = LinExpr()  # ΣΣ y[l][m][n]
                for m in C_prime:
                    for n in N_D:
                        if m != l and m != n and n != l: exprY2.addTerms(1, self.Y[l][m][n])
                exprT1 = LinExpr()
                for m in N_D:
                    if m != 0: exprT1.addTerms(1, self.T[0][m])
                exprT2 = LinExpr()
                for n in N_D:
                    if n != l: exprT2.addTerms(1, self.T[l][n])
                self.model.addConstr(
                    self.tp[l] - self.tp[k] - M * exprY1 + M * exprT1
                    - M * exprY2 + M * exprT2 >= -2 * M,
                    "c35depot_" + str(l) + "_" + str(k))
        # ============ (iv) 续航约束 (36)~(38) ============
        # 36: 单客户行程耗电 ≤ e
        for i in N_O:
            for j in C_prime:
                if i == j:
                    continue
                for k in N_D:
                    if i != k and j != k:
                        lhs = tau_r[i][j] + tau_r[j][k]
                        self.model.addConstr(lhs <= e + M - M * self.Y[i][j][k] + M * self.T[i][k],
                                             "c36_" + str(i) + "_" + str(j) + "_" + str(k))
        # 37: 双客户前半段 i→j'→j ≤ e
        for i in N_O:
            for j in C_prime:
                for jp in C_prime:
                    if jp == j or i == jp or i == j:
                        continue
                    exprY = LinExpr()
                    for k in N_D:
                        if k != j and k != i:
                            exprY.addTerms(1, self.Y[i][j][k])
                    lhs = tau_r[i][jp] + tau_r[jp][j]
                    self.model.addConstr(lhs <= e + 2 * M - M * self.Z[jp][j] - M * exprY,
                                         "c37_" + str(i) + "_" + str(j) + "_" + str(jp))
        # 38: 双客户后半段 j→k ≤ e
        for j in C_prime:
            for k in N_D:
                if j == k:
                    continue
                exprZ = LinExpr()
                for jp in C_prime:
                    if jp != j:
                        exprZ.addTerms(1, self.Z[jp][j])
                exprY = LinExpr()
                for i in N_O:
                    if i != j and i != k:
                        exprY.addTerms(1, self.Y[i][j][k])
                self.model.addConstr(tau_r[j][k] <= e + 2 * M - M * exprZ - M * exprY,
                                     "c38_" + str(j) + "_" + str(k))

        # 特殊首尾点 (同你原TSP写法)
        self.model.addConstr(self.t[0] == 0, "t_depot")
        self.model.addConstr(self.tp[0] == 0, "tp_depot")
        self.model.addConstr(self.X[0][cN] == 0, "no_direct_return")

        # 输出模型文件 & 求解
        self.model.write('MILPforTSPD.lp')
        self.model.write('MILPforTSPD.MPS')
        self.model.Params.timelimit = timelimit
        if solve_model == True:
            self.model.optimize()
        print("MILP输入成功")


def get_solution(holdinglist, tau, D, model):
    # ---- 解析变量名, 填回矩阵 (同你原get_solution的解析方式) ----
    num = holdinglist[-1] + 1
    X = [[0] * num for _ in range(num)]
    Y = [[[0] * num for _ in range(num)] for _ in range(num)]
    T = [[0] * num for _ in range(num)]
    Z = [[0] * num for _ in range(num)]
    t = [0] * num
    tp = [0] * num
    for v in model.getVars():
        s = re.split(r"_", v.VarName)
        if s[0] == "X" and v.x > 0.5:
            X[int(s[1])][int(s[2])] = 1
        elif s[0] == "Y" and v.x > 0.5:
            Y[int(s[1])][int(s[2])][int(s[3])] = 1
        elif s[0] == "T" and v.x > 0.5:
            T[int(s[1])][int(s[2])] = 1
        elif s[0] == "Z" and v.x > 0.5:
            Z[int(s[1])][int(s[2])] = 1
        elif s[0] == "t":
            t[int(s[1])] = v.x
        elif s[0] == "tp":
            tp[int(s[1])] = v.x
    # ---- 沿X重构骑手路径, 累积行程 (同你原来的while循环写法) ----
    route = [holdinglist[0]]
    accu = [0]
    a = holdinglist[0]
    cost = 0.0
    while a != holdinglist[-1]:
        for i in holdinglist[1:]:
            if X[a][i] == 1:
                route.append(i)
                cost += tau[a][i]
                accu.append(round(cost, 2))
                a = i
                break
    # ---- UAV行程 / 换电站 / 目标值 ----
    trips = [(i, j, k) for i in range(num) for j in range(num) for k in range(num) if Y[i][j][k] == 1]
    stations = [(i, k) for i in range(num) for k in range(num) if T[i][k] == 1]
    orders = [(jp, j) for jp in range(num) for j in range(num) if Z[jp][j] == 1]  # (先, 后)
    uav_cust = sorted(set(j for (i, j, k) in trips))
    OF1 = sum(tau[i][j] * X[i][j] for i in range(num) for j in range(num))
    OF2 = sum(D[j] for j in uav_cust)
    print(route, "累积行程", accu)
    print("UAV行程(发射,服务,回收):", trips, " 换电站:", stations, " 服务顺序(先,后):", orders)
    print("OF1 =", round(OF1, 3), " OF2 =", round(OF2, 3))
    return route, accu, trips


if __name__ == "__main__":
    c = 6                                  # 客户数
    holdinglist = list(range(c + 2))       # 0..c+1 (含虚拟终点)
    C_prime = [1, 3, 5]                    # UAV可服务客户
    v_c, v_r = 1.0, 1.5                    # 骑手/UAV速度
    s_L, s_R, e = 1.0, 1.0, 60.0
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 10, size=(c + 2, 2))
    dis = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)   # 欧氏: 骑手
    dis_M = np.abs(pts[:, None, :] - pts[None, :, :]).sum(axis=-1)     # 曼哈顿: UAV
    tau = (dis / v_c).tolist()
    tau_r = (dis_M / v_r).tolist()
    # 公式(3): 节点风险 D_j = 1/2 * Σ_{h≠j} v_hj^2 * dis_hj
    D = {j: float(0.5 * sum(v_c ** 2 * dis[h][j] for h in range(c + 2) if h != j))
         for j in holdinglist[1:-1]}

    cons = Constrain(holdinglist)
    cons.build_model(holdinglist, C_prime, tau, tau_r, D,
                     s_L=s_L, s_R=s_R, e=e, solve_model=True)
    get_solution(holdinglist, tau, D, cons.model)
