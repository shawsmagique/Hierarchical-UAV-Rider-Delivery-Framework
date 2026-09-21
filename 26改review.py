import math
import pandas as pd
import numpy as np
import csv
import IPforTSP_rr
import IPforTSP_cr
import manupulate
from matplotlib import pyplot as plt

np.set_printoptions(linewidth=np.inf)  # 打印不限制
global_salvation = {}
import time



class Data:
    def __init__(self):
        self.customer_num = 0
        self.node_num = 0
        self.range = 0
        self.launchtime = 0
        self.recovertime = 0

        self.x_coord = []
        self.y_coord = []

        self.x_coord_UAV = []
        self.y_coord_UAV = []

        self.x_coord_Truck = []
        self.y_coord_Truck = []

        self.C_UAV_lst = []
        self.danger = []
        self.IDlist = []
        self.crowdsourced = []

        self.OX = []
        self.OY = []
        self.DX = []
        self.DY = []

        self.demand = []
        self.servicetime = []
        self.readytime = []
        self.duetime = []
        self.dis_matrix = [[]]  # 比较怪的设置方法

    def readdata(self, path, node_num):
        data = Data()
        self.node_num = node_num
        data_red = pd.read_csv(path)
        for i in range(node_num):
            self.x_coord.append(int(data_red.loc[i, "XCOORD"]))
            self.y_coord.append(int(data_red.loc[i, "YCOORD"]))
            self.danger.append(float(data_red.loc[i, "DANGER"]))
            self.IDlist.append(int(data_red.loc[i, 'CUST NO'] - 1))
            if data_red.loc[i, "DRONE"] == 0:
                self.x_coord_UAV.append(int(data_red.loc[i, "XCOORD"]))
                self.y_coord_UAV.append(int(data_red.loc[i, "YCOORD"]))
                self.C_UAV_lst.append(int(data_red.loc[i, 'CUST NO'] - 1))
            else:
                self.x_coord_Truck.append(int(data_red.loc[i, "XCOORD"]))
                self.y_coord_Truck.append(int(data_red.loc[i, "YCOORD"]))

        # 在尾部添加虚拟点数据
        self.x_coord.append(int(data_red.loc[0, "XCOORD"]))
        self.y_coord.append(int(data_red.loc[0, "YCOORD"]))
        self.danger.append(float(data_red.loc[0, "DANGER"]))

        # 距离cost矩阵
        self.dis_matrix = np.zeros((self.node_num + 1, self.node_num + 1))
        self.dis_matrix_UAV = np.zeros((self.node_num + 1, self.node_num + 1))
        for i in range(0, self.node_num + 1):
            for j in range(0, self.node_num + 1):
                temp = (self.x_coord[i] - self.x_coord[j]) ** 2 + (self.y_coord[i] - self.y_coord[j]) ** 2
                self.dis_matrix_UAV[i][j] = round(temp ** 0.5, 2)
                self.dis_matrix[i][j] = abs(self.x_coord[i] - self.x_coord[j]) + abs(self.y_coord[i] - self.y_coord[j])

    def print_data(self, customer_num):
        print(self.IDlist)
        print(self.dis_matrix.shape)
        print("-------欧式距离矩阵-------\n")
        for i in range(self.node_num + 1):
            for j in range(self.node_num + 1):
                print("%6.2f" % (self.dis_matrix_UAV[i][j]), end=" ")
                # print("%6.2f" % (self.dis_matrix[i][j]), end=" ")
            print()
        print("-------曼哈顿距离矩阵-------\n")
        for i in range(self.node_num + 1):
            for j in range(self.node_num + 1):
                print("%6.2f" % (self.dis_matrix[i][j]), end=" ")
            print()

    def readrider(self, path, rider_num):
        data_red = pd.read_csv(path)
        for i in range(rider_num):
            self.crowdsourced.append(int(data_red.loc[i, 'RIDER NO']))
            self.OX.append(int(data_red.loc[i, "OX"]))
            self.OY.append(int(data_red.loc[i, "OY"]))
            self.DX.append(int(data_red.loc[i, "DX"]))
            self.DY.append(int(data_red.loc[i, "DY"]))
        print(self.crowdsourced)  # 骑手序列

    def divideup(self, path, num, cluster, K, epsilon, riderID):  # cluster簇数，K迭代次数上限。riderID从0开始取，
        # 聚类模块，得到rr个带0的序列
        TotalOrder = manupulate.kmeans(path, num, cluster, K, epsilon)
        ORDERforREGULARIDER = manupulate.remeans(path=path, TotalOrder=TotalOrder, cluster=2, K=K,
                                                 epsilon=epsilon)  # cluster是rr的数量，待参数化

        for i in range(2):
            if TotalOrder[i][0] != 0:
                ORDERforCROWDRIDER = TotalOrder[i]
        for i in range(len(ORDERforREGULARIDER)):
            ORDERforREGULARIDER[i].append(len(self.IDlist))  # 添加虚拟返回点

        print(TotalOrder)
        print(ORDERforREGULARIDER)
        print(ORDERforCROWDRIDER)
        print(self.C_UAV_lst)

        sorted_cr = sorted(ORDERforCROWDRIDER)  # 何意？

        # 给众包的点就不给可飞标记
        for i in ORDERforCROWDRIDER:
            if i in self.C_UAV_lst:
                self.C_UAV_lst.remove(i)
        # 首点加骑手序号（奇数点），尾点加骑手序号+1（偶数点）
        for i in self.crowdsourced:
            ORDERforCROWDRIDER.insert(0, i)
            sorted_cr.insert(0, i)
            ORDERforCROWDRIDER.append(i + 1)
            sorted_cr.append(i + 1)

            # 只有一个cs_rider的mini距离矩阵
            self.matrixWITHrider = np.zeros((len(sorted_cr), len(sorted_cr)))
            for i in sorted_cr:
                if i == sorted_cr[0]:  # 如果取首点，cs——rider的出发点
                    x1 = self.OX[riderID]
                    y1 = self.OY[riderID]
                    i = 0
                elif i == sorted_cr[-1]:
                    x1 = self.DX[riderID]
                    y1 = self.DY[riderID]
                    i = len(sorted_cr) - 1
                else:
                    x1 = self.x_coord[i]
                    y1 = self.y_coord[i]
                    i = sorted_cr.index(i)
                for j in sorted_cr:
                    # print(i, j)
                    if j == sorted_cr[0]:
                        x2 = self.OX[riderID]
                        y2 = self.OY[riderID]
                        j = 0
                    elif j == sorted_cr[-1]:
                        x2 = self.DX[riderID]
                        y2 = self.DY[riderID]
                        j = len(sorted_cr) - 1
                    else:
                        x2 = self.x_coord[j]
                        y2 = self.y_coord[j]
                        j = sorted_cr.index(j)
                    self.matrixWITHrider[i][j] = abs(x1 - x2) + abs(y1 - y2)
            print(self.matrixWITHrider)
        return ORDERforREGULARIDER, ORDERforCROWDRIDER, sorted_cr


# 骑手变速,UAV取恒定50迈
class GIVESPEED:
    def __init__(self):
        self.time_matrix = []
        self.time_matrix_UAV = []
        self.speed_matrix = []
        self.D = []

    def givespeed(self, data=None):
        self.time_matrix = np.zeros((data.node_num + 1, data.node_num + 1))
        self.time_matrix_UAV = np.zeros((data.node_num + 1, data.node_num + 1))
        self.speed_matrix = np.zeros((data.node_num + 1, data.node_num + 1))
        for i in range(0, data.node_num + 1):
            for j in range(0, data.node_num + 1):
                a = np.random.normal(loc=18, scale=8.65, size=1)  # 速度为正态，均值为18迈，标准差为8.65，生成1个数
                tem_speed = a[0]
                while tem_speed <= 1:
                    a = np.random.normal(loc=18, scale=8.65, size=1)  # 速度为正态，均值为18迈，标准差为8.65，生成1个数
                    tem_speed = a[0]
                tem_speed = round(float(tem_speed), 2)
                self.speed_matrix[i][j] = tem_speed

                self.time_matrix[i][j] = round(data.dis_matrix[i][j] / tem_speed, 2)
        self.time_matrix_UAV = 0.02 * data.dis_matrix_UAV  # 0.02是无人机速度50km/h的倒数，可调整
        print("-------用时矩阵-------\n")
        print(self.time_matrix)
        print("-------无人机用时矩阵-------\n")
        print(self.time_matrix_UAV)
        self.D = self.speed_matrix * data.dis_matrix
        row_sums = np.linalg.norm(self.D, ord=1, axis=1)
        data.danger = row_sums.tolist()
        print("------------------------------------------------------------------------------\n")
        print(data.danger)


# 计算去掉目标点后的单点时间
def timestamp(node, tagretnode, t, route, subRoute, subRoute_type):  # 去除会影响后续点的时序，前序点没关系,node为待去除点
    if route.index(tagretnode) > route.index(node):
        pre_node_index = route.index(node) - 1
        suc_node_index = route.index(node) + 1
        pre_node = route[pre_node_index]
        suc_node = route[suc_node_index]
        savings = calcSavings(node, route, subRoute,
                              subRoute_type)  # float(speed.time_matrix[pre_node][node] + speed.time_matrix[node][suc_node] - speed.time_matrix[pre_node][suc_node])
        if savings <= 0:
            print("发现华点")
        t_new_tagretnode = round(t[tagretnode] - savings, 2)
    if route.index(tagretnode) < route.index(node):
        t_new_tagretnode = t[tagretnode]
    # print(tagretnode)
    # print(t)

    return t_new_tagretnode


"待删点node的选择需要注意，否则+1-1需要考虑超限"


def calcSavings(node, route, subRoute, subRoute_type):  # 试图删去某个点以带来节省,以距离代替速度 #当待删点点在UAV相关的路线内，比较复杂
    # subRoute_type中，[0]的话是卡车路径，（i,j,k）则是无人机路径，可用此来判断子路径类型
    # 找前后紧邻点（先找索引再回推点）
    # 防止node为正数第一点或倒数第一点，或两端路的间隔点
    pre_node = node
    suc_node = node
    for clearsubroute in subRoute:
        if node in clearsubroute:
            pre_node_index = route.index(node) - 1
            suc_node_index = route.index(node) + 1
            pre_node = route[pre_node_index]
            suc_node = route[suc_node_index]  # 可能拿的是最后一个点而没有后续点，间隔点还会出现两次,但间隔点是会被删掉的啊，不应该去计算saving
            savings = speed.time_matrix[pre_node][node] + speed.time_matrix[node][suc_node] - \
                      speed.time_matrix[pre_node][suc_node]  # 一般情况下的计算方法
    for i in subRoute_type:
        if len(i) > 1 and node in subRoute[subRoute_type.index(i)]:
            launchnode = i[0]
            recovernode = i[-1]
            tagretnode = i[1]
            nodelist = subRoute[subRoute_type.index(i)]

            # 计算行程时间
            T_truck = 0
            for i in range(len(nodelist) - 1):
                start_id = nodelist[i]
                end_id = nodelist[i + 1]
                T_truck = T_truck + float(speed.time_matrix[start_id][end_id])
            T_UAV = speed.time_matrix_UAV[launchnode][tagretnode] + speed.time_matrix_UAV[tagretnode][recovernode]

            if T_truck <= T_UAV:
                savings = 0
                # print("车等机，无节省", node)
            else:
                # print("机等车，存在节省", node)
                referenceline = T_truck - T_UAV
                # print("飞机可冗余时间", referenceline)
                savings = min(referenceline, savings, )  # 缩短太厉害，变车等机，取前；缩短但还是机等车，取后
    if savings < 0:
        print("发现华点目标")

    return savings


"电量可以做文章，回车充电，能比标定里程更远"


def calcCostTruck(node, t, clearsubroute, savings, subRoute):  # 试图往用了UAV的卡车路线中插点，由卡车服务，会延长卡车与无人机的汇合时间
    tem_maxSavings = 0
    launchnode = clearsubroute[0]
    tagretnode = clearsubroute[1]
    recovernode = clearsubroute[-1]
    # 找到node原属子路径时需要特殊操作
    tem_clearsubroute = clearsubroute.copy()
    if node in tem_clearsubroute:
        tem_clearsubroute.remove(node)
    couple = [[0 for i in range(2)] for j in
              range(len(tem_clearsubroute) - 1)]  # [[0, 0], [0, 0], [0, 0], [0, 0]]#共n-1个空隙可插
    # 进行两两配对
    for i in range(len(tem_clearsubroute)):  # 首尾只会配对一次，中间点2次
        if i == 0:
            couple[i][0] = tem_clearsubroute[i]
        elif i == len(tem_clearsubroute) - 1:
            couple[-1][1] = tem_clearsubroute[i]
        else:
            couple[i][0] = tem_clearsubroute[i]
            couple[i - 1][1] = tem_clearsubroute[i]
    # print("配对完成", couple)
    # 尝试插点
    # 计算行程时间
    T_truck = 0
    for i in range(len(clearsubroute) - 1):
        start_id = clearsubroute[i]
        end_id = clearsubroute[i + 1]
        T_truck = T_truck + float(speed.time_matrix[start_id][end_id])
    T_UAV = speed.time_matrix_UAV[launchnode][tagretnode] + speed.time_matrix_UAV[tagretnode][recovernode]

    for i in couple:
        cost = speed.time_matrix[i[0]][node] + speed.time_matrix[node][i[1]] - speed.time_matrix[i[0]][i[1]]
        if T_truck + cost <= T_UAV:
            cost = 0
        elif T_truck + cost > T_UAV and T_truck < T_UAV:
            cost = T_UAV + cost - T_UAV
        elif T_truck > T_UAV:
            cost = cost
        if cost < savings:
            # if t[recovernode] - t[launchnode] + cost <= 当前电量
            if t[recovernode] - t[launchnode] + cost <= UAV_endurance:  # 有问题，地面路线拉长不一定同等拉长UAV使用时间
                if savings - cost > tem_maxSavings:
                    # 记录这个改变
                    launch_recover = [i[0], i[1]]
                    tem_maxSavings = savings - cost
            else:
                print("")
        else:
            print("cost更大", cost, savings)

    # print(tem_maxSavings)
    if tem_maxSavings <= 0:
        tem_maxSavings = 0
        launch_recover = [node, node]
    print(tem_maxSavings, launch_recover)
    return tem_maxSavings, launch_recover


def calcCostUAV(node, t, clearsubroute, savings, route, subRoute, subRoute_type,
                UAV_endurance):  # 试图往纯卡车路线插点，此点被UAV服务，纯卡车路线变，当来源为UAV相关的路线时，比较复杂
    "找到个移除node之后的点的时间，当node在UAV无关路线中，不影响，但在相关路线中需要准确计算t`[k]"
    tem_maxSavings = 0
    couple = []
    # 进行无序取两数配对

    tem_clearsubroute = clearsubroute.copy()
    if node in tem_clearsubroute:
        tem_clearsubroute.remove(node)

    for i in range(len(tem_clearsubroute)):
        for j in range(i + 1, len(tem_clearsubroute)):
            couple.append([tem_clearsubroute[i], tem_clearsubroute[j]])
    # print(couple)
    for i in couple:
        # print(i)
        if speed.time_matrix_UAV[i[0]][node] + speed.time_matrix_UAV[node][i[1]] <= UAV_endurance:
            t_launch = timestamp(node, i[0], t, route, subRoute, subRoute_type)
            # print("t_launch",t_launch)
            t_recover = timestamp(node, i[1], t, route, subRoute, subRoute_type)
            # print("t_recover",t_recover)
            item1 = speed.time_matrix_UAV[i[0]][node] + speed.time_matrix_UAV[node][i[1]]  # 飞总用时
            item2 = t_recover - t_launch  # 车总用时
            item = max(item1, item2) - (t_recover - t_launch)
            cost = max(0, item) + SL + SR
            # print("cost",cost)
            # print("savings",savings)
            if savings - cost > tem_maxSavings:
                # "记录这个改变"
                servedByUAV = True
                tem_maxSavings = savings - cost
                # print("maxSavings",maxSavings)
                launch_recover = [i[0], i[1]]
            else:
                pass
        else:
            print("两程距离超过电量限制", i, node)
    if tem_maxSavings <= 0:
        tem_maxSavings = 0
        launch_recover = [node, node]

    print(tem_maxSavings, launch_recover)
    return tem_maxSavings, launch_recover


def performUpdate(servedByUAV, node, launch_recover, t, Cprime, route, subRoute, clearsubroute, subRoute_type):
    # part1如果是UAV线的话
    if servedByUAV == True:
        # 更新将j从truckRoute中删除之后的t_new.
        print(")))))))1")
        # print(t)
        t_new = t.copy()
        # print(route)
        for k_id in route:
            if k_id != node:
                t_new_k = timestamp(node, k_id, t, route, subRoute, subRoute_type)
                # 计算从卡车路径中移除j，卡车到达某点k的时间
                t_new[k_id] = t_new_k
                # print(t_new)
            else:
                t_new_k = -1000
                t_new[k_id] = t_new_k

        route.remove(node)
        # 子路线割断
        j = subRoute.index(clearsubroute)
        if launch_recover[0] == clearsubroute[0] and launch_recover[1] == clearsubroute[-1]:  # 一段，同首尾
            subRoute_type[j] = [launch_recover[0], node, launch_recover[1]]

        if launch_recover[0] != clearsubroute[0] and launch_recover[1] == clearsubroute[-1]:  # 两段，同尾
            subRoute[j] = clearsubroute[clearsubroute.index(launch_recover[0]):]  # che
            subRoute.insert(j, clearsubroute[0:clearsubroute.index(launch_recover[0]) + 1])  # fei

            subRoute_type.insert(j + 1, [launch_recover[0], node, launch_recover[1]])

        if launch_recover[0] == clearsubroute[0] and launch_recover[1] != clearsubroute[-1]:  # 两段，同首
            subRoute[j] = clearsubroute[0:clearsubroute.index(launch_recover[1]) + 1]  # fei
            subRoute.insert(j + 1, clearsubroute[clearsubroute.index(launch_recover[1]):])  # che

            subRoute_type.insert(j, [launch_recover[0], node, launch_recover[1]])

        if launch_recover[0] != clearsubroute[0] and launch_recover[1] != clearsubroute[-1]:  # 三段，中间截开
            subRoute.insert(j, clearsubroute[clearsubroute.index(launch_recover[1]):])  # che
            subRoute.insert(j, clearsubroute[clearsubroute.index(launch_recover[0]):clearsubroute.index(
                launch_recover[1]) + 1])  # fei
            subRoute.insert(j, clearsubroute[0:clearsubroute.index(launch_recover[0]) + 1])  # che

            subRoute.remove(clearsubroute)

            subRoute_type.insert(j + 1, [0])
            subRoute_type.insert(j + 1, [launch_recover[0], node, launch_recover[1]])

        # 删subroute
        for i in subRoute:
            if node in i:
                i.remove(node)
                subRoute[subRoute.index(i)] = i

        # 更新cprime

        Cprime.remove(node)
        if launch_recover[0] in Cprime:
            Cprime.remove(launch_recover[0])
        if launch_recover[-1] in Cprime:
            Cprime.remove(launch_recover[-1])



    # part2 如果是卡车线的话
    elif servedByUAV == False:
        print(")))))))2")
        # 更新子路线subroute
        for i in range(0, len(subRoute)):
            if node in subRoute[i]:
                subRoute[i].remove(node)
            if (launch_recover[0] in subRoute[i]) and (launch_recover[1] in subRoute[i]):
                node_insert_index = subRoute[i].index(launch_recover[0]) + 1
                subRoute[i].insert(node_insert_index, node)

        # 更新路线route
        route.remove(node)
        route.insert(route.index(launch_recover[1]), node)

        # 更新时间,没有掺和UAV的影响，纯卡车的距离矩阵读出
        t_new = t.copy()
        cost = 0
        for i in range(0, len(route) - 1):
            start_id = route[i]
            end_id = route[i + 1]
            cost = cost + float(speed.time_matrix[start_id][end_id])
            t_new[end_id] = cost
        # 更新cprime
        Cprime.remove(node)

    # part3 时间对齐机制，也是t，t——new的更新原则，不是一变一更新，而是一变全部重新演算
    for i in subRoute_type:
        if len(i) > 1 and i[0] == launch_recover[0] and i[-1] == launch_recover[1]:
            launch = i[0]
            recover = i[-1]
            # print(t_new)
            # print(route)
            for m in route[route.index(launch):]:
                t_new[m] = t_new[m] + SL
            for n in route[route.index(recover):]:
                t_new[n] = t_new[n] + SR

            T_truck = t[recover] - t[launch]
            T_UAV = speed.time_matrix_UAV[launch][i[1]] + speed.time_matrix_UAV[i[1]][recover]  # 飞机行程时间
            if T_UAV > T_truck:
                delay = T_UAV - T_truck
                for l in route[route.index(recover):]:
                    t_new[l] = t_new[l] + delay

    return t_new, route, subRoute, subRoute_type, Cprime


# 一步sort，一二骑都可用
def sortdanger(orig_Cprime, danger, ORDERforREGULARIDER):
    cluster_num = len(ORDERforREGULARIDER)
    Cprime = [[] for i in range(cluster_num)]
    c = sorted(danger, reverse=True)
    # print("sortdanger")
    # print(ORDERforREGULARIDER)
    # print(orig_Cprime)
    for j in c:
        if danger.index(j) in orig_Cprime:
            for i in range(cluster_num):
                if danger.index(j) in ORDERforREGULARIDER[i]:
                    Cprime[i].append(danger.index(j))
    for sub in Cprime:
        sub.reverse()

    return Cprime


'对于【ijk】式路径上的黑点'


def cutdownblacknode_UAV(blacknode, route, subRoute, subRoute_type):  # blacknode是具体点,危险值点在UAV相关路线
    s = 0
    # 输入中转点，造中转点的矩阵
    # 定位黑点
    for clearsubroute in subRoute:
        if blacknode in clearsubroute:
            clearsubroute_index = subRoute.index(clearsubroute)
            break
    launchnode = subRoute_type[clearsubroute_index][0]
    targetnode = subRoute_type[clearsubroute_index][1]
    recovernode = subRoute_type[clearsubroute_index][-1]
    if speed.time_matrix_UAV[targetnode][blacknode] <= UAV_endurance:
        tem_subRoute = subRoute[clearsubroute_index + 1:]
        tem_subRoute_type = subRoute_type[clearsubroute_index + 1:]
        referencenode = subRoute[-1][-1]  # 初始化边界条件，尾点。覆盖情况为ijk后是一个长卡车线直至最后
        for i in tem_subRoute_type:
            if len(i) > 1:
                # print("找到边界飞行条件")
                referencenode = i[0]
                break
        # 构造待计算时间序列
        nodelist = route[route.index(launchnode) + 1: route.index(referencenode) + 1]  # 从起飞点的后续点计算到参考点,可能有问题
        nodelist.remove(blacknode)

        # print(nodelist)
        # UAV支
        anchor = speed.time_matrix_UAV[launchnode][targetnode] + speed.time_matrix_UAV[targetnode][blacknode]
        t_UAV = []
        for i in nodelist:
            j = float(anchor + speed.time_matrix_UAV[blacknode][i])
            t_UAV.append(j)

        # 卡车支
        t_Truck = []
        cost = 0
        start_id = launchnode
        for i in range(len(nodelist)):
            end_id = nodelist[i]
            cost = cost + float(speed.time_matrix[start_id][end_id])
            start_id = end_id
            t_Truck.append(cost)


    else:
        t_UAV = []
        t_Truck = []
        print("两程距离超电量限制，无法进行再捞服务", blacknode)
    # 寻找前一次飞起，阻止返程，尝试飞至中转点（考虑距离）
    # 尝试中转点飞至黑点（考虑距离）
    # 寻找落点，必须使其在下一次飞行前降落
    # cost为0的落点
    print("黑点t_UAV", t_UAV)
    print("黑点t_Truck", t_Truck)
    result = list(map(lambda x, y: x - y, t_Truck, t_UAV))
    gap = -999
    for i in result:
        if i > 0:  # 有cost为0的收点，需要进一步评估发点
            gap = i
            sele3recover = nodelist[result.index(i)]
            print("双飞后有收点", sele3recover, gap)
            s = calcSavings(blacknode, route, subRoute, subRoute_type)
            sele3 = [launchnode, targetnode, blacknode, sele3recover]
            break
        else:
            sele3 = None
            print("双飞后无收点", gap)

    return s, launchnode, sele3


"对于【0】式路径上的黑点"


def cutdownblacknode_Truck(blacknode, route, subRoute, subRoute_type):  # blacknode是具体点,危险值点在纯卡车路线
    tem_s = 0
    for clearsubroute in subRoute:
        if blacknode in clearsubroute:
            clearsubroute_index = subRoute.index(clearsubroute)
            break
    # 卡车路线可能被归到前一个UAV服务，也可能后一个，故分成两种情况
    pre_slice = subRoute_type[:clearsubroute_index]
    pre_referenceroute = None
    suc_referenceroute = None
    suc_slice = subRoute_type[clearsubroute_index + 1:]  # 切成两段，不含clearsubroute
    for i in pre_slice:
        if len(i) == 3:
            pre_referenceroute = i
            pre_referenceroute_index = pre_slice.index(i)
    for i in suc_slice:
        if len(i) == 3:
            suc_referenceroute = i  # "会被覆盖，和pre不一致"
            suc_referenceroute_index = len(pre_slice) + suc_slice.index(i) + 1
            break
    "电量也没考虑"
    # 前UAV服务，先tar再black，回收点可以重选
    if pre_referenceroute is not None:
        # print('pre_referenceroute', pre_referenceroute)
        launchnode = pre_referenceroute[0]
        targetnode = pre_referenceroute[1]
        anchor = speed.time_matrix_UAV[launchnode][targetnode] + speed.time_matrix_UAV[targetnode][blacknode]
        # 构造待计算时间序列
        if pre_referenceroute_index == clearsubroute_index - 1:
            nodelist = route[route.index(launchnode) + 1: route.index(clearsubroute[-1]) + 1]  # 只含尾点时为空列表
            nodelist.remove(blacknode)
        else:
            nodelist = subRoute[pre_referenceroute_index].copy()
            nodelist.remove(launchnode)
        # UAV支
        t_UAV = []
        for i in nodelist:
            j = float(anchor + speed.time_matrix_UAV[blacknode][i])
            t_UAV.append(j)
        # 卡车支
        t_Truck = []
        cost = 0
        start_id = launchnode
        for i in range(len(nodelist)):
            end_id = nodelist[i]
            cost = cost + float(speed.time_matrix[start_id][end_id])
            start_id = end_id
            t_Truck.append(cost)

        print("黑点t_UAV", t_UAV)
        print("黑点t_Truck", t_Truck)
        # "找同序号下，UAV等卡车，或差值最小，"
        pre_result = list(map(lambda x, y: x - y, t_Truck, t_UAV))
        pre_gap = -999
        for i in pre_result:
            if i > 0:  # 有cost为0的收点，需要进一步评估发点
                pre_gap = i
                sele3recover = nodelist[pre_result.index(i)]
                print("双飞后有收点（前序UAV服务）", sele3recover, pre_gap)
                sele3 = [launchnode, targetnode, blacknode, sele3recover]
                break
            else:
                sele3 = None
                print("双飞后无收点（前序UAV服务）", pre_gap)
        tem_s = calcSavings(blacknode, route, subRoute, subRoute_type)

    # 后UAV服务,先Black在tar，起飞点可以重选
    if suc_referenceroute is not None:
        # print('suc_referenceroute', suc_referenceroute)
        launchnode = suc_referenceroute[0]
        targetnode = suc_referenceroute[1]
        recovernode = suc_referenceroute[-1]
        anchor = speed.time_matrix_UAV[blacknode][targetnode] + speed.time_matrix_UAV[targetnode][recovernode]

        # 构造待计算时间序列
        if suc_referenceroute_index == clearsubroute_index + 1:
            nodelist = route[route.index(clearsubroute[0]):route.index(recovernode) + 1]  # 只含尾点时为空列表
            nodelist.remove(blacknode)
        else:
            nodelist = subRoute[suc_referenceroute_index].copy()
            print(nodelist)
            print(recovernode)
            print(blacknode)
            nodelist.remove(recovernode)

        # UAV支
        t_UAV = []
        for i in nodelist:
            j = float(anchor + speed.time_matrix_UAV[i][blacknode])
            t_UAV.append(j)
        # 卡车支,随序号增大，卡车行进路程越来越短
        t_Truck = []
        dune = nodelist.copy()
        dune.reverse()
        cost = 0
        start_id = recovernode
        for i in range(len(nodelist)):
            end_id = dune[i]
            cost = cost + float(speed.time_matrix[start_id][end_id])
            start_id = end_id
            t_Truck.append(cost)
        t_Truck.reverse()

        print("黑点t_UAV", t_UAV)
        print("黑点t_Truck", t_Truck)
        # "找同序号下，UAV等卡车，或差值最小，"
        suc_result = list(map(lambda x, y: x - y, t_Truck, t_UAV))
        suc_gap = -999
        for i in suc_result:
            if i > 0:  # 有cost为0的收点，需要进一步评估发点
                suc_gap = i
                sele3recover = nodelist[suc_result.index(i)]
                print("双飞后有发点（后续UAV服务）", sele3recover, suc_gap)
                sele3 = [sele3recover, targetnode, blacknode, recovernode]
                break
            else:
                sele3 = None
                print("双飞后无发点（后续UAV服务）", suc_gap)
        tem_s = calcSavings(blacknode, route, subRoute, subRoute_type)
    if (suc_referenceroute is None) and (pre_referenceroute is None):
        print("这条线前后就没有UAV可以依靠")
        launchnode = None
        sele3 = None
    return tem_s, launchnode, sele3  # 可能没值


def updateblacknode(blacknode, Cprime, sele3, t, route, subRoute, subRoute_type):  # blacknode是具体点
    for i in subRoute_type:
        if sele3[1] in i or sele3[2] in i:
            shuangfei_index = subRoute_type.index(i)
            break
    # route
    blacknode_index = route.index(blacknode)
    route.remove(blacknode)

    # subroute
    for i in subRoute:
        if blacknode in i:
            i.remove(blacknode)
            subRoute[subRoute.index(i)] = i

    # subRoute_type,ijk变ijlk
    subRoute_type[shuangfei_index] = sele3
    # 起飞首点相同
    if sele3[0] == subRoute_type[shuangfei_index][0]:
        if sele3[-1] in subRoute[shuangfei_index]:  # ijk内部即可消化这个黑点
            if sele3[-1] == subRoute[shuangfei_index][-1]:  # 原收点仍为收点
                pass
            else:  # 原夹板点变收点
                tem_slice = subRoute[shuangfei_index][subRoute[shuangfei_index].index(sele3[-1]):-2]
                subRoute[shuangfei_index] = subRoute[shuangfei_index][:subRoute[shuangfei_index].index(sele3[-1]) + 1]
                if len(subRoute_type[shuangfei_index + 1]) == 1:  ####可能没有这个index+1 的长度，index已经是最后一个
                    subRoute[shuangfei_index + 1][0:0] = tem_slice
                else:  # 两端飞行中间多出一段卡车
                    subRoute.insert(shuangfei_index + 1, tem_slice)
                    subRoute[shuangfei_index + 1].append(subRoute[shuangfei_index + 2][0])  # 补一个尾点
                    subRoute_type.insert(shuangfei_index + 1, [0])
        elif sele3[-1] == subRoute[shuangfei_index + 1][-1]:  # 如果回收点是后移把原0路占完，下一个子路全部并入这个
            print(shuangfei_index)
            print(subRoute)
            del subRoute_type[shuangfei_index + 1]  # 删除下一段子路
            subRoute[shuangfei_index].extend(subRoute[shuangfei_index + 1][1:])
            del subRoute[shuangfei_index + 1]
            print(subRoute_type)
            print(subRoute)





        else:  # 下一个子路还在，只部分并入
            subRoute_type[shuangfei_index] = sele3
            print(subRoute[shuangfei_index + 1])
            print(sele3[-1])
            print(subRoute[shuangfei_index + 1].index(sele3[-1]))
            tem_slice = subRoute[shuangfei_index + 1][1:subRoute[shuangfei_index + 1].index(sele3[-1]) + 1]
            subRoute[shuangfei_index] += tem_slice
            subRoute[shuangfei_index + 1] = subRoute[shuangfei_index + 1][
                                            subRoute[shuangfei_index + 1].index(sele3[-1]):]

    # 回收尾点相同
    elif sele3[-1] == subRoute_type[shuangfei_index][-1]:
        if sele3[0] == subRoute[shuangfei_index - 1][0]:  # 如果发出点是前移把原0路占完，上一个子路全部并入这个
            del subRoute_type[shuangfei_index - 1]  # 删除上一段子路
            subRoute[shuangfei_index - 1].extend(subRoute[shuangfei_index][1:])
            del subRoute[shuangfei_index]

        elif sele3[0] in subRoute[shuangfei_index]:  # ijk内部即可消化这个黑点
            if sele3[0] == subRoute[shuangfei_index][0]:
                pass
            else:  # 原夹板点变发点
                tem_slice = subRoute[shuangfei_index][subRoute[1:shuangfei_index].index(sele3[0]) + 1]
                subRoute[shuangfei_index] = subRoute[shuangfei_index][subRoute[shuangfei_index].index(sele3[0]):]
                if len(subRoute_type[shuangfei_index - 1]) == 1:  ####前本来即为卡车。可能没有这个index+1 的长度，index已经是第一个
                    subRoute[shuangfei_index - 1].extend(tem_slice)
                else:  # 两端飞行中间多出一段卡车
                    subRoute.insert(shuangfei_index, tem_slice)
                    subRoute[shuangfei_index].insert(0, subRoute[shuangfei_index - 1][-1])  # 补一个尾点
                    subRoute_type.insert(shuangfei_index, [0])


        else:  # 上一个子路还在，只部分并入

            subRoute_type[shuangfei_index] = sele3

            tem_slice = subRoute[shuangfei_index - 1][subRoute[shuangfei_index - 1].index(sele3[0]):-1]
            print(subRoute[shuangfei_index])
            print(sele3)
            subRoute[shuangfei_index].insert(0, tem_slice)
            subRoute[shuangfei_index - 1] = subRoute[shuangfei_index - 1][
                                            :subRoute[shuangfei_index - 1].index(sele3[0]) + 1]
            print(subRoute[shuangfei_index - 1])

    # 时间更新
    # if saving == 0:
    # t_new = t.copy()
    # else:
    # t_new = t.copy()
    # for i in route[blacknode_index:]:
    # t_new[i] -= saving
    # 没更新cprime
    Cprime.remove(blacknode)
    if sele3[0] in Cprime:
        Cprime.remove(sele3[0])
    if sele3[-1] in Cprime:
        Cprime.remove(sele3[-1])

    # 标准时间t——new的更新原则
    t_new = t.copy()
    cost = 0
    for i in range(0, len(route) - 1):
        start_id = route[i]
        end_id = route[i + 1]
        cost = cost + float(speed.time_matrix[start_id][end_id])
        t_new[end_id] = cost

    for i in subRoute_type:
        if len(i) > 1 and i[0] == sele3[0] and i[-1] == sele3[1]:
            launch = i[0]
            recover = i[-1]
            # print(t_new)
            # print(route)
            for m in route[route.index(launch):]:
                t_new[m] = t_new[m] + SL
            for n in route[route.index(recover):]:
                t_new[n] = t_new[n] + SR

            T_truck = t[recover] - t[launch]
            T_UAV = speed.time_matrix_UAV[launch][i[1]] + speed.time_matrix_UAV[i[1]][recover]  # 飞机行程时间
            if T_UAV > T_truck:
                delay = T_UAV - T_truck
                for l in route[route.index(recover):]:
                    t_new[l] = t_new[l] + delay
    print(subRoute_type)
    print(subRoute)

    return t_new, route, subRoute, subRoute_type, Cprime


def updatepart(blacknode, sele3, route, subRoute, subRoute_type):  # blacknode是具体点
    print()
    print(route,subRoute,subRoute_type)
    print(blacknode)
    for i in subRoute_type:
        if sele3[1] in i or sele3[2] in i:
            shuangfei_index = subRoute_type.index(i)
            break
    # route

    route.remove(blacknode)

    # subroute
    for i in subRoute:
        if blacknode in i:
            i.remove(blacknode)
            subRoute[subRoute.index(i)] = i

    # subRoute_type,ijk变ijlk
    subRoute_type[shuangfei_index] = sele3
    # 起飞首点相同
    if sele3[0] == subRoute_type[shuangfei_index][0]:
        if sele3[-1] in subRoute[shuangfei_index]:  # ijk内部即可消化这个黑点
            if sele3[-1] == subRoute[shuangfei_index][-1]:  # 原收点仍为收点
                pass
            else:  # 原夹板点变收点
                tem_slice = subRoute[shuangfei_index][subRoute[shuangfei_index].index(sele3[-1]):-2]
                subRoute[shuangfei_index] = subRoute[shuangfei_index][:subRoute[shuangfei_index].index(sele3[-1]) + 1]
                if len(subRoute_type[shuangfei_index + 1]) == 1:  ####可能没有这个index+1 的长度，index已经是最后一个
                    subRoute[shuangfei_index + 1][0:0] = tem_slice
                else:  # 两端飞行中间多出一段卡车
                    subRoute.insert(shuangfei_index + 1, tem_slice)
                    subRoute[shuangfei_index + 1].append(subRoute[shuangfei_index + 2][0])  # 补一个尾点
                    subRoute_type.insert(shuangfei_index + 1, [0])
        elif sele3[-1] == subRoute[shuangfei_index + 1][-1]:  # 如果回收点是后移把原0路占完，下一个子路全部并入这个
            del subRoute_type[shuangfei_index + 1]  # 删除下一段子路
            subRoute[shuangfei_index].extend(subRoute[shuangfei_index + 1][1:])
            del subRoute[shuangfei_index + 1]




        else:  # 下一个子路还在，只部分并入
            subRoute_type[shuangfei_index] = sele3

            tem_slice = subRoute[shuangfei_index + 1][1:subRoute[shuangfei_index + 1].index(sele3[-1]) + 1]
            subRoute[shuangfei_index] += tem_slice
            subRoute[shuangfei_index + 1] = subRoute[shuangfei_index + 1][
                                            subRoute[shuangfei_index + 1].index(sele3[-1]):]

    # 回收尾点相同
    elif sele3[-1] == subRoute_type[shuangfei_index][-1]:
        if sele3[0] == subRoute[shuangfei_index - 1][0]:  # 如果发出点是前移把原0路占完，上一个子路全部并入这个
            del subRoute_type[shuangfei_index - 1]  # 删除上一段子路
            subRoute[shuangfei_index - 1].extend(subRoute[shuangfei_index][1:])
            del subRoute[shuangfei_index]

        elif sele3[0] in subRoute[shuangfei_index]:  # ijk内部即可消化这个黑点
            if sele3[0] == subRoute[shuangfei_index][0]:
                pass
            else:  # 原夹板点变发点
                tem_slice = subRoute[shuangfei_index][subRoute[1:shuangfei_index].index(sele3[0]) + 1]
                subRoute[shuangfei_index] = subRoute[shuangfei_index][subRoute[shuangfei_index].index(sele3[0]):]
                if len(subRoute_type[shuangfei_index - 1]) == 1:  ####前本来即为卡车。可能没有这个index+1 的长度，index已经是第一个
                    subRoute[shuangfei_index - 1].extend(tem_slice)
                else:  # 两端飞行中间多出一段卡车
                    subRoute.insert(shuangfei_index, tem_slice)
                    subRoute[shuangfei_index].insert(0, subRoute[shuangfei_index - 1][-1])  # 补一个尾点
                    subRoute_type.insert(shuangfei_index, [0])


        else:  # 上一个子路还在，只部分并入

            subRoute_type[shuangfei_index] = sele3

            tem_slice = subRoute[shuangfei_index - 1][subRoute[shuangfei_index - 1].index(sele3[0]):-1]
            print(subRoute[shuangfei_index])
            print(sele3)
            subRoute[shuangfei_index].insert(0, tem_slice)
            subRoute[shuangfei_index - 1] = subRoute[shuangfei_index - 1][
                                            :subRoute[shuangfei_index - 1].index(sele3[0]) + 1]
            print(subRoute[shuangfei_index - 1])

    return route, subRoute, subRoute_type


def first2selec(t, Cprime, route):  # t是大t，Cprime一维，route从二骑的接收点开始截断
    global global_salvation
    maxSavings = 0
    t_new = []
    lastCprime = []
    subRoute = [route]
    subRoute_type = [[0]]
    servedByUAV = False
    SR = 0.05  # UAV回收时间
    SL = 0.05
    blacknode = []
    orig_t = t[-1]
    while Cprime != [] and Cprime != lastCprime:  # 会因为【11,2】和【2,11】陷入循环
        lastCprime = Cprime.copy()
        i = Cprime[0]
        print("______目标删去点______", i)
        print("可飞集合", Cprime)
        print(i, route, subRoute, subRoute_type)
        savings = calcSavings(i, route, subRoute, subRoute_type)
        if savings > 0:
            print("______拿出有节省，目标可删去______", savings)
            for clearsubroute in subRoute:
                print(subRoute)
                print("子路线类型", subRoute_type)
                print("目前子路线编号", subRoute.index(clearsubroute))
                if len(subRoute_type[subRoute.index(clearsubroute)]) > 1:  # and i not in subRoute_type:#备选子路线为飞机相关
                    print("_____该路线由卡车服务_______")
                    a, b = calcCostTruck(i, t, clearsubroute, savings, subRoute)
                    if a > maxSavings:
                        maxSavings = a
                        launch_recover = b
                        servedByUAV = False
                        clearsubroute_index = subRoute.index(clearsubroute)
                elif len(subRoute_type[subRoute.index(clearsubroute)]) == 1:  # 备选子路线为纯卡车
                    print("_____该路线预计由飞机服务_______")
                    a, b = calcCostUAV(i, t, clearsubroute, savings, route, subRoute, subRoute_type, UAV_endurance)
                    if a > maxSavings:
                        maxSavings = a
                        launch_recover = b
                        servedByUAV = True
                        clearsubroute_index = subRoute.index(clearsubroute)

            if maxSavings > 0:  # 找到一个有效变动
                global_salvation[i] = maxSavings
                maxSavings = 0
                print("改动相关点", launch_recover)
                clearsubroute = subRoute[clearsubroute_index]
                t_new, route, subRoute, subRoute_type, Cprime = performUpdate(servedByUAV, i, launch_recover, t, Cprime,
                                                                              route, subRoute, clearsubroute,
                                                                              subRoute_type)

                t = t_new.copy()
                print("likeastar")
                print(t_new, route, subRoute, subRoute_type, Cprime)

                if launch_recover[0] in blacknode:
                    blacknode.remove(launch_recover[0])
                    print("分割点删除", launch_recover[0])
                if launch_recover[1] in blacknode:
                    blacknode.remove(launch_recover[1])
                    print("分割点删除", launch_recover[1])
                # 需要避免重新计算分割点，最棘手情况，先进入blacknode，后变成分割点
                # 发生变动后，重新审时之前的高danger点
                if blacknode != []:
                    print("---------------------------------------成功进入二筛---------------------------------------")
                    print(blacknode)
                    for i in blacknode:
                        print("______二筛目标点______", i)
                        savings = calcSavings(i, route, subRoute, subRoute_type)
                        if savings > 0:
                            for clearsubroute in subRoute:
                                print(subRoute)
                                print("子路线类型", subRoute_type)
                                print("目前子路线编号", subRoute.index(clearsubroute))
                                if len(subRoute_type[subRoute.index(clearsubroute)]) > 1:  # 备选子路线为飞机相关
                                    print("_____该路线由卡车服务_______")
                                    a, b = calcCostTruck(i, t, clearsubroute, savings, subRoute)
                                    if a > maxSavings:
                                        maxSavings = a
                                        launch_recover = b
                                        servedByUAV = False
                                        clearsubroute_index = subRoute.index(clearsubroute)
                                elif len(subRoute_type[subRoute.index(clearsubroute)]) == 1:  # 备选子路线为纯卡车
                                    print("_____该路线由飞机服务_______")
                                    a, b = calcCostUAV(i, t, clearsubroute, savings, route, subRoute, subRoute_type,
                                                       UAV_endurance)
                                    if a > maxSavings:
                                        maxSavings = a
                                        launch_recover = b
                                        servedByUAV = True
                                        clearsubroute_index = subRoute.index(clearsubroute)
                            if maxSavings > 0:  # 找到一个有效变动
                                global_salvation[i] = maxSavings
                                print("二筛改动相关点", launch_recover)
                                maxSavings = 0
                                clearsubroute = subRoute[clearsubroute_index]
                                t_new, route, subRoute, subRoute_type, blacknode = performUpdate(servedByUAV, i,
                                                                                                 launch_recover, t,
                                                                                                 blacknode, route,
                                                                                                 subRoute,
                                                                                                 clearsubroute,
                                                                                                 subRoute_type)
                                t = t_new.copy()
                                if launch_recover[0] in Cprime:
                                    Cprime.remove(launch_recover[0])
                                if launch_recover[1] in Cprime:
                                    Cprime.remove(launch_recover[1])
                                if i in blacknode:
                                    blacknode.remove(i)
                                print("blacknode_removed", i)
                                print(t_new, route, subRoute, subRoute_type, blacknode)
                else:
                    print("无黑点，无须二筛")

            else:
                Cprime.remove(i)
                if i not in blacknode:
                    blacknode.append(i)
        else:
            print("______无节省______")
            Cprime.remove(i)
            if i not in blacknode:
                blacknode.append(i)
    print(t_new, route, subRoute, subRoute_type, blacknode, end="**** ")
    print()
    if t_new == []:
        t_new = t.copy()
    return t_new, route, subRoute, subRoute_type, blacknode


# 双人交接UAV
def get_transitnode(sele3, speed, res_rr, route, res_rr2, route2, UAV_endurance):
    print("一骑有黑点，可以操作")
    desionode = sele3[2]
    targetnode = sele3[1]
    Blaunchnode = sele3[0]
    Blaunchtime = res_rr[route.index(Blaunchnode)]
    endtime = Blaunchtime + speed.time_matrix_UAV[Blaunchnode][targetnode] + speed.time_matrix_UAV[targetnode][
        desionode]
    # 寻找从黑点最早可到二骑的点
    transitnode = None
    for index, value in enumerate(res_rr2):
        timeTOeachnode = endtime + speed.time_matrix_UAV[0][route2[index]]  # 飞机飞往各点的真实时刻
        if timeTOeachnode > value:
            print(route2[index], '太晚，已经服务完毕')
        else:  # 飞机转飞后来得及
            if speed.time_matrix_UAV[0][route2[index]] <= UAV_endurance:
                print(route2[index], "可飞")
                transitnode = route2[index]
                break
            else:
                print(route2[index], '太远，飞机续航不过来')

    return transitnode


def selection3(node, subRoute, route, subRoute_type):
    s = 0
    """
    三筛 的构造函数
    :param node: 具体某个黑点
    """
    # 要找最先可行的黑点，有待讨论(or最冗余)
    for clearsubroute in subRoute:
        if node in clearsubroute:
            clearsubroute_index = subRoute.index(clearsubroute)
            if len(subRoute_type[clearsubroute_index]) == 3:
                print(f"黑点{node}在单UAV路")
                tem_s, tem_Blaunchnode, sele3 = cutdownblacknode_UAV(node, route, subRoute,
                                                                     subRoute_type)  # sele3是个4点序列，ijlk
                if tem_s > s and sele3 != None:
                    s = tem_s
                    Blaunchnode = tem_Blaunchnode
                    tem_node = node
            elif len(subRoute_type[clearsubroute_index]) == 4:
                print(f"黑点{node}在双飞路，无法救")
            else:
                print(f"黑点{node}在卡车路")
                tem_s, tem_Blaunchnode, sele3 = cutdownblacknode_Truck(node, route, subRoute, subRoute_type)
                if tem_s > s and sele3 != None:
                    s = tem_s
                    Blaunchnode = tem_Blaunchnode
                    tem_node = node
    if ('tem_node' in locals().keys()) == True:
        print(f"黑点{tem_node}变动")
        chara = True
        referencenode = tem_node
    else:
        print("黑点无变动")
        chara = False
        Blaunchnode = None
        sele3 = None
        referencenode = None
    return s, chara, sele3, referencenode, Blaunchnode


def push(sele3_list, t_sele3, route, subRoute_type, t2, res_rr, res_rr2, route2, D2_Cprime, UAV_endurance, speed):
    """
           :param t2: 二骑的TSP时间，供给三筛
           :param t_sele3: 一骑三筛后时间
           :return: 求解之后的收敛值
           """
    delta = 0
    trip = []
    route_2 = route2.copy()
    route_f2 = route2.copy()
    subRoute_type_f2 = [[0]]
    subRoute_type2 = [[0]]
    subRoute2 = [route_2]
    # 一骑找点
    for sele3 in sele3_list:
        desionode = sele3[2]
        targetnode = sele3[1]
        Blaunchnode = sele3[0]
        for j in subRoute_type:
            if desionode in j:
                desionode_index = subRoute_type.index(j)
                break
        # 找不能被影响的点reference
        for j in subRoute_type[desionode_index:]:
            print(j)
            if len(j) > 1:
                reference = j[0]
                break
        if ('reference' in locals().keys()) == True:
            # 得对面的接收点

            transitnode = get_transitnode(sele3, speed, res_rr, route, res_rr2, route2, UAV_endurance)
            if transitnode == None:
                print("二骑收不到，没得飞")
            else:
                print(
                    "*****************************************情况一***************************************************",
                    sele3)
                # 二骑三次筛，找desionode和返一骑点
                # 二骑的C集合
                Cprime2 = []
                for i in route2[route2.index(transitnode) + 1:]:  # +1防止第一个点就去测试
                    if i in D2_Cprime[1]:
                        Cprime2.append(i)
                orig_Cprime2 = Cprime2.copy()
                # 二骑从tn后的片段双筛
                print("本真", t2)
                print(route2)
                print()
                # fake2为尾标，fake系列可能因不成而回退
                t2_C, route_part2, subRoute_part2, subRoute_type_part2, blacknode_part2 = first2selec(t2, Cprime2,
                                                                                                      route2[
                                                                                                      route2.index(
                                                                                                          transitnode):])

                route_part2C = route_part2.copy()

                # 二筛没变动，无法三筛，即二骑无desionnode
                if blacknode_part2 == orig_Cprime2:
                    print("二骑双筛无变动，没有UAV，用不了三筛")
                else:
                    # 真值复制
                    subRoute2_C = subRoute2.copy()
                    route2_C = route2.copy()
                    subRoute_type2_C = subRoute_type2.copy()
                    print()

                    # 二筛小更新
                    tem_subRoute_part2 = subRoute_part2.copy()
                    tem_subRoute_type_part2 = subRoute_type_part2.copy()
                    route2_C = route2_C[:route2_C.index(transitnode)] + route_part2
                    if subRoute_type_part2[0] != [0]:
                        subRoute2_C = [route2[:route2.index(transitnode)]] + tem_subRoute_part2
                        tem_subRoute_type_part2.insert(0, [0])
                    else:
                        tem_subRoute_part2[0] = route2[:route2.index(transitnode)] + tem_subRoute_part2[0]
                        subRoute2_C = tem_subRoute_part2.copy()
                    subRoute_type2_C = tem_subRoute_type_part2.copy()

                    print("二骑片段三筛")
                    # 二骑tn后的片段三筛
                    s = 0
                    chara = False
                    backTOone = False

                    for node in route_part2C:
                        if node in blacknode_part2:
                            s, T_chara, sele3_2, referencenode, Blaunchnode = selection3(node, subRoute_part2,
                                                                                         route_part2,
                                                                                         subRoute_type_part2)
                            if T_chara == True:
                                chara = True  # 调整全局变量
                                # 因为二骑三筛做一个片段的大更新
                                print("---------------------------------------假成", sele3_2)
                                t2_C, route2C, subRoute2_C, subRoute_type2_C, blacknode_part2 = updateblacknode(node,
                                                                                                                blacknode_part2,
                                                                                                                sele3_2,
                                                                                                                t2_C,
                                                                                                                route2_C,
                                                                                                                subRoute2_C,
                                                                                                                subRoute_type2_C)
                                # part更新，方便遍历下一个黑点
                                route_part2, subRoute_part2, subRoute_type_part2 = updatepart(node, sele3_2,
                                                                                              route_part2,
                                                                                              subRoute_part2,
                                                                                              subRoute_type_part2)

                    # 如果二骑有黑点（中转回来的可能）
                    if chara == True:
                        print("进入backTOone")
                        backTOone, duandian_all = backtoone(sele3, transitnode, reference, t_sele3, route,
                                                            t2_C, subRoute_type_part2)
                        # duandian_all是整个线路，分割点是duandian[0],起飞中转点是duandian[2]
                    if backTOone == True:  # bool
                        print("有来有回")
                        duandian = duandian_all[0]
                        # 记录路径
                        trip.append([desionode, transitnode])
                        trip.append([duandian_all[2], reference])
                        # 更新
                        # 取一个baseline时间
                        baseline = t2_C[duandian]

                        # 取到二骑的剩下点
                        ORDERforREST = route_f2[route_f2.index(duandian):]
                        for l in subRoute_type_part2:
                            if l[0] == duandian:
                                duandian_index = subRoute_type_f2.index(l)
                                break
                        for l in subRoute_type_part2[duandian_index:]:
                            if len(l) == 3:
                                ORDERforREST.insert(1, l[1])
                            if len(l) == 4:
                                ORDERforREST.insert(1, l[1])
                                ORDERforREST.insert(1, l[2])
                        # 做一个TSP
                        constrain = IPforTSP_rr.Constrain(holdinglist=ORDERforREST)
                        constrain.build_model(holdinglist=ORDERforREST, speed=speed, solve_model=True)
                        route_rest, accudis2 = IPforTSP_rr.get_solution(ORDERforREST, data, constrain.model)
                        t2 = [0 for i in range(route2[-1] + 1)]
                        # 把这个新的线代替二骑

                        # 更新route
                        route2 = route2_C[:route2_C.index(duandian)] + route_rest
                        route_2 = route2.copy()
                        # 更新type
                        tem = subRoute_type2_C[:subRoute_type2_C.index(duandian_all) + 1]
                        tem[-1] = duandian[:4]
                        subRoute_type2 = tem + [[0]]
                        # 更新subroute
                        tem_index = subRoute_type2_C.index(duandian_all)
                        subRoute2 = subRoute2_C[:tem_index] + [route_rest]
                        # 更新t
                        t2 = t2_C.copy()

                        # 距离转时间
                        cost2, t2[duandian] = baseline
                        for i in route_rest:
                            if i != route_rest[-1]:
                                start_id = i
                                end_id = route_rest[route_rest.index(i) + 1]
                                # print(start_id, end_id)
                                cost2 = cost2 + float(speed.time_matrix[start_id][end_id])
                                t2[end_id] = cost2
                        # 时间去占位0
                        res_rr2 = []
                        for i in route2:
                            res_rr2.append(t2[i])

        ######################################################################
        else:
            print("*****************************************情况二***************************************************")
            print("一骑后续无飞行需求，找到收点即可")
            delta = 0
            transitnode = get_transitnode(sele3, speed, res_rr, route, res_rr2, route2, UAV_endurance)
            if transitnode == None:
                print("二骑收不到，没得转")
            else:
                # 二骑的C集合
                Cprime2 = []
                for i in route2[route2.index(transitnode) + 1:]:  # +1防止第一个点就去测试
                    if i in D2_Cprime[1]:
                        Cprime2.append(i)
                orig_Cprime2 = Cprime2.copy()
                # 二骑从tn后的片段双筛
                print("前", t2)
                t2, route_part2, subRoute_part2, subRoute_type_part2, blacknode_part2 = first2selec(t2, Cprime2, route2[
                                                                                                                 route2.index(
                                                                                                                     transitnode):])

                route_part2C = route_part2.copy()

                # 二筛没变动，无法三筛，即二骑无desionnode
                if blacknode_part2 == orig_Cprime2:
                    print("二骑双筛无变动，无须飞，无须三筛")

                else:
                    trip.append([desionode, transitnode])
                    # 二筛小更新
                    tem_subRoute_part2 = subRoute_part2.copy()
                    tem_subRoute_type_part2 = subRoute_type_part2.copy()
                    route_2 = route2[:route2.index(transitnode)] + route_part2
                    if subRoute_type_part2[0] != [0]:
                        subRoute2 = [route2[:route2.index(transitnode)]] + tem_subRoute_part2
                        tem_subRoute_type_part2.insert(0, [0])
                    else:
                        tem_subRoute_part2[0] = route2[:route2.index(transitnode)] + tem_subRoute_part2[0]
                        subRoute2 = tem_subRoute_part2.copy()
                    subRoute_type2 = tem_subRoute_type_part2.copy()
                    # 二骑tn后的片段三筛
                    s = 0
                    for node in route_part2C:
                        if node in blacknode_part2:
                            s, T_chara, sele3_2, referencenode, Blaunchnode = selection3(node, subRoute_part2,
                                                                                         route_part2,
                                                                                         subRoute_type_part2)

                            if T_chara == True:
                                chara = True  # 调整全局变量
                                # 因为二骑三筛做一个片段的大更新
                                print("成了一组", sele3_2)
                                print(route_2, subRoute2, subRoute_type2)
                                print(t2)
                                t2, route_2, subRoute2, subRoute_type2, blacknode_part2 = updateblacknode(node,
                                                                                                          blacknode_part2,
                                                                                                          sele3_2,
                                                                                                          t2,
                                                                                                          route_2,
                                                                                                          subRoute2,
                                                                                                          subRoute_type2)
                                # part更新，方便遍历下一个黑点
                                route_part2, subRoute_part2, subRoute_type_part2 = updatepart(node, sele3_2,
                                                                                              route_part2,
                                                                                              subRoute_part2,
                                                                                              subRoute_type_part2)

    if trip == []:  # 三筛不行，继续考虑返仓
        print("********************************************情况三***************************************************")
        print("----------------------------------------三筛不行，继续考虑返仓--------------------------------------\n")
        subRoute_type2 = [[0]]
        transitnode = None
        endtime = res_rr[-1]  # 一骑结束时间,UAV开始转点时刻
        for index, value in enumerate(res_rr2):
            timeTOeachnode = endtime + speed.time_matrix_UAV[0][route2[index]]  # 飞机飞往各点的真实时刻
            if timeTOeachnode > value:
                print(route2[index], '太晚，已经服务完毕')
            else:  # 飞机转飞后来得及
                if speed.time_matrix_UAV[0][route2[index]] <= UAV_endurance and route2[index] != route2[-1]:
                    print(route2[index], "可飞")
                    transitnode = route2[index]
                    Cprime2 = []
                    for i in route2[route2.index(transitnode) + 1:]:  # +1防止第一个点就去测试
                        if i in D2_Cprime[1]:
                            Cprime2.append(i)
                    print(Cprime2)
                    orig_Cprime2 = Cprime2.copy()
                    # 二骑片段双筛
                    t2, route_part2, subRoute_part2, subRoute_type_part2, blacknode_part2 = first2selec(t2, Cprime2,
                                                                                                        route2[
                                                                                                        route2.index(
                                                                                                            transitnode):])

                    route_part2C = route_part2.copy()

                    print()
                    print(route_2)
                    print(route_part2)
                    print(subRoute_type_part2)

                    # 二筛没变动，无法三筛，即二骑无desionnode

                    print(blacknode_part2)
                    print(orig_Cprime2)
                    if blacknode_part2 == orig_Cprime2:
                        print("二骑双筛无变动，没有UAV，用不了三筛")
                    else:
                        print("可以三筛")
                        trip.append([0, transitnode])
                        print(blacknode_part2)
                        # cprime2会变空
                        # 二筛小更新
                        tem_subRoute_part2 = subRoute_part2.copy()
                        tem_subRoute_type_part2 = subRoute_type_part2.copy()
                        route_2 = route2[:route2.index(transitnode)] + route_part2
                        if subRoute_type_part2[0] != [0]:
                            subRoute2 = [route2[:route2.index(transitnode)]] + tem_subRoute_part2
                            tem_subRoute_type_part2.insert(0, [0])
                        else:
                            tem_subRoute_part2[0] = route2[:route2.index(transitnode)] + tem_subRoute_part2[0]
                            subRoute2 = tem_subRoute_part2.copy()
                        subRoute_type2 = tem_subRoute_type_part2.copy()
                        for node in route_part2C:
                            if node in blacknode_part2:
                                s, T_chara, sele3_2, referencenode, Blaunchnode = selection3(node, subRoute_part2,
                                                                                             route_part2,
                                                                                             subRoute_type_part2)
                                if T_chara == True:
                                    chara = True  # 全局判断是否变化
                                    print("变")
                                    print(route_2, subRoute2, subRoute_type2)

                                    t2, route_2, subRoute2, subRoute_type2, blacknode_part2 = updateblacknode(node,
                                                                                                              blacknode_part2,
                                                                                                              sele3_2,
                                                                                                              t2,
                                                                                                              route_2,
                                                                                                              subRoute2,
                                                                                                              subRoute_type2)
                                    # part更新，方便遍历下一个黑点
                                    route_part2, subRoute_part2, subRoute_type_part2 = updatepart(node, sele3_2,
                                                                                                  route_part2,
                                                                                                  subRoute_part2,
                                                                                                  subRoute_type_part2)
                    break
                else:
                    print(route2[index], '太远，飞机续航不过来')

    return delta, t2, trip, route_2, subRoute_type2


def backtoone(sele3, transitnode, reference, t_sele3, route, t_part2, subRoute_type_f2):
    # 寻找转点后的那个中转点
    backTOone = False
    duandian = False

    for k in subRoute_type_f2:  # 遍历二骑截断的type
        if len(k) == 4:  # 逢双飞即考察
            # 二骑UAV服务双点后的时间
            Blaunchnode = k[0]
            print(t_part2)
            Blaunchtime = t_part2[Blaunchnode]  # 更新好这个t，片段假成的时间，需要加上一个基础时间
            endtime = Blaunchtime + speed.time_matrix_UAV[Blaunchnode][k[1]] + speed.time_matrix_UAV[k[1]][k[2]]
            # 是否能在不影响一骑的下次飞行前返回
            for i in route[route.index(sele3[0]):route.index(reference) + 1]:
                timeTOnode = endtime + speed.time_matrix_UAV[k[2]][i]  # 飞机飞往各点的真实时刻
                if timeTOnode <= t_sele3[i] and speed.time_matrix_UAV[k[2]][i] <= UAV_endurance:
                    print("可飞")
                    backTOone = True
                    # 更新二骑，二骑的路线变了，用了一小段UAV

                    print(i, "TO", transitnode, "TO", k, "Back")
                    duandian = k
                    # 应该继续往二骑的下一个4点考察，不应该直接断掉
    return backTOone, duandian


def plot_final(route, route2, trip, subRoute_type, subRoute_type_f2):
    line_data_rider1_x = []
    line_data_rider1_y = []
    line_data_rider2_x = []
    line_data_rider2_y = []
    line_data_UAV_x = []
    line_data_UAV_y = []
    line_transit_x = []
    line_transit_y = []
    fig, ax = plt.subplots(1, 1)
    for i in route:
        line_data_rider1_x.append(data.x_coord[i])
        line_data_rider1_y.append(data.y_coord[i])
    for i in route2:
        line_data_rider2_x.append(data.x_coord[i])
        line_data_rider2_y.append(data.y_coord[i])
    for i in subRoute_type:
        if len(i) != 1:
            for j in i:
                line_data_UAV_x.append(data.x_coord[j])
                line_data_UAV_y.append(data.y_coord[j])
            line_1, = ax.plot(line_data_UAV_x, line_data_UAV_y, c='red', marker='o', markersize=5, linestyle='--',
                              linewidth=0.5)
            line_data_UAV_x = []
            line_data_UAV_y = []
    # 画出箭头
    for k in range(len(route) - 1):
        start1 = route[k]
        end1 = route[k + 1]
        plt.arrow(data.x_coord[start1], data.y_coord[start1], data.x_coord[end1] - data.x_coord[start1],
                  data.y_coord[end1] - data.y_coord[start1],
                  length_includes_head=True, head_width=0.8, head_length=0.5, lw=0.5, color='blue')

    for k in range(len(route2) - 1):
        start2 = route2[k]
        end2 = route2[k + 1]
        plt.arrow(data.x_coord[start2], data.y_coord[start2], data.x_coord[end2] - data.x_coord[start2],
                  data.y_coord[end2] - data.y_coord[start2], length_includes_head=True, head_width=0.8, head_length=0.5,
                  lw=0.5, color="#FF00FF")
    # line_2, = ax.plot(line_data_rider1_x, line_data_rider1_y, c='blue', marker='s', markersize=5, linewidth=1)
    # line_3, = ax.plot(line_data_rider2_x, line_data_rider2_y, c="#FF00FF", marker='s', markersize=5, linewidth=1)
    if trip != []:
        for i in trip:
            for j in i:
                line_transit_x.append(data.x_coord[j])
                line_transit_y.append(data.y_coord[j])
            line_4, = ax.plot(line_transit_x, line_transit_y, c="#008080", marker='+', markersize=8, linewidth=1,
                              zorder=10)
            line_transit_x = []
            line_transit_y = []
    print(subRoute_type_f2)
    for i in subRoute_type_f2:
        if len(i) != 1:
            for j in i:
                line_data_UAV_x.append(data.x_coord[j])
                line_data_UAV_y.append(data.y_coord[j])
            line_5, = ax.plot(line_data_UAV_x, line_data_UAV_y, c='#008080', marker='o', markersize=5, linestyle='--',
                              linewidth=0.5)
            line_data_UAV_x = []
            line_data_UAV_y = []
    plt.show()
    # ax.set_title()
    # ax.set_xlabel()
    # ax.set_ylabel()
start_time = time.time()


for _ in range(200):
    if __name__ == '__main__':
        data = Data()
        customer_num = 70
        path = 'data/01change.txt'
        UAV_endurance = 2
        data.readdata(path, customer_num)
        data.print_data(customer_num)
        data.readrider('data/rider.txt', 1)

        ORDERforREGULARIDER, ORDERforCROWDRIDER, sorted_cr = data.divideup('data/danger.txt', customer_num, cluster=2, K=20,
                                                                           epsilon=1e-5, riderID=0)

        print("ORDERforREGULARIDER",ORDERforREGULARIDER)
        speed = GIVESPEED()
        speed.givespeed(data=data)

        D = data.danger

        # 先拿rr二维分配结果中的第0项算
        constrain = IPforTSP_rr.Constrain(holdinglist=ORDERforREGULARIDER[0])
        constrain.build_model(holdinglist=ORDERforREGULARIDER[0], speed=speed, solve_model=True)
        route, accudis = IPforTSP_rr.get_solution(ORDERforREGULARIDER[0], data, constrain.model)

        orig_Cprime = data.C_UAV_lst  # 整体的可飞大表
        print("orig", orig_Cprime)
        danger = data.danger
        D2_Cprime = sortdanger(orig_Cprime, danger, ORDERforREGULARIDER)
        print(orig_Cprime)
        print("Cprime", D2_Cprime)
        Cprime = D2_Cprime[0]

        # 距离转时间
        t = [0 for i in range(route[-1] + 1)]
        cost = 0
        accu_D1 = 0
        for i in route:
            if i != route[-1]:
                start_id = i
                end_id = route[route.index(i) + 1]
                # print(start_id, end_id)
                cost = cost + float(speed.time_matrix[start_id][end_id])
                t[end_id] = cost
                accu_D1 = accu_D1 + data.danger[i]
                # accu_D1 = accu_D1 + float(D[start_id][end_id])
        print(t)
        print(route)

        maxSavings = 0
        t_new = []
        lastCprime = []
        subRoute = [route]
        subRoute_type = [[0]]
        servedByUAV = False
        SR = 0.05  # UAV回收时间
        SL = 0.05
        blacknode = []

        orig_t = t[-1]

        print("______准备工作完成______")

        while Cprime != [] and Cprime != lastCprime:  # 会因为【11,2】和【2,11】陷入循环
            lastCprime = Cprime.copy()
            i = Cprime[0]
            print("______目标删去点______", i)
            print("可飞集合", Cprime)
            savings = calcSavings(i, route, subRoute, subRoute_type)
            if savings > 0:
                print("______拿出有节省，目标可删去______")
                for clearsubroute in subRoute:
                    print(subRoute)
                    print("子路线类型", subRoute_type)
                    print("目前子路线编号", subRoute.index(clearsubroute))
                    if len(subRoute_type[subRoute.index(clearsubroute)]) > 1:  # and i not in subRoute_type:#备选子路线为飞机相关
                        print("_____该路线由卡车服务_______")
                        a, b = calcCostTruck(i, t, clearsubroute, savings, subRoute)
                        if a > maxSavings:
                            maxSavings = a
                            launch_recover = b
                            servedByUAV = False
                            clearsubroute_index = subRoute.index(clearsubroute)
                    elif len(subRoute_type[subRoute.index(clearsubroute)]) == 1:  # 备选子路线为纯卡车
                        print("_____该路线由飞机服务_______")
                        a, b = calcCostUAV(i, t, clearsubroute, savings, route, subRoute, subRoute_type, UAV_endurance)
                        if a > maxSavings:
                            maxSavings = a
                            launch_recover = b
                            servedByUAV = True
                            clearsubroute_index = subRoute.index(clearsubroute)

                if maxSavings > 0:  # 找到一个有效变动
                    maxSavings = 0
                    global_salvation[i] = maxSavings
                    print("改动相关点", launch_recover)
                    clearsubroute = subRoute[clearsubroute_index]
                    t_new, route, subRoute, subRoute_type, Cprime = performUpdate(servedByUAV, i, launch_recover, t, Cprime,
                                                                                  route, subRoute, clearsubroute,
                                                                                  subRoute_type)
                    t = t_new
                    if launch_recover[0] in blacknode:
                        blacknode.remove(launch_recover[0])
                    if launch_recover[1] in blacknode:
                        blacknode.remove(launch_recover[1])
                    print("likeastar")
                    print(t_new, route, subRoute, subRoute_type, Cprime)

                    # 需要避免重新计算分割点，最棘手，先进入blacknode，后变成分割点
                    # 发生变动后，重新审时之前的高danger点
                    if blacknode != []:
                        print("---------------------------------------成功进入二筛---------------------------------------")
                        for i in blacknode:
                            print("______二筛目标点______", i)
                            savings = calcSavings(i, route, subRoute, subRoute_type)
                            if savings > 0:
                                for clearsubroute in subRoute:
                                    print(subRoute)
                                    print("子路线类型", subRoute_type)
                                    print("目前子路线编号", subRoute.index(clearsubroute))
                                    if len(subRoute_type[subRoute.index(clearsubroute)]) > 1:  # 备选子路线为飞机相关
                                        print("_____该路线由卡车服务_______")
                                        a, b = calcCostTruck(i, t, clearsubroute, savings, subRoute)
                                        if a > maxSavings:
                                            maxSavings = a
                                            launch_recover = b
                                            servedByUAV = False
                                            clearsubroute_index = subRoute.index(clearsubroute)
                                    elif len(subRoute_type[subRoute.index(clearsubroute)]) == 1:  # 备选子路线为纯卡车
                                        print("_____该路线由飞机服务_______")
                                        a, b = calcCostUAV(i, t, clearsubroute, savings, route, subRoute, subRoute_type,
                                                           UAV_endurance)
                                        if a > maxSavings:
                                            maxSavings = a
                                            launch_recover = b
                                            servedByUAV = True
                                            clearsubroute_index = subRoute.index(clearsubroute)
                                if maxSavings > 0:  # 找到一个有效变动
                                    global_salvation[i] = maxSavings
                                    print("二筛改动相关点", launch_recover)
                                    maxSavings = 0
                                    clearsubroute = subRoute[clearsubroute_index]
                                    t_new, route, subRoute, subRoute_type, blacknode = performUpdate(servedByUAV, i,
                                                                                                     launch_recover, t,
                                                                                                     blacknode, route,
                                                                                                     subRoute,
                                                                                                     clearsubroute,
                                                                                                     subRoute_type)
                                    t = t_new
                                    if launch_recover[0] in Cprime:
                                        Cprime.remove(launch_recover[0])
                                    if launch_recover[1] in Cprime:
                                        Cprime.remove(launch_recover[1])
                                    if i in blacknode:
                                        blacknode.remove(i)
                                    print("blacknode_removed", i)
                                    print(t_new, route, subRoute, subRoute_type, blacknode)

                else:
                    Cprime.remove(i)
                    if i not in blacknode:
                        blacknode.append(i)
            else:
                print("______无节省______")
                Cprime.remove(i)
                if i not in blacknode:
                    blacknode.append(i)
        print(t_new, route, subRoute, subRoute_type, Cprime, end="**** ")
        print()
        print(orig_t)
        if t_new == []:
            t_new = t
        print("使用前两筛结果", t_new[-1])  # 这里可能为空，因为1.2筛后没有产生任何调整
        print("---------------------------------------三筛---------------------------------------\n")
        print("一骑三筛黑点", blacknode)
        print(subRoute_type)
        s = 0
        # 要找最先可行的黑点，有待讨论(or最冗余)
        print(route)
        routeFORsele3 = route.copy()
        sele3_list = []
        sele3_node = []
        t_sele3 = t.copy()
        chara = False
        for node in routeFORsele3:
            if node in blacknode:
                s, T_chara, sele3, referencenode, Blaunchnode = selection3(node, subRoute, route, subRoute_type)
                if T_chara == True:
                    chara = True  # 全局判断是否变化
                    t_sele3, route, subRoute, subRoute_type, blacknode = updateblacknode(node, blacknode, sele3, t_sele3,
                                                                                         route, subRoute,
                                                                                         subRoute_type)
                    global_salvation[node] = s
                    sele3_list.append(sele3)
                    sele3_node.append(node)
                    print(subRoute)
                    print(sele3)

        print(route)
        print(subRoute)
        print(subRoute_type)
        print("一骑未三筛时间轴", t_new)
        # print("三筛后", t_sele3[-1])
        print(sele3_list)
        after_D1 = 0
        for i in route:
            if i != route[-1]:
                start_id = i
                end_id = route[route.index(i) + 1]
                after_D1 = after_D1 + data.danger[i]
                # after_D1 = after_D1 + float(D[start_id][end_id])

        print("---------------------------------------第二线程(专送二骑TSP)---------------------------------------\n")
        constrain = IPforTSP_rr.Constrain(holdinglist=ORDERforREGULARIDER[1])
        constrain.build_model(holdinglist=ORDERforREGULARIDER[1], speed=speed, solve_model=True)
        route2, accudis2 = IPforTSP_rr.get_solution(ORDERforREGULARIDER[1], data, constrain.model)
        # 距离转时间
        t_2 = [0 for i in range(route2[-1] + 1)]
        cost2 = 0
        accu_D2 = 0

        for i in route2:
            if i != route2[-1]:
                start_id = i
                end_id = route2[route2.index(i) + 1]
                # print(start_id, end_id)
                cost2 = cost2 + float(speed.time_matrix[start_id][end_id])
                t_2[end_id] = cost2
                accu_D2 = accu_D2 + data.danger[i]
                # accu_D2 = accu_D2+ float(D[start_id][end_id])
        print(t_2)

        # 时间去占位0
        res_rr2 = []
        for i in route2:
            res_rr2.append(t_2[i])
        print("二骑时间轴", res_rr2)
        print(route2)
        res_rr = []
        for i in route:
            res_rr.append(t_new[i])
        print("一骑未三筛时间轴", res_rr)
        # print("（双车取大）未中转的最晚结束时间", max(res_rr[-1], res_rr2[-1]))

        # 更新t2,含0，可直接索引
        t2 = [0 for i in range(route2[-1] + 1)]
        for index, value in enumerate(route2):
            t2[value] = res_rr2[index]
        print(t2)
        orig_t2 = t2[-1]
        # 无黑点，只能到后交接
        if chara == False:
            print("一骑无黑点，只能先返仓\n")
            print("********************************************情况0***************************************************")
            trip = []
            subRoute_type2 = [[0]]
            transitnode = None
            endtime = res_rr[-1]  # 一骑结束时间,UAV开始转点时刻
            for index, value in enumerate(res_rr2):
                timeTOeachnode = endtime + speed.time_matrix_UAV[0][route2[index]]  # 飞机飞往各点的真实时刻
                if timeTOeachnode > value:
                    print(route2[index], '太晚，已经服务完毕')
                else:  # 飞机转飞后来得及
                    if speed.time_matrix_UAV[0][route2[index]] <= UAV_endurance and route2[index] != route2[-1]:
                        print(route2[index], "可飞")
                        transitnode = route2[index]

                        Cprime2 = []
                        for i in route2[route2.index(transitnode) + 1:]:  # +1防止第一个点就去测试
                            if i in D2_Cprime[1]:
                                Cprime2.append(i)
                        print(Cprime2)
                        orig_Cprime2 = Cprime2.copy()
                        # 二骑片段双筛
                        print(
                            "******************************************方块*************************************************")
                        print(route2)
                        print("qian", t2)
                        t2, route_part2, subRoute_part2, subRoute_type_part2, blacknode_part2 = first2selec(t2, Cprime2,
                                                                                                            route2[
                                                                                                            route2.index(
                                                                                                                transitnode):])

                        print()
                        print(route_part2)
                        print("hou", t2)
                        route_part2C = route_part2.copy()
                        print(subRoute_type_part2)
                        # 二筛没变动，无法三筛，即二骑无desionnode

                        print(blacknode_part2)
                        print(orig_Cprime2)
                        if blacknode_part2 == orig_Cprime2:
                            print("二骑双筛无变动，没有UAV，用不了三筛")
                        else:
                            trip.append([0, transitnode])
                            print("可以三筛")
                            print(blacknode_part2)
                            # cprime2会变空

                            # 二筛小更新
                            tem_subRoute_part2 = subRoute_part2.copy()
                            tem_subRoute_type_part2 = subRoute_type_part2.copy()
                            route2 = route2[:route2.index(transitnode)] + route_part2
                            if subRoute_type_part2[0] != [0]:
                                subRoute2 = [route2[:route2.index(transitnode)]] + tem_subRoute_part2
                                tem_subRoute_type_part2.insert(0, [0])
                            else:
                                tem_subRoute_part2[0] = route2[:route2.index(transitnode)] + tem_subRoute_part2[0]
                                subRoute2 = tem_subRoute_part2.copy()
                            subRoute_type2 = tem_subRoute_type_part2.copy()

                            for node in route_part2C:
                                if node in blacknode_part2:
                                    s, T_chara, sele3, referencenode, Blaunchnode = selection3(node, subRoute_part2,
                                                                                               route_part2,
                                                                                               subRoute_type_part2)
                                    if T_chara == True:
                                        chara = True  # 全局判断是否变化
                                        print(route2, subRoute2, subRoute_type2)
                                        # 全局更新，方便结果
                                        t2, route2, subRoute2, subRoute_type2, blacknode_part2 = updateblacknode(node,
                                                                                                                 blacknode_part2,
                                                                                                                 sele3, t2,
                                                                                                                 route2,
                                                                                                                 subRoute2,
                                                                                                                 subRoute_type2)
                                        # part更新，方便遍历下一个黑点
                                        route_part2, subRoute_part2, subRoute_type_part2 = updatepart(node, sele3,
                                                                                                      route_part2,
                                                                                                      subRoute_part2,
                                                                                                      subRoute_type_part2)

                                        # baseline = t2[transitnode]
                                        # for i in route_part2:
                                        # if i != route_part2[0]:
                                        # t_part2[i] = t_part2[i] + baseline

                        break
                    else:
                        print(route2[index], '太远，飞机续航不过来')
        else:

            print("----------------------------------------一骑有三筛，二骑跑断腿--------------------------------------\n")
            print("可能错误")
            print(sele3_list)
            print(subRoute_type)
            delta, aftert2, trip, route2, subRoute_type2 = push(sele3_list, t_sele3, route, subRoute_type, t2, res_rr,
                                                                res_rr2, route2, D2_Cprime, UAV_endurance, speed)

        print()
        if ("trip" in locals().keys()) == True:
            pass
        else:
            trip = []
        print('route', route)
        print(subRoute_type)
        print('trip', trip)
        print('route2', route2)

        plot_final(route, route2, trip, subRoute_type, subRoute_type2)

        if ("aftert2" in locals().keys()) == True:
            pass
        else:
            aftert2 = t2.copy()

        after_D2 = 0
        for i in route2:
            if i != route2[-1]:
                start_id = i
                end_id = route2[route2.index(i) + 1]
                after_D2 = after_D2 + data.danger[i]
                # after_D2 = after_D2 + float(D[start_id][end_id])

        print("一骑原时间", orig_t)
        print("一骑现时间", t_sele3[-1])
        print("二骑原时间", orig_t2)
        print("二骑现时间", aftert2[-1])
        print()
        print("一骑原D", accu_D1)
        print("一骑现D", after_D1)
        print("二骑原D", accu_D2)
        print("二骑现D", after_D2)
        results = [orig_t, t_sele3[-1], orig_t2, aftert2[-1], accu_D1, after_D1, accu_D2, after_D2]

        print(results)
        with open('26改.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            # 写入标题（可选）
            # writer.writerow(['一骑原时间', '一骑现时间', '二骑原时间', '二骑现时间'])
            writer.writerow(results)
        del aftert2
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time}秒")
