"""
读取csv文件中的地图信息，并且其中内置了许多有关地图感知的相关函数
同时包含了A*算法用来规划最短路径
"""
import pandas as pd
import sys
import math
import os

from random import randint
"""
读取文件中的地图信息，并存储为矩阵形式
矩阵中的1表示道路、0表示建筑。（后续需要对建筑信息进行扩展，详细信息记录在备注中）
需要对得到的矩阵进行转置，因为在矩阵中，row表示的是y坐标，col表示的是x坐标，转置之后进行遍历得到的坐标就和xy对应了起来
"""

# 读取地图信息
# file_map = 'map.csv'
current_directory = os.path.dirname(os.path.abspath(__file__))
file_map = os.path.join(current_directory, 'map.csv')
df_map = pd.read_csv(file_map, header=None)
map_matrix = df_map.values
# 对地图矩阵进行转置
map_matrix = map_matrix.transpose()

# 读取地图代号信息
# file_code = 'building_code.csv'
file_code = os.path.join(current_directory, 'building_code.csv')
df_code = pd.read_csv(file_code, header=None)
building_code = df_code.values


class ReadMap:
    def __init__(self):
        print("开始阅读地图信息...")
        self.code_dic = dict()
        self.door = [[(-1, -1)], [(-1, -1)]]
        self.code_to_dict()
        self.places = [
                        "加油站", "体育场", "服装店", "公司", "鞋店", "饭店", "礼品店", "医院", "工厂", "百货大楼",
                        "咖啡店", "音乐坊", "药房", "书店", "面包店", "水果店", "4s店", "教学楼", "学校食堂", "停车场", "洋房", "别墅",
                        "公园", "高层", "游乐园", "消防站", "银行", "酒店", "商场", "超市", "警察局"
                    ]
    def get_all_list(self):
        """
        返回所有的门口坐标

        Returns:
            list: 所有的位置
        """
        ans = []
        for door in self.places:
            ans += self.get_door(door)
        return ans 

    def code_to_dict(self):
        """
        1.将地图的代号信息转换为字典
        2.记录每个地点的大门在哪里
        """
        name = []
        code_num = []
        # 将地图的代号转换为字典
        for code in building_code:
            name.append(code[0])
            tmp_num = code[1:]
            len_num = len(tmp_num)
            for i in range(len_num):
                if not math.isnan(tmp_num[i]):
                    tmp_num[i] = int(tmp_num[i])
                else:
                    tmp_num = tmp_num[:i]
                    code_num.append(tmp_num)
                    break
            self.code_dic[code[0]] = tmp_num

        # 初始化门列表，需要获取一共有多少栋建筑物
        building_num = 0
        for key in self.code_dic:
            num = len(self.code_dic[key])
            building_num += num

        for i in range(building_num - 2):
            # 因为有的建筑不止一个门，所以采用列表的形式
            self.door.append([])

        # 将地图上每种建筑物的门确定
        for i in range(map_matrix.shape[0]):
            for j in range(map_matrix.shape[1]):
                if map_matrix[i][j] == 1:
                    self.door[1].append((i,j))
                    around = self.look_around(i, j)
                    for k in range(4):
                        if around[k] != 0 and around[k] != 1:
                            self.door[around[k]].append((i, j))
                            break


    def get_door(self, building_type):
        """根据给出的建筑类型寻找建筑的门的位置"""
        flag = 0
        door_list = []
        building_list = []
        for key in self.code_dic:
            if building_type == key:
                building_list = self.code_dic[key]
                flag = 1
                break

        if flag == 0:
            # 表示在字典中没有找到对应的建筑
            print("未找到对应建筑！")
            return

        for building in building_list:
            door_list.append(self.door[building])
        return door_list

    def look_around(self, current_x, current_y):
        """
        agent观察前后左右的环境，主要是用于确定道路以及建筑信息
        边界异常需要进行处理（捕获异常来解决，用-1来表示错误代码，即数组越界，该方向已到达地图边界）
        """
        try:
            map_matrix[current_x][current_y]
        except IndexError:
            print("********************坐标超出边界范围，请输入正确的坐标！********************")
            return [-1, -1, -1, -1]
        try:
            up = map_matrix[current_x][current_y - 1]
        except IndexError:
            up = -1

        try:
            down = map_matrix[current_x][current_y + 1]
        except IndexError:
            down = -1

        try:
            left = map_matrix[current_x - 1][current_y]
        except IndexError:
            left = -1

        try:
            right = map_matrix[current_x + 1][current_y]
        except IndexError:
            right = -1

        return [up, down, left, right]

    def go_up(self, current_x, current_y):
        """向上移动"""
        new_position = (current_x, current_y + 1)
        return new_position

    def go_down(self, current_x, current_y):
        """向下移动"""
        new_position = (current_x, current_y - 1)
        return new_position

    def go_left(self, current_x, current_y):
        """向左移动"""
        new_position = (current_x - 1, current_y)
        return new_position

    def go_right(self, current_x, current_y):
        """向右移动"""
        new_position = (current_x + 1, current_y)
        return new_position

    def get_position_type(self, current_x, current_y):
        """获得地图上某坐标点的类型，例如：马路、建筑等信息"""
        try:
            pos_type = map_matrix[current_x][current_y]
            return pos_type
        except IndexError:
            print("********************坐标超出边界范围，请输入正确的坐标！********************")

    def is_inbounds(self, current_x, current_y):
        """判断移动的目的地是否在边界内"""
        try:
            pos = map_matrix[current_x][current_y]
            return True
        except IndexError:
            return False

    def move_with_direction(self, move_type, current_x, current_y):
        """
        四个方位选择一个方位移动
        1: 前进
        2：后退
        3：左转
        4：右转
        """
        if move_type == 1:
            new_position = self.go_up(current_x, current_y)
        elif move_type == 2:
            new_position = self.go_down(current_x, current_y)
        elif move_type == 3:
            new_position = self.go_left(current_x, current_y)
        else:
            new_position = self.go_right(current_x, current_y)
        return new_position


# A*算法
class AStarPlanner:

    def __init__(self, start_x, start_y, goal_x, goal_y):
        self.open_list = []
        self.close_list = []
        self.node_start = self.Node(start_x, start_y)
        self.node_goal = self.Node(goal_x, goal_y)

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.cost = sys.maxsize
            self.parent = -1

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.parent)

    def BaseCost(self, p):
        x_dis = abs(self.node_start.x - p.x)
        y_dis = abs(self.node_start.y - p.y)
        return x_dis + y_dis

    def HeuristicCost(self, p):
        x_dis = abs(self.node_goal.x - p.x)
        y_dis = abs(self.node_goal.y - p.y)
        return x_dis + y_dis

    def TotalCost(self, p):
        return self.BaseCost(p) + self.HeuristicCost(p)

    def SelectPointInOpenList(self):
        index = 0
        selected_index = -1
        min_cost = sys.maxsize
        for p in self.open_list:
            cost = self.TotalCost(p)
            if cost < min_cost:
                min_cost = cost
                selected_index = index
            index += 1
        return selected_index

    # def isValidPoint(self, x, y):
    #     """判断是否是一个有效的点"""
    #     if x < 0 or y < 0:
    #         return False
    #     elif x > map_matrix.shape[0] or y > map_matrix.shape[1]:
    #         return False
    #     elif map_matrix[x][y] == 0:
    #         return False
    #     return True

    def isValidPoint(self, x, y):
        """判断是否是一个有效的点"""
        if x < 0 or y < 0:
            return False
        elif x >= map_matrix.shape[0] or y >= map_matrix.shape[1]:
            return False
        elif map_matrix[x][y] == 1:
            return True
        return False

    def IsInPointList(self, p, point_list):
        for point in point_list:
            if point.x == p.x and point.y == p.y:
                return True
        return False

    def IsInOpenList(self, p):
        return self.IsInPointList(p, self.open_list)

    def IsInCloseList(self, p):
        return self.IsInPointList(p, self.close_list)

    def IsStartPoint(self, p):
        return p.x == self.node_start.x and p.y == self.node_start.y

    def IsEndPoint(self, p):
        return p.x == self.node_goal.x and p.y == self.node_goal.y

    def ProcessPoint(self, x, y, parent):
        if not self.isValidPoint(x, y):
            return
        p = self.Node(x, y)
        if self.IsInCloseList(p):
            return
        if not self.IsInOpenList(p) and map_matrix[x][y] == 1:
            p.parent = parent
            # p.cost = self.TotalCost(p)
            p.cost = parent.cost + 1
            self.open_list.append(p)

    def BuildPath(self, p):
        path = []
        while True:
            path.insert(0, p)
            if self.IsStartPoint(p):
                break
            else:
                p = p.parent

        # data_base = SocietyDatabase("AI_society")
        insert_variables = "(x, y)"
        path_x = []
        path_y = []
        for p in path:
            # insert_data = (p.x, p.y)
            path_x.append(p.x)
            path_y.append(p.y)
            # data_base.insert_data("test_AStar", insert_variables, insert_data)
        # data_base.close_database()
        return path_x, path_y

    def planning(self):
        """
        A star 寻路算法
        """

        self.node_start.cost = 0
        self.open_list.append(self.node_start)

        while True:
            index = self.SelectPointInOpenList()
            if index < 0:
                print(self.node_start, self.node_goal)
                print(map_matrix[self.node_start.x][self.node_start.y])
                print('No path found, algorithm failed!!!', self.node_start, self.node_goal)
                return

            p = self.open_list[index]

            if self.IsEndPoint(p):
                path = self.BuildPath(p)
                result = list(zip(path[0], path[1]))
                return path

            del self.open_list[index]
            self.close_list.append(p)

            x = p.x
            y = p.y
            self.ProcessPoint(x - 1, y, p)
            self.ProcessPoint(x, y - 1, p)
            self.ProcessPoint(x + 1, y, p)
            self.ProcessPoint(x, y + 1, p)


# 测试代码
if __name__ == '__main__':
    a_star_test = AStarPlanner(148, 96, 336, 119)
    route = a_star_test.planning()
    print(route)
    read_map = ReadMap()
    door_list = read_map.get_door("洋房") + read_map.get_door("别墅") + read_map.get_door("饭店") + read_map.get_door("公司")
    ans = []
    for door in door_list:
        ans.append([door[0][0] - 58, door[0][1] - 97])
    print("door_list", ans)
    print(door_list[randint(0, 10)][0])
