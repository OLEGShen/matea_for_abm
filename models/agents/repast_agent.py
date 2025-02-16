import random
from repast4py.core import Agent
import repast4py.space as space
from repast4py.core import Agent
from repast4py.space import DiscretePoint as dpt
from repast4py.space import ContinuousSpace
from repast4py.space import ContinuousPoint as cpt
from repast4py import context as ctx
from repast4py import schedule, logging



class Agent(Agent):
    """针对2d或者3d的栅格地图使用的agent

    Args:
        Agent (_type_): repast4py提供的agent，针对计算实验平台进行完善
    """
    def __init__(self,
                 id: int,
                 rank,
                 t,
                 pt
                 ):
        """_summary_

        Args:
            id (int): agent的编号
            rank (_type_): _description_
            t (_type_): _description_
            pt (tuple): 三元元组，分别是agent的初始的x y z 坐标值
        """
        super().__init__(id=id, rank=rank, type=t)
        self.direction = 0
        self.pt = pt

    # x轴向上
    def go_up(self):
        self.pt.x += 1
    
    # x轴向下
    def go_down(self):
        self.pt.x -= 1
    
    # y轴向下
    def go_left(self):
        self.pt.y -= 1
    
    # y轴向上
    def go_right(self):
        self.pt.x += 1

    def move_to_point(self, p):
        self.pt = dpt(p[0], p[1], p[2])



