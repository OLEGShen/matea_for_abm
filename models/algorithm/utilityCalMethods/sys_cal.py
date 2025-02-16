import math
class SysCal:
    def __init__(
                    self, 
                    agent_num=0, 
                    profit=0, 
                    Tf=10, 
                    Th=10, 
                    Rh=20, 
                    Tei=10, 
                    Rd=20
                ):
        self.num_gov = agent_num
        if profit is None:
            self.profit = 0
        else:
            self.profit = profit
        self.profit_list = []
        
        # 系统效能
        self.utility = 0
        self.utility_list = []

        # 公平性
        self.fairness = 0
        self.fairness_list = []
        # 公平性度量周期
        if Tf is None:
            self.Tf = 10
        else:
            self.Tf = Tf

        # 多样性
        self.variety = 0
        self.variety_list = []
        # 多样性度量周期
        if Th is None:
            self.Th = 10
        else:
            self.Th = Th
        # 多样性划分区间Rh
        if Rh is None:
            self.Rh = 20
        else:
            self.Rh = Rh

        # 熵增速率
        self.entropy_increase = 0
        self.entropy_increase_list = []
        # 熵增速率度量周期
        if Tei is None:
            self.Tei = 10
        else:
            self.Tei = Tei
        # 熵增速率划分区间Rd
        if Rd is None:
            self.Rd = 20
        else:
            self.Rd = Rd

        self.w1 = 1
        self.w2 = 1
        self.w3 = 1
        self.w4 = 1

    def update_profit(self, money):
        self.profit += money

    # 更新公平性
    def update_fairness(self, agents):
        self.fairness = 0
        x_gs = []
        for gov in agents: # 等待修改
            if len(gov.target.profit_list_present_time) <= self.Tf:
                self.fairness_list.append(self.fairness)
                return
            x_gk = 0
            len_present_time = len(gov.target.profit_list_present_time)
            for i in range(len_present_time - self.Tf, len_present_time):
                x_gk += gov.target.profit_list_present_time[i]
            x_gk /= self.Tf
            x_gs.append(x_gk)

        fz = 0
        for xg1 in x_gs:
            for xg2 in x_gs:
                fz += math.fabs(xg1 - xg2)
        sum_xg = 0
        for xg in x_gs:
            sum_xg += xg
        mu = 2 * self.num_gov * self.num_gov * sum_xg / len(x_gs)
        if mu == 0:
            self.fairness = 0  
        else:
            self.fairness = 1 - fz / mu
        self.fairness_list.append(self.fairness)

    # 更新多样性
    def update_variety(self, agents):
        self.variety = 0
        avg_profit_list = []
        for gov in agents:
            if len(gov.target.profit_list_present_time) <= self.Th:
                self.variety_list.append(self.variety)
                return
            avg_profit = 0
            len_present_time = len(gov.target.profit_list_present_time)
            for i in range(len_present_time - self.Th, len(gov.target.profit_list_present_time)):
                avg_profit += gov.target.profit_list_present_time[i]
            avg_profit /= self.Th
            avg_profit_list.append(avg_profit)

        rh_dict = {}
        for avg_profit in avg_profit_list:
            if str(int(avg_profit / self.Rh)) not in rh_dict:
                rh_dict[str(int(avg_profit / self.Rh)) ] = 1
            else:
                rh_dict[str(int(avg_profit / self.Rh))] += 1

        n = len(rh_dict)
        fz = 0
        for value in rh_dict.values():
            p = value / self.num_gov
            fz += p * math.log(p)
        if n == 1:
            self.variety = 1
        else:
            self.variety = -fz / math.log(n)
        self.variety_list.append(self.variety)

    # 更新熵增速率
    def update_entropy_increase(self, agents):
        self.entropy_increase = 0
        profit_list1 = []
        profit_list2 = []
        for gov in agents:
            if len(gov.target.profit_list_present_time) <= self.Tei:
                self.entropy_increase_list.append(self.entropy_increase)
                return
            len_present_time = len(gov.target.profit_list_present_time)
            profit_list1.append(gov.target.profit_list_present_time[len_present_time - 1])
            profit_list2.append(gov.target.profit_list_present_time[len_present_time - self.Tei])

        d1 = 1
        rd_dict = {}
        for profit in profit_list1:
            if str(int(profit / self.Rd)) not in rd_dict:
                rd_dict[str(int(profit / self.Rd)) ] = 1
            else:
                rd_dict[str(int(profit / self.Rd))] += 1
        for value in rd_dict.values():
            d1 -= (value / self.num_gov) * (value / self.num_gov)

        d2 = 1
        rd_dict = {}
        for profit in profit_list2:
            if str(int(profit / self.Rd)) not in rd_dict:
                rd_dict[str(int(profit / self.Rd))] = 1
            else:
                rd_dict[str(int(profit / self.Rd))] += 1
        for value in rd_dict.values():
            d2 -= (value / self.num_gov) * (value / self.num_gov)

        self.entropy_increase = 1 - math.fabs((d1 - d2) / self.Tei)
        self.entropy_increase_list.append(self.entropy_increase)

    # 更新效能
    def update_utility(self):
        self.utility = self.w1 * self.profit + self.w2 * self.fairness + \
                       self.w3 * self.variety + self.w4 * self.entropy_increase
        # self.utility = random.randint(100, 300)
        self.utility_list.append(self.utility)