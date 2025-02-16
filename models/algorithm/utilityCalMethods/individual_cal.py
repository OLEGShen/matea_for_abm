import math

class IndividualCal:
    def __init__(self, Tw=10, Tin=10, Ts=10):
        # 平稳度
        self.stability = 0
        self.stability_list = []
        # 稳健性
        self.robustness = 0
        self.robustness_list = []
        # 反演程度
        self.inv = 0
        self.inv_list = []

        # 效能
        self.utility = 0
        self.utility_list = []

        # 当前周期（unit_time）成本、收入、利益
        self.cost_present_time = 0
        self.income_present_time = 0
        self.profit_present_time = 0

        self.cost_list_present_time = []
        self.income_list_present_time = []
        self.profit_list_present_time = []

        # 平稳度度量周期数
        if Ts is None:
            self.Ts = 10
        else:
            self.Ts = Ts
        # 稳健性度量周期数
        if Tw is None:
            self.Tw = 10
        else:
            self.Tw = Tw
        # 反演程度度量周期数
        if Tin is None:
            self.Tin = 10
        else:
            self.Tin = Tin

        self.w1 = 1
        self.w2 = 1
        self.w3 = 1
        self.w4 = 1

    def update_stability(self):
        self.stability = 0
        if len(self.profit_list_present_time) <= self.Ts:
            self.stability_list.append(self.stability)
            return
        sum_profit = 0
        list_len = len(self.profit_list_present_time)
        for i in range(list_len - self.Ts, list_len):
            sum_profit += self.profit_list_present_time[i]
        mu = sum_profit / self.Ts

        sigma = 0
        for i in range(list_len - self.Ts, list_len):
            sigma += (self.profit_list_present_time[i] - mu) * (self.profit_list_present_time[i] - mu)

        sigma = math.sqrt(sigma / self.Ts)
        if sigma == 0:
            self.stability = 1
        else:
            self.stability = 1.0 / sigma
        self.stability_list.append(self.stability)

    def update_robustness(self, other_agents=[], sub_agents=[]):
        self.robustness = 0
        if len(self.profit_list_present_time) <= self.Tw:
            self.robustness_list.append(self.robustness)
            return

        # 计算z1
        self_a1 = 0
        list_len = len(self.profit_list_present_time)
        for i in range(list_len - self.Tw, list_len):
            self_a1 += self.profit_list_present_time[i]
        self_a1 /= self.Tw

        mu_list1 = [self_a1]
        for gov in other_agents:  # 修改
            other_sum = 0
            list_len = len(gov.profit_list_present_time)
            for i in range(list_len - gov.Tw, list_len):
                other_sum += gov.profit_list_present_time[i]
            mu_list1.append(other_sum / gov.Tw)

        mu1 = 0
        for mu in mu_list1:
            mu1 += mu
        mu1 /= 8

        sigma1 = 0
        for mu in mu_list1:
            sigma1 += (mu - mu1) * (mu - mu1)
        sigma1 /= 8
        sigma1 = math.sqrt(sigma1)

        if sigma1:
            z1 = (self_a1 - mu1) / sigma1
        else:
            z1 = 0

        # 计算z2
        # self_a2 = 0
        # agent_list = sub_agents
        # for agent in agent_list:
        #     self_a2 += agent.eff
        # self_a2 /= self.num_agent

        # mu_list2 = [self_a2]
        #暂时注释掉
        # for gov in other_agents:
        #     sum_eff = 0
        #     other_agent_list = gov.people_list + gov.robot_list
        #     for agent in other_agent_list:
        #         sum_eff += agent.eff
        #     mu_list2.append(sum_eff / gov.num_agent)

        # mu2 = 0
        # for mu in mu_list2:
        #     mu2 += mu
        # mu2 /= 8

        # sigma2 = 0
        # for mu in mu_list2:
        #     sigma2 += (mu - mu2) * (mu - mu2)
        # sigma2 /= 8
        # sigma2 = math.sqrt(sigma2)

        # if sigma2:
        #     z2 = (self_a2 - mu2) / sigma2
        # else:
        #     z2 = 0

        z2 = 0

        self.robustness = math.sqrt(z1 * z1 + z2 * z2)
        self.robustness_list.append(self.robustness)

    def update_inv(self):
        self.inv = 0
        if len(self.cost_list_present_time) <= self.Tin or len(self.income_list_present_time) <= self.Tin:
            self.inv_list.append(self.inv)
            return
        deltaR = 0
        deltaC = 0
        for income in self.income_list_present_time:
            deltaR += income
        for cost in self.cost_list_present_time:
            deltaC += cost

        if deltaC == 0:
            self.inv = 1
        else:
            self.inv = deltaR / deltaC
        self.inv_list.append(self.inv)

    def update_utility(self):
        self.utility = 0
        R_max = 0
        C_min = 1e9
        for income in self.income_list_present_time:
            R_max = max(R_max, income)
        for cost in self.cost_list_present_time:
            C_min = min(C_min, cost)
        if self.profit_present_time == 0:
            f1 = 0
        else:    
            f1 = self.profit_present_time / (R_max - C_min)
        if self.stability >= 1:
            f2 = 1
        else:
            f2 = self.stability
        f3 = 1 / (1 + math.exp(-self.inv))

        self.utility = self.w1 * f1 + self.w2 * f2 + self.w3 * self.robustness + self.w4 * f3
        self.utility_list.append(self.utility)