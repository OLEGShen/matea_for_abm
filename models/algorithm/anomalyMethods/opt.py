import numpy as np
from sko.GA import GA
from sko.SA import SA
from sko.DE import DE
from pyswarm import pso
from niapy.algorithms.basic import ArtificialBeeColonyAlgorithm, FireflyAlgorithm, BatAlgorithm, GreyWolfOptimizer, \
    FishSchoolSearch
from niapy.task import Task
from niapy.problems import Problem


# 定义目标函数
def polynomial(x):
    return sum(x ** 2) + 3 * x[0] + 2 * x[1] + x[2]


# 自定义问题类
class MyProblem(Problem):
    def __init__(self):
        super().__init__(dimension=5, lower=-10, upper=10)

    def _evaluate(self, x):
        return polynomial(x)


# 下界和上界
lower_bound = [-10, -10, -10, -10, -10]
upper_bound = [10, 10, 10, 10, 10]


# 遗传算法（Genetic Algorithm）
def genetic_algorithm():
    ga = GA(func=polynomial, n_dim=5, size_pop=50, max_iter=100, lb=lower_bound, ub=upper_bound, prob_mut=0.001)
    best_x, best_y = ga.run()
    print(f"Genetic Algorithm - Best Individual: {best_x}, Best Value: {best_y}")


# 模拟退火（Simulated Annealing）
def simulated_annealing():
    sa = SA(func=polynomial, x0=np.random.uniform(low=-10, high=10, size=5), T_max=100, T_min=1e-7, L=300,
            max_stay_counter=100)
    best_x, best_y = sa.run()
    print(f"Simulated Annealing - Best Individual: {best_x}, Best Value: {best_y}")


# 差分进化（Differential Evolution）
def differential_evolution():
    de = DE(func=polynomial, n_dim=5, size_pop=50, max_iter=100, lb=lower_bound, ub=upper_bound)
    best_x, best_y = de.run()
    print(f"Differential Evolution - Best Individual: {best_x}, Best Value: {best_y}")


# 粒子群优化（Particle Swarm Optimization）
def particle_swarm_optimization():
    best_x, best_y = pso(polynomial, lower_bound, upper_bound)
    print(f"Particle Swarm Optimization - Best Individual: {best_x}, Best Value: {best_y}")


# 人工蜂群算法（Artificial Bee Colony）
def artificial_bee_colony():
    problem = MyProblem()
    task = Task(problem, max_evals=10000)
    abc = ArtificialBeeColonyAlgorithm(population_size=50)
    best_x = abc.run(task)
    print(f"Artificial Bee Colony - Best Individual: {best_x[0]}, Best Value: {best_x[1]}")


# 萤火虫算法（Firefly Algorithm）
def firefly_algorithm():
    problem = MyProblem()
    task = Task(problem, max_evals=10000)
    fa = FireflyAlgorithm(population_size=50)
    best_x = fa.run(task)
    print(f"Firefly Algorithm - Best Individual: {best_x[0]}, Best Value: {best_x[1]}")


# 蝙蝠算法（Bat Algorithm）
def bat_algorithm():
    problem = MyProblem()
    task = Task(problem, max_evals=10000)
    ba = BatAlgorithm(population_size=50)
    best_x = ba.run(task)
    print(f"Bat Algorithm - Best Individual: {best_x[0]}, Best Value: {best_x[1]}")


# 灰狼优化算法（Grey Wolf Optimizer）
def grey_wolf_optimizer():
    problem = MyProblem()
    task = Task(problem, max_evals=10000)
    gwo = GreyWolfOptimizer(population_size=50)
    best_x = gwo.run(task)
    print(f"Grey Wolf Optimizer - Best Individual: {best_x[0]}, Best Value: {best_x[1]}")


# 鱼群算法（Fish Swarm Algorithm, FSA）
def fish_swarm_algorithm():
    problem = MyProblem()
    task = Task(problem, max_evals=10000)
    fsa = FishSchoolSearch(population_size=50)
    best_x = fsa.run(task)
    print(f"Fish Swarm Algorithm - Best Individual: {best_x[0]}, Best Value: {best_x[1]}")


# 蚁群算法
def mayi():
    # 初始化参数
    n_ants = 10  # 蚂蚁数量
    n_iterations = 500  # 迭代次数
    alpha = 1.0  # 信息素重要性
    beta = 2.0  # 启发式信息重要性
    rho = 0.5  # 信息素挥发率
    Q = 10  # 信息素常数

    # 定义搜索空间
    lower_bound = np.array([-10, -10, -10, -10, -10])
    upper_bound = np.array([10, 10, 10, 10, 10])
    dimension = len(lower_bound)

    # 初始化信息素
    pheromones = np.ones((n_ants, dimension))

    # 启发式信息初始化为1
    heuristic_info = np.ones((n_ants, dimension))

    # 存储每次迭代的最佳解
    best_solution = None
    best_value = float("inf")

    for iteration in range(n_iterations):
        solutions = []

        # 每只蚂蚁构建解决方案
        for i in range(n_ants):
            # 构造蚂蚁的解
            solution = []
            for j in range(dimension):
                # 根据信息素和启发式信息采样
                prob = pheromones[i, j] ** alpha * heuristic_info[i, j] ** beta
                if np.random.rand() < prob:
                    value = lower_bound[j] + np.random.rand() * (upper_bound[j] - lower_bound[j])
                else:
                    value = np.random.uniform(lower_bound[j], upper_bound[j])
                solution.append(value)

            # 计算目标函数值
            value = polynomial(np.array(solution))
            solutions.append((solution, value))

            # 更新最优解
            if value < best_value:
                best_solution = solution
                best_value = value

        # 信息素更新
        for i in range(n_ants):
            for j in range(dimension):
                pheromones[i, j] = (1 - rho) * pheromones[i, j] + Q / solutions[i][1]

        print(f"Iteration {iteration + 1}: Best Value = {best_value}")


if __name__ == "__main__":
    genetic_algorithm()
    simulated_annealing()
    differential_evolution()
    particle_swarm_optimization()
    artificial_bee_colony()
    firefly_algorithm()
    bat_algorithm()
    grey_wolf_optimizer()
    fish_swarm_algorithm()
    mayi()
