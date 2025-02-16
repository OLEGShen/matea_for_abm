# from model import Order,Rider
from dispatch import two_stage_fast_heuristic


class Order:
    def __init__(self, id, pickup_location, delivery_location, pickup_time, delivery_time):
        self.id = id
        self.pickup_location = pickup_location  # 取货地点，表示为一个坐标（例如，(0, 0)）
        self.delivery_location = delivery_location  # 送货地点，表示为一个坐标（例如，(1, 1)）
        self.pickup_time = pickup_time  # 取货时间窗口，表示为一个时间范围（例如，(0, 10)），表示骑手可以在这个时间范围内到达取货地点
        self.delivery_time = delivery_time  # 送货时间窗口，表示为一个时间范围（例如，(0, 20)），表示骑手可以在这个时间范围内到达送货地点
        self.status = 'unprocessed'


class Rider:
    def __init__(self, id, max_orders, location=(0, 0)):
        self.id = id
        self.route = []  # This will hold tuples of ('pickup' or 'delivery', order_id)
        self.order_count = 0  # 当前处理的订单数
        self.max_orders = max_orders  # 最多处理订单数
        self.location = location  # 骑手的当前位置
        self.check = 0

    def step(self):
        # 骑手送单的行为
        pass


def test_two_stage_fast_heuristic():
    # 模拟数据
    orders = [
        Order(0,(0, 0), (1, 1), (0, 10), (0, 20)),
        Order(1,(2, 2), (3, 3), (0, 15), (0, 25)),
        Order(2,(5, 6), (1, 1), (1, 6), (1, 25)),
        Order(3,(2, 3), (7, 1), (0, 10), (0, 20)),
        Order(4,(1, 8), (4, 3), (0, 15), (0, 25)),
        Order(5,(5, 2), (1, 9), (1, 20), (1, 30)),
        Order(6, (1, 8), (4, 3), (0, 15), (0, 25)),
        Order(7, (5, 2), (1, 9), (1, 20), (1, 30)),
    ]
    riders = [Rider(0, 3, location=(4, 1)), Rider(1, 4, location=(3, 2))]
    D = 100
    for i in range(4):
        orders[i].status = 'processed'
    riders[1].order_count = 2

    # 调用算法
    result = two_stage_fast_heuristic(orders, riders, D)
    print('-' * 20, result)
    for rider in riders:
        print(rider.route)
    print("rider0:", riders[0].route)
    print("rider1:", riders[1].route)

    # 断言
    assert result is not None  # 用于检查two_stage_fast_heuristic函数的返回值不是None。如果返回值为None，说明算法没有正确执行，或者出现了错误。
    assert isinstance(result, dict)  # 检查返回值是否为字典类型。two_stage_fast_heuristic函数应该返回一个字典，其中的键为骑手的索引，值为骑手的路径。
    for rider_index, route in result.items():
        assert isinstance(rider_index, int)  # 检查键是否为整数类型。骑手的索引应该是整数类型。
        assert isinstance(route, list)  # 检查值是否为列表类型。骑手的路径应该是一个列表。


if __name__ == "__main__":
    test_two_stage_fast_heuristic()
