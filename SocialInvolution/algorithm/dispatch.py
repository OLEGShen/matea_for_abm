# 目标函数是最小化订单超时时间和路径长度的和 f = ∑(max(0, T_i - t_i)) + ∑d_i
def objective(route, orders):
    total_delay = 0
    total_distance = 0
    current_time = 0
    current_location = (0, 0)  # 初始位置，根据实际情况可能需要修改

    order_dict = {order.id: order for order in orders}
    for event_type, order_id in route:
        order = order_dict[order_id]
        travel_time = distance(current_location, order.pickup_location if event_type == 'pickup' else order.delivery_location)
        current_time += travel_time  # 增加从上一个位置到当前位置的行驶时间
        total_distance += travel_time  # 增加到总距离

        # 对于取货事件，如果骑手提前到达，则等待
        if event_type == 'pickup':
            current_location = order.pickup_location
            if current_time < order.pickup_time[0]:
                current_time = order.pickup_time[0]
        else:  # 对于送货事件，计算延迟
            current_location = order.delivery_location
            if current_time > order.delivery_time[1]:
                delay = current_time - order.delivery_time[1]
                total_delay += delay

    return total_delay + total_distance



# 分层聚类算法 输入是所有的取送点的坐标，输出是每个取送点所属的集群
def hierarchical_clustering(locations, D):
    clusters = []
    for i, location in enumerate(locations):
        assigned = False
        for cluster in clusters:
            if min(distance(location, other) for other in cluster) < D:
                cluster.append(location)
                assigned = True
                break
        if not assigned:
            clusters.append([location])
    return clusters

def get_order_by_id(orders,id):
    for order in orders:
        if order.id == id:
            return order

# 添加聚类的贪婪插入初始化的算法
def greedy_insertion_with_clustering(orders, rider, D):
  # 初始化事件列表
    events = []
    for order in orders:
        events.append(('pickup', order.id))
        events.append(('delivery', order.id))

    # 将所有取货点和送货点的位置合并到一个列表中
    locations = [order.pickup_location for order in orders] + [order.delivery_location for order in orders]
    # 对这些点进行聚类
    clusters = hierarchical_clustering(locations, D)
    # 对事件进行排序，优先处理取货，且送货需要确保取货已在路线中
    sorted_events = sorted(events, key=lambda x:get_order_by_id(orders,x[1]).pickup_time[0] if x[0] == 'pickup' else float('inf'))
    # 初始化一个字典来跟踪订单的取送货状态
    order_status = {order.id: {'pickup': False, 'delivery': False} for order in orders}

    for event_type, order_id in sorted_events:
        order = next(o for o in orders if o.id == order_id)
        # 如果订单已处理或骑手订单数达到限制，则跳过
        if order.status == 'processed' or rider.order_count >= rider.max_orders:
            continue

        # 检查插入的位置是否合法，对于送货事件，还要检查取货事件是否已经在路线中
        valid_positions = []
        for i in range(len(rider.route) + 1):
            new_route = rider.route[:i] + [(event_type, order_id)] + rider.route[i:]
            if is_valid_route(new_route, orders):
                # 对于送货事件，确保其对应的取货事件已经在路线中
                if event_type == 'delivery' and not any(et == 'pickup' and oid == order_id for et, oid in new_route):
                    continue
                valid_positions.append((i, objective(new_route, orders)))

        # 如果没有有效位置，则跳过此订单
        if not valid_positions:
            continue

        # 选择目标函数值最小的位置
        best_position, best_value = min(valid_positions, key=lambda x: x[1])

        # 更新骑手的路线，订单状态和骑手的订单计数
        rider.route = rider.route[:best_position] + [(event_type, order_id)] + rider.route[best_position:]
        # 更新订单的取送货状态
        order_status[order_id][event_type] = True
        # 检查是否两个事件都完成了，如果是，则更新订单状态和骑手订单计数
        if all(order_status[order_id].values()):
            order.status = 'processed'
            rider.order_count += 1  # 完成一个完整的订单，骑手订单计数增加1
        print(f"Rider {rider.id}'s route after insertion: {rider.route}")



def is_valid_route(route, orders):
    # 这个函数现在会检查是否每个送货事件之前都有对应的取货事件
    picked_up_orders = set()
    for event_type, order_id in route:
        if event_type == 'pickup':
            picked_up_orders.add(order_id)
        else:  # delivery
            if order_id not in picked_up_orders:
                return False  # 如果送货事件没有对应的取货事件先行，则路径无效
            picked_up_orders.remove(order_id)
    return True


# 局部搜索 输入是初始解以及所有的订单，输出是经过优化后的解
def local_search(route, orders):
    best_route = route
    best_value = objective(route, orders)
    while True:
        improved = False
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                new_route = best_route[:i] + best_route[j:] + best_route[i:j]
                if is_valid_route(new_route, orders):
                    value = objective(new_route, orders)
                    if value < best_value:
                        best_route = new_route
                        best_value = value
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    return best_route


# 返回一个字典，其中的键为骑手的索引，值为骑手的运动坐标路径
def two_stage_fast_heuristic(orders, riders, D):
    # Assign orders to riders
    for rider in riders:
        greedy_insertion_with_clustering(orders, rider, D)
        rider.route = local_search(rider.route, orders)

    # Generate paths for riders based on the orders in their routes
    routes = {}
    for rider in riders:
        path = [rider.location]  # Start with rider's current location
        for action, order_id in rider.route:
            order = next((o for o in orders if o.id == order_id), None)
            if action == 'pickup':
                path.append(order.pickup_location)
            elif action == 'delivery':
                path.append(order.delivery_location)
        routes[rider.id] = path
    return routes


# 按照实际情况修改
def distance(location, other):
    x1, y1 = location
    x2, y2 = other
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5