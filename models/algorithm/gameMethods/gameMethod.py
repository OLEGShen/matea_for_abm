import math


def K31R(J, M, K, L, R, JA):
    result = JA  # 对应原代码中添加的记录自己旧值的操作，不过不太明确这里具体用途，先按此逻辑转换
    if M == 1:
        s = 0
    s = s + J
    a = s / M
    result = 1
    if a < 0.5:
        result = 0
    return result


def K32R(J, M, K, L, R, JA):
    result = JA  # 对应原代码中添加的记录自己旧值的操作，不过不太明确这里具体用途，先按此逻辑转换

    if M > 1:
        # 对应Fortran代码中的标号520开始的逻辑块
        c1 = 0
        c2 = 0
        c3 = 0
        c4 = 0
        j2 = 0
        j1 = 0
        i2 = 0
        i1 = 0
        p = 0
    else:
        if M == 2:
            # 对应Fortran代码中跳转到550的逻辑，先直接执行550相关赋值
            result = 0
            if j1 != J:
                # 对应Fortran代码中的标号570开始的逻辑块
                p = 0.6
                if J == 1:
                    p = 0.7
            elif j2 != j1:
                # 对应Fortran代码中的标号580的逻辑
                p = 0.9
            result = J
            if R < p:
                result = 1 - J

            # 对应Fortran代码中590标号处的变量更新逻辑
            j2 = j1
            j1 = J
            i2 = i1
            i1 = result
            return result

        # # OF HIS COOPS AFTER MY DEF.
        c1 = 0
        # # OF HIS DEFECTIONS AFTER MY DEF.
        c2 = 0
        # # OF HIS COOPS AFTER MY COOPERATION
        c3 = 0
        # # OF HIS DEF. AFTER MY COOPERATION
        c4 = 0
        # HIS 3RD PREV. CHOICE
        j2 = 0
        # HIS 2ND PREV. CHOICE
        j1 = 0
        # MY 2ND PREV. CHOICE
        i2 = 0
        # MY PREV. CHOICE
        i1 = 0
        # PROB. OF MY RESPONDING IN KIND
        p = 0

    # 对应Fortran代码中520标号之后进一步判断并执行的逻辑
    if M >= 27:
        if c1 < ((c1 + c2) - 1.5 * math.sqrt(c1 + c2)) / 2:
            return result
        if c4 < ((c3 + c4) - 1.5 * math.sqrt(c3 + c4)) / 2:
            return result
        result = 1
    else:
        if I2 == 0:
            if J == 0:
                c1 += 1
            else:
                c2 += 1
        else:
            if J == 0:
                c4 += 1
            else:
                c3 += 1

    if J1 != J:
        p = 0.6
        if J == 1:
            p = 0.7
    elif J2 != J1:
        p = 0.9
    result = J
    if R < p:
        result = 1 - J

    # 对应Fortran代码中590标号处的变量更新逻辑
    j2 = j1
    j1 = J
    i2 = i1
    i1 = result
    return result


def K33R(J, M, K, L, R, JA):
    # 对应Fortran中的逻辑型变量TWIN
    TWIN = True
    # 对应Fortran中的数组声明，这里使用Python列表来模拟
    COOP = [0] * 4
    COUNT = [0] * 4
    P = [0] * 4
    COEFF = [[36., 16., 0., 12., 0., 0.],
             [0., 12., 18., 12., 16., 0.],
             [0., 12., 24., 9., 16., 0.],
             [0., 0., 0., 9., 12., 48.]]
    CONST = [0., 4., 6., 6., 8., 12.]

    result = JA  # 对应原代码中添加的记录自己旧值的操作，不过不太明确这里具体用途，先按此逻辑转换

    if M > 1:
        # 对应Fortran代码中跳转到标号2的逻辑
        if M <= 2:
            # 对应Fortran代码中跳转到标号3的逻辑
            INDEX = 2 * int(LAST2) + int(LAST1) + 1
            if M == 1:
                # 对应Fortran代码中跳转到标号4的逻辑
                if J!= LAST1:
                    TWIN = False
                # 对应Fortran代码中跳转到标号24的逻辑
                if M <= 22:
                    if TWIN:
                        result = 0
                        LAST2 = LAST1
                        LAST1 = result
                        return result
                    else:
                        # 对应Fortran代码中计算最佳期望得分以及选择执行最佳策略的逻辑块
                        BEST = float('-inf')
                        for II in range(6):
                            SUM = CONST[II]
                            for JJ in range(4):
                                SUM += COEFF[II][JJ] * P[JJ]
                            if SUM <= BEST:
                                continue
                            BEST = SUM
                            IPOL = II
                        # 根据IPOL的值进行跳转执行相应策略
                        dispatch_dict = {
                            0: lambda: (30,),
                            1: lambda: (40, 30, 30, 30),
                            2: lambda: (40, 30, 40, 30),
                            3: lambda: (40, 40, 30, 30),
                            4: lambda: (40, 40, 40, 30),
                            5: lambda: (40, 40, 40, 40)
                        }
                        actions = dispatch_dict[IPOL]()
                        for action in actions:
                            if action == 30:
                                result = 0
                            elif action == 40:
                                result = 1
                            LAST2 = LAST1
                            LAST1 = result
                            return result
                else:
                    if TWIN:
                        result = 0
                    else:
                        result = 1
                    LAST2 = LAST1
                    LAST1 = result
                    return result
            else:
                # 更新相关概率估计的逻辑，对应Fortran代码中COOP和COUNT数组以及P列表的更新操作
                INDEX = 2 * int(LAST2) + int(LAST1) + 1
                COOP[INDEX] += 1 - J
                COUNT[INDEX] += 1
                P[INDEX] = COOP[INDEX] / COUNT[INDEX]
    else:
        # 对应Fortran代码中初始化所有状态变量的逻辑，在M <= 1时执行
        for JJ in range(4):
            COOP[JJ] = 0
            COUNT[JJ] = 0
        LAST1 = 1
        LAST2 = 1
        TWIN = True

    # 更新历史记录，对应Fortran代码中50标号处的逻辑
    LAST2 = LAST1
    LAST1 = result
    return result