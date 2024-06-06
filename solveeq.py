from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

def debug_calculate_ri(positions, happen_position, happen_time,v) -> list[list[float]]:
    ri=[]
    for p in positions:
        distance = np.sqrt(sum([(happen_position[i]-p[i])**2 for i in range(3)]))
        t = happen_time + distance/v
        ri.append([p[0],p[1],p[2],t])
    return ri


# 定义方程组
def equations(r, ri, v) -> list[float]:
    def _eq(r, r0, v):
        x, y, z, t = r
        x0, y0, z0, t0 = r0
        return (x - x0)**2 + (y - y0)**2 + (z - z0)**2 - (v * (t - t0))**2
    return [_eq(r, r0, v) for r0 in ri]


def get_initial_bounds(ri, v) -> list[(float, float)]:
    zip_ri = list(zip(*ri))
    bounds = []
    max_distance=0
    # 设置位置的取值范围
    for i in range(3):
        delta = max(zip_ri[i]) - min(zip_ri[i])
        middle = (max(zip_ri[i]) + min(zip_ri[i])) / 2
        bounds.append((middle - delta-1, middle + delta+1))
        max_distance += delta
    # 设置时间的取值范围，注意只有可能比最早的记录时间更早
    bounds.append((min(zip_ri[3])-2*max_distance/v, min(zip_ri[3])-2))
    return bounds

def get_initial_pramas(ri, v) -> list[(float, float)]:
    # 方差的大小系数，越大则初始数据越集中
    sigma = [3,3,3,2]

    zip_ri = list(zip(*ri))
    pramas = []
    max_distance=0
    # 设置位置的取值平均数与方差
    for i in range(3):
        delta = max(zip_ri[i]) - min(zip_ri[i])
        middle = (max(zip_ri[i]) + min(zip_ri[i])) / 2
        pramas.append((middle, delta/sigma[i]))
        max_distance += delta
    middle_t = min(zip_ri[3])-max_distance/v
    delta_t = max_distance/v
    pramas.append((middle_t, delta_t/sigma[3]))

    return pramas

# 获取随机初始值
def get_initial_guess(ri, v,mode = 'normal') -> list[(float, float)]:
    if mode == 'uniform':
        bounds = get_initial_bounds(ri, v)
        return [np.random.uniform(low, high) for low, high in bounds]
    elif mode == 'normal':
        params = get_initial_pramas(ri, v)
        return [np.random.normal(middle, sigma) for middle, sigma in params]


def is_unique_solution(solutions, new_solution, tol=1e-6) -> bool:
    # 检查解的唯一性
    for sol in solutions:
        if np.allclose(sol, new_solution, atol=tol):
            return False
    return True


def is_valid_solution(solution, ri, v, tol=1e-6) -> bool:
    # 如果解有效的话，代回方程组应该全都是0
    val = equations(solution, ri, v)

    # 调试用，打印出所有的方程值
    # print(val)

    return not is_unique_solution([[0,0,0,0]], val, tol)


def is_reasonable_solution(solution, ri, v, tol=1e+1) -> bool:
    # 解的时间应该晚于所有记录的时间
    return solution[3] <= min(list(zip(*ri))[3])

# 获取所有可能解
def get_solutions(ri, v, max_attempts=50) -> list[list[float, float, float, float]]:
    solutions = []
    attempts = 0

    debug_invalid_count = 0  # 用于调试无效解的计数器
    debug_same_count = 0  # 用于调试重复解的计数器
    debug_unreasonable_count = 0  # 用于调试不合理解的计数器

    # 重复尝试寻找新的解，直到达到最大尝试次数
    while attempts < max_attempts:
        initial_guess = get_initial_guess(ri, v, 'normal')

        # 解方程组
        solution = fsolve(equations, initial_guess, args=(ri, v))

        # 调试用，打印出所有的解
        #print(solution)

        if not is_valid_solution(solution, ri, v):
            # 解错了就再解一遍
            debug_invalid_count += 1
            attempts += 0.1
        elif not is_reasonable_solution(solution, ri, v):
            # 解对了但是不合理，是舍去的根。
            debug_unreasonable_count += 1
            attempts += 1
        elif not is_unique_solution(solutions, solution):
            # 解对了但是重复了，是根。
            debug_same_count += 1
            attempts += 1
        else:
            # 解对了而且唯一，加进解集里。
            solutions.append(solution)
            attempts = 0

            # 调试用，打印出合理解的初始猜测值
            print(f"initial_bounds: {get_initial_bounds(ri, v)}")
            print(f"initial_pramas: {get_initial_pramas(ri, v)}")
            print(f"initial_guess: {initial_guess}")
    
        # 测试用，将所有找到的解都加进去，不管是否有效、重复、合理
        # solutions.append(solution) 
    
    print("无效解的数量:", debug_invalid_count)
    print("重复解的数量:", debug_same_count)
    print("不合理解的数量:", debug_unreasonable_count)

    return solutions


def plot_solutions(solutions) -> None:
    t_values = [sol[3] for sol in solutions]  # 提取 t 值
    # r_values = [np.sqrt(sol[0]**2 + sol[1]**2 + sol[2]**2) for sol in solutions]  # 计算 r 值
    r_values = [sol[2] for sol in solutions]  # 直接使用 z 值作为 r 值
    
    plt.figure(figsize=(10, 6))
    plt.scatter(t_values, r_values, c='blue', marker='o')
    plt.xlabel('t')
    # plt.ylabel('r = sqrt(x^2 + y^2 + z^2)')
    plt.ylabel('z')
    plt.title('Scatter Plot of Solutions')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    positions = [[0, 0, 0], [52446, 29038, -97], [45830.1840000004, 64643.8029999999, -82], [973.0400000005, 69094.3229999999, 26]]
    happen_position = [-50000,50000,200]
    happen_time = 10
    v = 340.0

    ri = debug_calculate_ri(positions, happen_position, happen_time,v)
    v = v

    solutions = get_solutions(ri, v)
    print("有效解的数量:", len(solutions))
    print("所有找到的解:")
    for sol in solutions:
        print(sol)
    plot_solutions(solutions)