import numpy as np

'''
其实对偶只是分析框架，应用到具体算法还是内点法，惩罚因子法，admm等等
'''

# 定义目标函数和约束
def objective_function(x):
    return x**4 - 3 * x**3 + 2

def constraint(x):
    return x - 1  # 约束 g(x) <= 0

def lagrangian(x, lambda_):
    # 拉格朗日函数
    return objective_function(x) + lambda_ * constraint(x)

# 初始值
x = 0.5  # 初始原始变量
lambda_ = 1.0  # 初始对偶变量（乘子）
alpha_x = 0.01  # 原始变量更新步长
alpha_lambda = 0.01  # 对偶变量更新步长
tolerance = 1e-4  # 收敛阈值

# 迭代过程
for i in range(1000):
    # 更新原始变量 x：通过梯度下降最小化拉格朗日函数
    grad_x = 4 * x**3 - 9 * x**2 + lambda_  # 目标函数和约束的梯度
    x = x - alpha_x * grad_x

    # 更新对偶变量 lambda：通过梯度上升最大化拉格朗日函数
    lambda_ = max(0, lambda_ + alpha_lambda * constraint(x))  # 保证 lambda >= 0

    # 检查收敛条件
    if abs(grad_x) < tolerance and constraint(x) <= 0:
        print(f"Converged in {i+1} iterations.")
        break

# 输出结果
print("Optimal x:", x)
print("Optimal value of the objective function:", objective_function(x))
print("Dual variable (lambda):", lambda_)
