import numpy as np
from scipy import stats


def normality_test(data: np.ndarray):
    """
    Perform normality tests on each column of the input array.
    For each column:
    - Use Shapiro-Wilk test if sample size < 5000.
    - Use Kolmogorov-Smirnov test if sample size >= 5000.

    Parameters:
    - data: m x n array

    Returns:
    - A dictionary with keys as column indices and values as tuples (test_name, test_statistic, p_value, result)
    """
    m, n = data.shape
    results = {}

    for col in range(n):
        column_data = data[:, col]
        if m < 5000:
            test_name = "Shapiro-Wilk"
            stat, p = stats.shapiro(column_data)
        else:
            test_name = "Kolmogorov-Smirnov"
            stat, p = stats.kstest(column_data, 'norm')

        result = "Normal" if p > 0.05 else "Not Normal"
        results[col] = (test_name, stat, p, result)

    # Print the results
    for col, (test_name, stat, p, result) in results.items():
        print(f"Column {col}: {test_name} Test -> Statistic: {stat:.4f}, p-value: {p:.4f}, Result: {result}")

    return results

#
# # Example usage
# data = np.random.normal(loc=0, scale=1, size=(6000, 5))
# normality_test(data)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def analyze_gmm_components(data, max_components=5, plot=True, random_state=42):
    """
    使用 BIC 分析最适合数据的高斯混合模型（GMM）组件数。

    参数：
    - data: 一维 numpy array
    - max_components: 最大尝试的 GMM 分量数（默认最多 5）
    - plot: 是否绘制 BIC 曲线图（默认绘制）
    - random_state: GMM 初始化随机种子

    返回：
    - best_n: 最佳分量数量（对应 BIC 最小值）
    - bic_scores: 所有尝试分量数对应的 BIC 分数
    """
    data = np.asarray(data).reshape(-1, 1)
    bic_scores = []

    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=random_state)
        gmm.fit(data)
        bic = gmm.bic(data)
        bic_scores.append(bic)
        print(f"GMM with {n} components: BIC = {bic:.2f}")

    best_n = np.argmin(bic_scores) + 1
    print(f"\n🔍 Best number of components: {best_n} (lowest BIC)")

    if plot:
        plt.plot(range(1, max_components + 1), bic_scores, marker='o')
        plt.title('GMM: BIC vs Number of Components')
        plt.xlabel('Number of Components')
        plt.ylabel('BIC Score')
        plt.grid(True)
        plt.show()

    return best_n, bic_scores

from scipy import stats
import numpy as np

def test_common_distributions(data, distributions=None, alpha=0.05, sort=True):
    """
    对输入数据进行常见分布拟合，并使用 K-S 检验进行检验。

    参数：
    - data: 一维 numpy array
    - distributions: 要尝试的分布列表，默认常见分布
    - alpha: 显著性水平（用于判断是否拒绝原假设，默认 0.05）
    - sort: 是否按 p 值降序输出

    返回：
    - results: 包含分布名、p 值、是否通过检验的字典列表
    """
    if distributions is None:
        distributions = ['norm', 'laplace', 't', 'expon', 'lognorm']

    results = []

    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        params = dist.fit(data)
        ks_stat, p_value = stats.kstest(data, dist_name, args=params)
        passed = p_value > alpha
        results.append({
            'distribution': dist_name,
            'p_value': p_value,
            'passed': passed
        })

    if sort:
        results.sort(key=lambda x: -x['p_value'])

    print(f"Alpha = {alpha}，p > alpha 表示“无法拒绝来自该分布”的假设\n")
    for r in results:
        mark = "✅" if r['passed'] else "❌"
        print(f"{r['distribution']:>8s}: p = {r['p_value']:.4f} {mark}")

    return results


def error_trace(test,pred,y_dim):
    """
    计算并绘制随着测试点数量增加的均方误差（error）曲线。

    参数：
    - test: 测试数据集
    - pred: 模型预测结果
    dim_i:要做第几个维度的
    """
    # 确保 test 和 pred 是 numpy 数组
    test = test.cpu().detach().numpy() if hasattr(test, 'cpu') else np.array(test)
    pred = pred.cpu().detach().numpy() if hasattr(pred, 'cpu') else np.array(pred)
    error = np.zeros(y_dim)
    y_dim = pred.shape[1] 
    error_trace = np.zeros((len(test),y_dim))

    # 逐步计算从前1个点到前N个点的 error
    for i in range(len(test)):
        error += ((pred[i,:] - test[i,:]) ** 2) # 多维输出时保持 axis=0
        error_trace[i] = error

    # 转换成 numpy 数组以便画图
    error_trace = np.array(error_trace)
    return error_trace
    # print(error_trace)
    # 如果是多维输出，可以画每一维的 error 曲线
def error_curve(error_trace,dim_i=0,layer = 0, summation = True):
    # plt.figure(figsize=(10, 5))
    if summation == True:
        plt.plot(error_trace.sum(axis = 1), label=f'sum of layer{layer}')
        
    else:
        for d in range(error_trace.shape[1]):
            plt.plot(error_trace[:, d], label=f"Dim {d}")

    plt.xlabel("Number of test points")
    plt.ylabel("Cumulative error")
    plt.title("error over growing test set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
