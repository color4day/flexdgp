import numpy as np
from scipy import stats


def normality_test(data: np.ndarray,d):
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
        print(f"Column {d}: {test_name} Test -> Statistic: {stat:.4f}, p-value: {p:.4f}, Result: {result}")

    return results

#
# # Example usage
# data = np.random.normal(loc=0, scale=1, size=(6000, 5))
# normality_test(data)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def analyze2_gmm_components(data, max_components=5, plot=True, random_state=42):
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

    # if plot:
    #     plt.plot(range(1, max_components + 1), bic_scores, marker='o')
    #     plt.title('GMM: BIC vs Number of Components')
    #     plt.xlabel('Number of Components')
    #     plt.ylabel('BIC Score')
    #     plt.grid(True)
    #     plt.show()

    return best_n, bic_scores

from scipy import stats
import numpy as np

from scipy import stats

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
        dist_str = dist_name.item() if isinstance(dist_name, np.ndarray) else str(dist_name)
        dist = getattr(stats, dist_str)
        params = dist.fit(data)
        ks_stat, p_value = stats.kstest(data, dist_str, args=params)
        passed = p_value > alpha
        results.append({
            'distribution': dist_str,
            'p_value': p_value,
            'passed': passed
        })

    if sort:
        results.sort(key=lambda x: -x['p_value'])

    print(f"Alpha = {alpha}，p > alpha 表示“无法拒绝来自该分布”的假设\n")
    for r in results:
        mark = "T" if r['passed'] else "F"
        print(f"{r['distribution']}: p = {r['p_value']:.4f} {mark}")

    return results

def test2_common_distributions(data, distributions=None, alpha=0.05, sort=True):
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
    print(results)
    # print(f"Alpha = {alpha}，p > alpha 表示“无法拒绝来自该分布”的假设\n")
    # for r in results:
    #     mark = "✅" if r['passed'] else "❌"
    #     print(f"{str(r['distribution']):>8s}: p = {r['p_value']:.4f} {mark}")

    return results

import numpy as np
from scipy import stats

def fit_and_rank_distributions(data, distributions=None, alpha=0.05):
    """
    对数据进行多个常见分布拟合，返回 AIC/BIC 排名、K-S 检验结果等

    参数：
    - data: 一维 numpy array
    - distributions: 分布名列表（str），默认常见的
    - alpha: K-S 检验显著性水平（默认 0.05）

    返回：
    - results: 列表，每个元素是包含分布名、AIC、BIC、K-S p 值等的字典
    """
    data = np.ravel(data)  # 确保是一维

    if distributions is None:
        distributions = ['norm', 'laplace', 't', 'lognorm', 'expon']

    results = []
    n = len(data)

    for dist_name in distributions:
        try:
            dist = getattr(stats, dist_name)
            params = dist.fit(data)
            log_likelihood = np.sum(dist.logpdf(data, *params))
            k = len(params)

            aic = 2 * k - 2 * log_likelihood
            bic = np.log(n) * k - 2 * log_likelihood

            ks_stat, ks_p = stats.kstest(data, dist_name, args=params)

            results.append({
                'distribution': dist_name,
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood,
                'ks_p_value': ks_p,
                'ks_passed': ks_p > alpha
            })
        except Exception as e:
            print(f"⚠️  {dist_name} 拟合失败: {e}")
            continue

    results.sort(key=lambda r: r['bic'])  # 按 BIC 升序排列

    print("\n📊 分布拟合结果（按 BIC 排序）:")
    print(f"{'Dist':>10} | {'BIC':>10} | {'AIC':>10} | {'K-S p':>10} | {'K-S✅?'}")
    print("-" * 55)
    for r in results:
        mark = "✅" if r['ks_passed'] else "❌"
        print(f"{r['distribution']:>10} | {r['bic']:10.2f} | {r['aic']:10.2f} | {r['ks_p_value']:10.4f} | {mark}")

    return results




