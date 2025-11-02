import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.decomposition import PCA


def plot_output_predictions(x, y_true, y_pred, name='Train', savepath=None):
    """
    根据数据维度自动选择合适的预测可视化方式：
    - 1D输入 1D输出：曲线图
    - 1D输入 多输出：分输出曲线图 + PCA
    - 多维输入：默认不显示 x，仅用 y 显示每维变化 + PCA 降维
    """

    x = np.asarray(x)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    os.makedirs(savepath, exist_ok=True) if savepath else None

    x_dim = x.shape[1] if x.ndim == 2 else 1
    y_dim = y_true.shape[1] if y_true.ndim == 2 else 1

    # print("x dim = ", x_dim, ", y_dim = ", y_dim)
    # print(savepath)

#     # ==== Case 1: 1D input, 1D output ====
#     if x_dim == 1 and y_dim == 1:
#         idx_sorted = np.argsort(x[:, 0])
#         x_sorted = x[idx_sorted]
#         y_true_sorted = y_true[idx_sorted]
#         y_pred_sorted = y_pred[idx_sorted]

#         fig, ax = plt.subplots(figsize=(8, 6))
#         ax.plot(x_sorted, y_true_sorted, 'k*', label=f'{name} True')
#         ax.plot(x_sorted, y_pred_sorted, 'o-', label=f'{name} Pred', alpha=0.8)
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")
#         ax.set_title(f"{name} Prediction (1D Input/Output)")
#         ax.legend()
#         ax.grid(True)
#         plt.tight_layout()
#         if savepath:
#             fig.savefig(os.path.join(savepath, f'{name.lower()}_prediction.png'))
#         plt.close(fig)

#     # ==== Case 2: 1D input, multi-D output ====
#     elif x_dim == 1 and y_dim > 1:
#         for i in range(y_dim):
#             fig, ax = plt.subplots(figsize=(8, 5))
#             ax.plot(y_true[:, i], 'k*', label=f'{name} True y[{i}]')
#             ax.plot(y_pred[:, i], 'o', label=f'{name} Pred y[{i}]')
#             ax.set_xlabel("Sample Index")
#             ax.set_ylabel(f"y[{i}]")
#             ax.set_title(f"{name} Prediction y[{i}]")
#             ax.legend()
#             plt.tight_layout()
#             if savepath:
#                 fig.savefig(os.path.join(savepath, f"{name.lower()}_prediction_y{i}.png"))
#             plt.close(fig)

#         # 添加 PCA 降维图
#         _plot_pca(y_true, y_pred, name, savepath)

#     # ==== Case 3: Multi-D input ====
#     elif x_dim > 1:
#         # 默认只画 y_true vs y_pred
#         for i in range(y_dim):
#             fig, ax = plt.subplots(figsize=(8, 5))
#             ax.plot(y_true[:, i], 'k*', label=f'{name} True y[{i}]')
#             ax.plot(y_pred[:, i], 'o', label=f'{name} Pred y[{i}]')
#             ax.set_xlabel("Sample Index")
#             ax.set_ylabel(f"y[{i}]")
#             ax.set_title(f"{name} Prediction y[{i}]")
#             ax.legend()
#             plt.tight_layout()
#             if savepath:
#                 fig.savefig(os.path.join(savepath, f"{name.lower()}_prediction_y{i}.png"))
#             plt.close(fig)

#         # 添加 PCA 降维图
#         _plot_pca(y_true, y_pred, name, savepath)

# def _plot_pca(y_true, y_pred, name='Train', savepath=None):
#     """
#     内部函数：做 PCA 降维可视化。若 y 为 1D，则跳过。
#     """
#     if y_true.ndim < 2 or y_true.shape[1] < 2:
#         print(f"[PCA] Skipped PCA plot for {name}: only 1D output.")
#         return  # PCA 不适用于 1D 输出

#     pca = PCA(n_components=2)
#     combined = np.concatenate([y_true, y_pred], axis=0)
#     reduced = pca.fit_transform(combined)
#     n = len(y_true)

#     fig, ax = plt.subplots(figsize=(6, 5))
#     ax.scatter(reduced[:n, 0], reduced[:n, 1], c='black', label='True', alpha=0.6)
#     ax.scatter(reduced[n:, 0], reduced[n:, 1], c='red', label='Predicted', alpha=0.6)
#     ax.set_title(f"{name} PCA of Output Vectors")
#     ax.set_xlabel("PC 1")
#     ax.set_ylabel("PC 2")
#     ax.legend()
#     plt.tight_layout()
#     if savepath:
#         fig.savefig(os.path.join(savepath, f"{name.lower()}_pca_output.png"))
#     plt.close(fig)

