import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pyro
import torch
import pickle

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pyro.distributions as dist

from torch import Tensor
from tqdm.auto import trange
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from pyro.distributions import constraints
from pyro.infer.autoguide import AutoDiagonalNormal
# from pyro.infer.autoguide import AutoBNAFNormal
from pyro.optim import ClippedAdam

from src.dgp_rff.DGPmodel import ModelDGP, ModelDGP_c_r
from src.dgp_rff.deep_layer import DeepGP, DeepGPNoBias, DeepEnsembleGP

from scipy.stats import shapiro

import torch
import time
torch.cuda.is_available()
torch.cuda.current_device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(device)
from src.makeplot import plot_output_predictions


def train_rfdgp(data,  depth=2, num_epochs=1000, lr = 0.001, j = 20, plot = False, plot_savepath=None):
    x_train, x_test, y_train, y_test = data
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # pyro.set_rng_seed(seed)
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    y_dim = y_train.shape[1]
    dim_list = [x_train.shape[1]] + [y_dim]*(depth-1) + [y_dim]
    J_list = [j] * (depth) 
    model1 = ModelDGP(dim_list=dim_list, J_list=J_list)
    # print("xtrainshape",x_train.shape)
    # print("ytrainshape",y_train.shape)
    model1 = model1.to('cuda').float()

    mean_field_guide = AutoDiagonalNormal(model1)
    # mean_field_guide = AutoBNAFNormal(model1, hidden_dim=100, num_blocks=2, max_plate_nesting=1)
    # optimizer = pyro.optim.Adam({"lr": 0.01})
    optimizer = pyro.optim.ClippedAdam({
        "lr": lr,      # 初始学习率
        "lrd": 0.9999999     # 每次迭代后学习率乘以 0.999
    })


    svi = SVI(model1, mean_field_guide, optimizer, loss=Trace_ELBO())
    pyro.clear_param_store()

    progress_bar = trange(num_epochs)
    l = 100
    losslist1 = []

    interval = max(num_epochs//l, 1)

    # for epoch in progress_bar:
    #     loss = svi.step(x_train, y_train)
    #     if epoch % interval == 0:
    #         losslist1.append(loss/ x_train.shape[0])
    #     progress_bar.set_postfix(loss=f"{loss / x_train.shape[0]:.3f}")
    best_loss = float('inf')
    patience = 500         # 容忍轮数
    counter = 0           # 没提升的连续轮数
    for epoch in progress_bar:
        loss = svi.step(x_train, y_train) / x_train.shape[0]
        if epoch % interval == 0:
            losslist1.append(loss)

        # Early stopping check
        # if loss < best_loss - 1e-4:  # 提升需超过一定阈值（避免浮动）
        #     best_loss = loss
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f"Early stopping triggered at epoch {epoch}. Best loss: {best_loss:.4f}")
        progress_bar.set_postfix(loss=f"{loss:.4f}")


    # plot result, get loss
    if plot:    
        # 生成对应的 epoch 索引
        epochs_shown = list(range(0, num_epochs, interval))
        if len(epochs_shown) > len(losslist1):  # 防止最后一个epoch未采样
            epochs_shown = epochs_shown[:len(losslist1)]

        # 绘图
        fig, ax = plt.subplots()
        ax.plot(epochs_shown, losslist1, marker='o', linestyle='-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Epochs')
        ax.grid(True)

        # 控制 xticks 间距
        max_xticks = 10
        if len(epochs_shown) > max_xticks:
            xtick_interval = max(len(epochs_shown) // max_xticks, 1)
            ax.set_xticks(epochs_shown[::xtick_interval])

        plt.tight_layout()
        os.makedirs(plot_savepath, exist_ok=True)
        fig.savefig(os.path.join(plot_savepath, 'loss.png'))
        plt.close(fig)

    predictive = Predictive(model1, guide=mean_field_guide, num_samples=50)
    preds= predictive(x_train)
    y_pred_train = preds['obs'].cpu().detach().numpy().mean(axis=0)
    y_std = preds['obs'].cpu().detach().numpy().std(axis=0)

    train_mse = ((y_pred_train - y_train.cpu().numpy()) ** 2).mean()


    preds= predictive(x_test)
    y_pred_test = preds['obs'].cpu().detach().numpy().mean(axis=0)
    y_std = preds['obs'].cpu().detach().numpy().std(axis=0)

    test_mse = ((y_pred_test - y_test.cpu().numpy()) ** 2).mean()

    if plot:
        plot_output_predictions(x_train.cpu().numpy(), y_train.cpu().numpy(), y_pred_train, name='Train', savepath=plot_savepath)
        plot_output_predictions(x_test.cpu().numpy(), y_test.cpu().numpy(), y_pred_test, name='Test', savepath=plot_savepath)


    # # # get residue, scale it
    # # preds_val = predictive(x_test)
    # # y_pred_val = preds_val['obs'].cpu().detach().numpy().mean(axis=0)
    # x_dim = x_test.shape[1]
    # if x_dim == 2:
    #     # 从预测分布中获取均值和方差
    #     y_pred_mean = y_pred_test[:, 0]
    #     y_true = y_test[:, 0].cpu().numpy()
    #     x_test_cpu = x_test.cpu().numpy()

    #     # 获取置信区间（假设你已有 variance）
    #     # pred_dist = preds['obs']
    #     mean = y_pred_mean
    #     std = y_std
    #     lower = mean - 2 * std
    #     upper = mean + 2 * std

    #     # 对 y 排序提升可视化质量
    #     sort_idx = np.argsort(y_true)
    #     new_x = x_test_cpu[sort_idx]
    #     new_y = y_true[sort_idx]
    #     new_mean = mean[sort_idx]
    #     new_lower = lower[sort_idx].squeeze()
    #     new_upper = upper[sort_idx].squeeze()

    #     # 绘图
    #     fig = plt.figure(figsize=(10, 6))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(new_x[:, 0], new_x[:, 1], new_y, s=2, c='k', label='True y')
    #     ax.scatter(new_x[:, 0], new_x[:, 1], new_mean, s=2, c='b', label='Predicted Mean')
    #     ax.plot_trisurf(new_x[:, 0], new_x[:, 1], new_lower, color='lightblue', alpha=0.3, label='Lower Bound')
    #     ax.plot_trisurf(new_x[:, 0], new_x[:, 1], new_upper, color='lightgreen', alpha=0.3, label='Upper Bound')
    #     ax.set_xlabel('x1')
    #     ax.set_ylabel('x2')
    #     ax.set_zlabel('y')
    #     ax.set_title("Prediction with 95% Confidence Interval")
    #     ax.legend()
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(plot_savepath, f'predicted_vs_true.png'))
    #     plt.close(fig)

    # if x_dim == 1:
    #     # x_dim == 1: 1D 可视化
    #     y_pred_mean = y_pred_test[:, 0]
    #     y_true = y_test[:, 0].cpu().numpy()
    #     x_test_cpu = x_test.cpu().numpy().squeeze()

    #     # 获取置信区间
    #     mean = y_pred_mean
    #     std = y_std[:, 0]
    #     lower = mean - 2 * std
    #     upper = mean + 2 * std

    #     # 排序以便画线
    #     sort_idx = np.argsort(x_test_cpu)
    #     x_sorted = x_test_cpu[sort_idx]
    #     y_true_sorted = y_true[sort_idx]
    #     mean_sorted = mean[sort_idx]
    #     lower_sorted = lower[sort_idx]
    #     upper_sorted = upper[sort_idx]

    #     fig, ax = plt.subplots(figsize=(8, 5))
    #     ax.plot(x_sorted, y_true_sorted, 'k.', label='True y', markersize=3)
    #     ax.plot(x_sorted, mean_sorted, 'b-', label='Predicted Mean')
    #     ax.fill_between(x_sorted, lower_sorted, upper_sorted, color='lightblue', alpha=0.5, label='95% CI')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_title("Prediction with 95% Confidence Interval")
    #     ax.legend()
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(plot_savepath, f'predicted_vs_true.png'))
    #     plt.close(fig)


    # for d in range(y_dim):
    #     # print(f"Output dimension {d}:")
    #     y_res = y_pred_test[:, d] - y_test[:, d].cpu().detach().numpy()
    #     y_res = y_res / y_std[:,d] # scale residue
    #     fig, ax = plt.subplots()
    #     ax.hist(y_res, bins=30, alpha=0.7, color='blue', edgecolor='black')
    #     ax.set_title(f'Residual Histogram for Output Dimension {d}')
    #     ax.set_xlabel('Scaled Residual')
    #     ax.set_ylabel('Frequency')
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(plot_savepath, f'residual_hist_dim{d}.png'))
    #     plt.close(fig)

    #     # # shapiro test
    #     # statG, p_valueG = shapiro(y_res)
    #     # print("Shapiro-Wilk Test Statistic:", statG)
    #     # print("p-value:", p_valueG)
    #     # # Check if the error for G is normally distributed
    #     # if p_valueG > 0.05:
    #     #     print("Normal (fail to reject H0).")
    #     # else:
    #     #     print("not Normal (reject H0).")
    #     # print('\n')
        
    return model1, train_mse, test_mse 
def train_rfdgp_multiclass(data, depth=2, num_epochs=1000, lr=1e-3, j=20,plot = False, plot_savepath=None):
    x_train, x_test, y_train, y_test = data
    num_classes = len(np.unique(y_train))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    x_test  = torch.tensor(x_test,  dtype=torch.float32, device=device)

    # 标签必须是 0..K-1 的 LongTensor；若是 one-hot，先转索引
    # print("y dim: ", y_train.shape[1])
    print("y classes:", num_classes)
    if isinstance(y_train, np.ndarray) and y_train.shape[1] > 1:
        print("change one-hot to index")
        y_train = np.argmax(y_train, axis=1)
        y_test  = np.argmax(y_test,  axis=1)
    
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test  = torch.tensor(y_test,  dtype=torch.long,  device=device)
    # print("y train shape:", y_train.shape)
    if y_train.shape[1] == 1:
        y_train = y_train.squeeze(1)
        y_test  = y_test.squeeze(1)

    # print(y_train.unique())

    # 让最后一层 = K（你的深度串联逻辑原来是用 y_dim 做中间宽度）
    dim_list = [x_train.shape[1]] + [10*num_classes]*(depth-1) + [num_classes]
    J_list = [j] * (depth)
    
    model = ModelDGP_c_r(dim_list=dim_list, J_list=J_list,
                     num_classes=num_classes, likelihood="classification").to(device)
    print(model)
    guide = AutoDiagonalNormal(model)
    optimizer = ClippedAdam({"lr": lr})#, "lrd": 0.9999999})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    pyro.clear_param_store()

    from tqdm.auto import trange
    pbar = trange(num_epochs)
    for epoch in pbar:
        loss = svi.step(x_train, y_train) / x_train.shape[0]
        pbar.set_postfix(loss=f"{loss:.4f}")

    # 预测：拿 logits 的后验平均 → softmax → 概率 → 取 argmax
    predictive = Predictive(model, guide=guide, num_samples=100, return_sites=["logits"])
    logits_te = predictive(x_test)["logits"].mean(0)     # [N, K]
    probs_te  = torch.softmax(logits_te, dim=-1)
    y_pred_te = probs_te.argmax(dim=-1)
    test_acc  = (y_pred_te == y_test).float().mean().item()

    logits_tr = predictive(x_train)["logits"].mean(0)
    probs_tr  = torch.softmax(logits_tr, dim=-1)
    y_pred_tr = probs_tr.argmax(dim=-1)
    train_acc = (y_pred_tr == y_train).float().mean().item()

    print(logits_tr)
    print(y_pred_tr)

    return model, train_acc, test_acc
