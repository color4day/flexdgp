import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from rich.progress import Progress
import numpy as np
from src.makeplot import plot_output_predictions

# 定义简单的 DNN 模型类
class DNN(nn.Module):
    def __init__(self, dim_list=None):
        super().__init__()
        assert dim_list is not None and len(dim_list) >= 2, "dim_list must be a list with at least input and output dimensions."

        layers = []
        for in_dim, out_dim in zip(dim_list[:-1], dim_list[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.pop()

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class DNN_c_r(nn.Module):
    def __init__(self, dim_list=None):
        super().__init__()
        assert dim_list is not None and len(dim_list) >= 2
        layers = []
        for in_dim, out_dim in zip(dim_list[:-1], dim_list[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.pop()  # remove last ReLU to keep final layer linear (logits)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # <-- return logits


# Utility functions for plotting multi-output predictions and PCA
# def plot_multi_output(y_true, y_pred, name='Train', savepath=None):
#     n_outputs = y_true.shape[1]
#     for i in range(n_outputs):
#         fig, ax = plt.subplots(figsize=(8, 5))
#         ax.plot(y_true[:, i], 'k*', label=f'{name} True y[{i}]')
#         ax.plot(y_pred[:, i], 'o', label=f'{name} Pred y[{i}]')
#         ax.legend()
#         ax.set_xlabel("Sample Index")
#         ax.set_ylabel(f"y[{i}]")
#         plt.tight_layout()
#         if savepath:
#             if not os.path.exists(savepath):
#                 os.makedirs(savepath)
#             fig.savefig(os.path.join(savepath, f"{name.lower()}_prediction_y{i}.png"))
#         plt.close(fig)

# def plot_pca_projection(y_true, y_pred, name='Train', savepath=None):
#     pca = PCA(n_components=2)
#     combined = np.concatenate([y_true, y_pred], axis=0)
#     reduced = pca.fit_transform(combined)
#     n = len(y_true)
#     fig, ax = plt.subplots()
#     ax.scatter(reduced[:n, 0], reduced[:n, 1], c='black', label='True', alpha=0.6)
#     ax.scatter(reduced[n:, 0], reduced[n:, 1], c='red', label='Predicted', alpha=0.6)
#     ax.set_title(f"{name} PCA of Output Vectors")
#     ax.set_xlabel("PC 1")
#     ax.set_ylabel("PC 2")
#     ax.legend()
#     if savepath:
#         if not os.path.exists(savepath):
#             os.makedirs(savepath)
#         fig.savefig(os.path.join(savepath, f"{name.lower()}_pca_output.png"))
#     plt.close(fig)



# 定义训练函数
def train_dnn_model(
    data, num_epochs=1000, lr=1e-3, plot=True, plot_savepath=None, depth=3, hidden_dim=20, J=None
):
    test_mse = None 
    x_train, x_test, y_train, y_test = data
    # 转为 PyTorch tensor
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    # 初始化模型、损失函数和优化器
    input_dim = x_tensor.shape[1] if x_tensor.ndim > 1 else 1
    output_dim = y_tensor.shape[1] if y_tensor.ndim > 1 else 1
    if J >0:
        dim_list = [input_dim] + [J, output_dim] * (depth - 1) + [output_dim]
    else:
        dim_list = [input_dim] + [hidden_dim] * (depth - 1) + [output_dim]

    # dim_list = [input_dim] + [hidden_dim, output_dim] * (depth - 1) + [output_dim]
    model = DNN(dim_list=dim_list)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    l = 100
    losslist1 = []
    interval = max(num_epochs//l, 1)

    # 训练过程
    with Progress() as progress:
        task = progress.add_task("[cyan]Training...", total=num_epochs)

        for epoch in range(num_epochs):
            model.train()
            y_pred = model(x_tensor)
            loss = loss_fn(y_pred, y_tensor)
            if epoch % interval == 0:
                losslist1.append(loss.cpu().detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.update(task, advance=1, description=f"[cyan]Training... Loss: {loss.item():.6f}")

    # 可视化预测结果
    # if plot and x_tensor.ndim == 2 and x_tensor.shape[1] == 1 and y_tensor.ndim == 2 and y_tensor.shape[1] == 1:
        # === 计算训练集 MSE ===
    model.eval()
    with torch.no_grad():
        y_train_pred = model(x_tensor)
        train_mse = loss_fn(y_train_pred, y_tensor).item()
    print(f"✅ Train MSE: {train_mse:.6f}")

    # if x_train.shape[1] == 1 and y_train.shape[1] == 1:
    #     plot = True

    model.eval()
    with torch.no_grad():
        y_fit = model(x_tensor).cpu().numpy()
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

    # if plot and x_train.shape[1] == 1 and y_train.shape[1] == 1:
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.plot(x_train, y_train, 'k*', label='Train True')
    #     ax.plot(x_train, y_fit, 'o', label='DNN Prediction')
    #     ax.legend()
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     # ax.grid(True)
    #     plt.tight_layout()
    #     # plt.show()
    #     if plot_savepath:
    #         if not os.path.exists(plot_savepath):
    #             os.makedirs(plot_savepath)
    #         fig.savefig(plot_savepath+'/train_prediction.png')
    # elif plot and y_train.shape[1] > 1:
    #     plot_multi_output(y_train, y_fit, name='Train', savepath=plot_savepath)
    #     plot_pca_projection(y_train, y_fit, name='Train', savepath=plot_savepath)

    if plot:
        plot_output_predictions(x_train, y_train, y_fit, name='Train', savepath=plot_savepath)

    # Plot test predictions if test data is provided
    if x_test is not None and y_test is not None:
        model.eval()
        with torch.no_grad():
            y_test_pred = model(torch.tensor(x_test, dtype=torch.float32)).cpu().numpy()
        # if plot and x_test.shape[1] == 1 and y_test.shape[1] == 1:
        #     fig, ax = plt.subplots(figsize=(8, 6))
        #     ax.plot(x_test, y_test, 'k*', label='Test True')
        #     ax.plot(x_test, y_test_pred, 'o', label='DNN Prediction')
        #     ax.legend()
        #     ax.set_xlabel("x")
        #     ax.set_ylabel("y")
        #     # ax.grid(True)
        #     plt.tight_layout()
        #     # plt.show()
        #     if plot_savepath:
        #         fig.savefig(plot_savepath+'/test_prediction.png')
        # elif plot and y_test.shape[1] > 1:
        #     plot_multi_output(y_test, y_test_pred, name='Test', savepath=plot_savepath)
        #     plot_pca_projection(y_test, y_test_pred, name='Test', savepath=plot_savepath)
        if plot:
            plot_output_predictions(x_test, y_test, y_test_pred, name='Test', savepath=plot_savepath)
        test_mse = loss_fn(torch.tensor(y_test_pred), torch.tensor(y_test, dtype=torch.float32)).item()
        print(f"✅ Test MSE: {test_mse:.6f}")

    return model, train_mse, test_mse



def train_dnn_model_multiclass(
    data, num_epochs=1000, lr=1e-3, plot=True, plot_savepath=None, depth=3, hidden_dim=20, J=None
):
    x_train, x_test, y_train, y_test = data

    # —— 1) 统一标签到索引形式，拿到类别数 ——
    # 若是 one-hot（N, K），先转索引
    if isinstance(y_train, np.ndarray) and y_train.ndim == 2 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
        y_test  = np.argmax(y_test,  axis=1)
    # 若是 (N,1) -> (N,)
    if isinstance(y_train, np.ndarray) and y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.squeeze(1)
    if isinstance(y_test, np.ndarray)  and y_test.ndim  == 2 and y_test.shape[1]  == 1:
        y_test = y_test.squeeze(1)

    # 计算类别数（假设标签是 0..K-1）
    num_classes = int(np.max(y_train)) + 1
    print("y classes:", num_classes)

    # —— 2) 张量与设备 ——
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    x_test  = torch.tensor(x_test,  dtype=torch.float32, device=device) if x_test is not None else None
    y_train = torch.tensor(y_train, dtype=torch.long,    device=device)
    y_test  = torch.tensor(y_test,  dtype=torch.long,    device=device) if y_test is not None else None

    # —— 3) 模型结构：隐藏层维度固定为 hidden_size，输出为 num_classes ——
    input_dim = x_train.shape[1] if x_train.ndim > 1 else 1
    hidden_size = J if J > 0 else hidden_dim
    if depth <= 1:
        dim_list = [input_dim, num_classes]
    else:
        dim_list = [input_dim] + [hidden_size] * (depth - 1) + [num_classes]

    # 重要：使用不带 softmax 的模型（输出 logits）
    model = DNN(dim_list=dim_list).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # —— 4) 训练循环 ——
    loss_history = []
    model.train()
    for epoch in range(num_epochs):
        logits = model(x_train)                 # [N, C]
        loss = loss_fn(logits, y_train)         # y_train: [N]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, num_epochs // 100) == 0:
            loss_history.append(loss.item())

    # —— 5) 训练集指标 ——
    model.eval()
    with torch.no_grad():
        logits_train = model(x_train)
        train_loss = loss_fn(logits_train, y_train).item()
        train_pred = torch.argmax(logits_train, dim=-1)
        train_acc  = (train_pred == y_train).float().mean().item()
    print(f"✅ Train Loss: {train_loss:.6f} | Train Acc: {train_acc*100:.2f}%")

    # —— 6) 测试集指标（如果提供） ——
    test_loss = None
    test_acc  = None
    if x_test is not None and y_test is not None:
        with torch.no_grad():
            logits_test = model(x_test)
            test_loss = loss_fn(logits_test, y_test).item()
            test_pred = torch.argmax(logits_test, dim=-1)
            test_acc  = (test_pred == y_test).float().mean().item()
        print(f"✅ Test  Loss: {test_loss:.6f} | Test  Acc: {test_acc*100:.2f}%")

    # —— 7) 可选：画损失曲线 ——
    if plot:
        os.makedirs(plot_savepath, exist_ok=True)
        fig, ax = plt.subplots()
        ax.plot(np.linspace(1, num_epochs, len(loss_history)), loss_history, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss (sampled)')
        ax.set_title('Multiclass Training Loss')
        ax.grid(True)
        fig.savefig(os.path.join(plot_savepath, 'loss_multiclass.png'))
        plt.close(fig)

    return model, train_loss, test_loss, train_acc, test_acc