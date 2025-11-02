import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pyro
import torch
import time

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pyro.distributions as dist

from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import trange
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from pyro.distributions import constraints
from pyro.infer.autoguide import AutoDiagonalNormal
# from pyro.infer.autoguide import AutoBNAFNormal
from pyro.optim import ClippedAdam

from src.dgp_with_init.deep_layer import DeepGP, DeepGPNoBias, DeepEnsembleGP
import torch
torch.cuda.is_available()
torch.cuda.current_device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
from src.makeplot import plot_output_predictions

class InitParams:
    def __init__(self, mean: float, var: float) -> None:
        self.mean = mean
        self.var = var

    def __repr__(self) -> str:
        return f"InitParams(mean={self.mean}, var={self.var})"
    
def split_array_by_lengths(array, lengths):
    result = []
    current_position = 0

    for length in lengths:
        if current_position + length <= len(array):
            result.append(array[current_position:current_position + length])
            current_position += length
        else:
            result.append(array[current_position:])
            break

    return result

def recursive_to_device(module, device):
    module.to(device)
    for name, child in module.named_children():
        recursive_to_device(child, device)

def get_batch(x, y, batch_size, i):
    """
    获取第 i 个 batch 的数据。
    如果数据不足一个完整 batch，则自动补齐到数据末尾。

    Args:
        x (Tensor): 输入数据 (N, D)
        y (Tensor): 标签数据 (N, 1)
        batch_size (int): 每个 batch 大小
        i (int): 当前是第 i 个 batch（从 0 开始）

    Returns:
        xb (Tensor): 第 i 个 batch 的输入 (batch_size, D)
        yb (Tensor): 第 i 个 batch 的输出 (batch_size, 1)
    """
    start = i * batch_size
    end = min(start + batch_size, x.shape[0])
    xb = x[start:end]
    yb = y[start:end]
    return xb, yb

class Model_inheritage_c_r(PyroModule):
    def __init__(self, dim_list=[1, 1, 1], J_list=[5, 10], device="cuda", 
                 likelihood: str = "classification",  # "regression" 或 "classification"
                 num_classes: int | None = None,
                 projector=None,
                 **kwargs):
        super().__init__()
        self.out_dim = dim_list[-1]
        self.device = device
        self.model = DeepGP(dim_list, J_list, **kwargs)
        self.likelihood = likelihood
        self.num_classes = num_classes  # 分类时必须指定 K
        self.kwargs = kwargs
        self.projector = projector

    def forward(self, x, y=None):
        # f(x)：[N, D]；回归 D = out_dim；多分类时要求 D -> K
        mu = self.model(x)  # [N, D]
        logits_or_mean = mu @ self.projector  # [N, D] @ [D, K] -> [N, K]
        N, D = logits_or_mean.shape
        if self.likelihood == "classification":
            assert self.num_classes is not None, "请传入 num_classes"
            assert D == self.num_classes, f"最后一层维度 {D} 必须等于 num_classes={self.num_classes}"

            # 让 Predictive 能拿到 logits 做后验平均
            pyro.deterministic("logits", logits_or_mean)

            # 观测：Categorical + logits（不要自己先 softmax）
            with pyro.plate("data", N):
                return pyro.sample("obs", dist.Categorical(logits=logits_or_mean), obs=y)
        else:
            print("This is regression likelihood!")




# Define the model
class Model_inheritage(PyroModule):
    def __init__(self, dim_list=[1, 1, 1], J_list=[5, 10], out_dim=1, device="cuda", projector=None, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.device = device
        self.model = DeepGP(dim_list, J_list, **kwargs).to(self.device)
        self.kwargs = kwargs
        self.projector = projector

    def forward(self, x, y=None, sample_obs=True):
        mu = self.model(x)  # [B, D]
        mu = mu @ self.projector  # [B, D] @ [D, out_dim] -> [B, out_dim]

        if 'init_scale' in self.kwargs:
            init_w = self.kwargs['init_scale']
            sigma = pyro.sample("sigma", dist.Gamma(init_w.mean, init_w.var))  # [D]
            scale = sigma.expand([x.shape[0], self.out_dim])  # [B, D]
            print("scale initialization success!")
        else:
            sigma = pyro.param("sigma", torch.tensor(0.5), constraint=pyro.distributions.constraints.positive)  # [D]
            scale = sigma.expand([x.shape[0], self.out_dim])  # [B, D]

        with pyro.plate("data", x.shape[0]):
            if sample_obs:
                if y is not None:
                    pyro.sample("obs", dist.Normal(mu, scale).to_event(1), obs=y)
                else:
                    pyro.sample("obs", dist.Normal(mu, scale).to_event(1))

        return mu

    # def forward(self, x, y=None, sample_obs=True):
    #     mu = self.model(x)#.squeeze()
    #     # scale = pyro.param("sigma", torch.tensor(0.5), constraint=pyro.distributions.constraints.positive).expand(self.out_dim)
    #     scale = pyro.param("sigma", torch.tensor(0.5), constraint=pyro.distributions.constraints.positive)
    #     scale = scale.expand([x.shape[0], self.out_dim])    

    #     if 'init_scale' in self.kwargs:
    #         init_w = self.kwargs['init_scale']
    #         scale = pyro.sample("sigma", dist.Gamma(init_w.mean, init_w.var)).expand(self.out_dim)
    #         print("scale initialization success!")

        
    #     with pyro.plate("data", x.shape[0]):
    #         if sample_obs and y is not None:
    #             if y.ndim == 2 and y.shape[-1] == 1:
    #                 y = y.squeeze(-1)
    #             pyro.sample("obs", dist.Normal(mu, scale).to_event(1), obs=y)
    #         elif sample_obs:
    #             pyro.sample("obs", dist.Normal(mu, scale).to_event(1))  

    #     return mu



def train_inheritage_single(
    data, num_epochs=1000, lr=0.001,
    plot=False, plot_savepath=None, model_token=None,
    batch_size=1024, dim_list=None, J_list=None, projector=None, **kwargs
):
    ###module_depth: 每个拼起来GP有几层 当然一层就够了所以暂时算是无用的参数
    #有需要请改lrd
    x_train, x_test, y_train, y_test = data
    if not isinstance(x_train, torch.Tensor):
        x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    d_out = y_train.shape[1]
    model1 = Model_inheritage(dim_list=dim_list, J_list=J_list, out_dim=d_out, projector=projector, **kwargs)
    model1.to(device)
    print(model1)
    # recursive_to_device(model1, device)
    guide = AutoDiagonalNormal(model1)
    optimizer = ClippedAdam({"lr": lr, "lrd": 1})
    svi = SVI(model1, guide, optimizer, loss=Trace_ELBO())

    pyro.clear_param_store()

    num_batches = (x_train.shape[0] + batch_size - 1) // batch_size
    l = 100
    losslist1 = []
    interval = max(num_epochs//l, 1)
    progress_bar = trange(num_epochs)
    for epoch in progress_bar:
        epoch_loss = 0.0
        for i in range(num_batches):
            xb, yb = get_batch(x_train, y_train, batch_size, i)
            loss = svi.step(xb.to(device), yb.to(device))
            epoch_loss += loss

        avg_loss = epoch_loss / x_train.shape[0]
        if epoch % interval == 0:
            losslist1.append(avg_loss)
        progress_bar.set_postfix(loss=f"{avg_loss:.3f}")

    # # === batched Evaluation ===
    # predictive1 = Predictive(model1, guide=guide, num_samples=50)
    # y_pred1_train, _ = batched_predict(predictive1, x_train, batch_size)
    # y_pred1_test, _ = batched_predict(predictive1, x_test, batch_size)

    predictive1 = Predictive(model1, guide=guide, num_samples=50)
    y_pred1_train = predictive1(x_train)['obs'].cpu().detach().numpy().mean(axis=0)
    y_pred1_test = predictive1(x_test)['obs'].cpu().detach().numpy().mean(axis=0)

    # === Extract Parameters ===
    param_store = pyro.get_param_store()
    module_depth = len(J_list)
    in_dim_list = dim_list[:-1]
    out_dim_list = dim_list[1:]

    len_list = []
    for i in range(module_depth):
        len_list.append(J_list[i]*in_dim_list[i])
        len_list.append(out_dim_list[i]*2*J_list[i])
        len_list.append(out_dim_list[i])
    len_list.append(1)

    loc = param_store["AutoDiagonalNormal.loc"]
    scale = param_store["AutoDiagonalNormal.scale"]

    result_l = split_array_by_lengths(loc, len_list)
    result_s = split_array_by_lengths(scale, len_list)

    store_copy = pyro.get_param_store().get_state()  # 保存参数副本

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
        
        fig.tight_layout()
        os.makedirs(plot_savepath, exist_ok=True)
        fig.savefig(os.path.join(plot_savepath, f'loss_{model_token if model_token else "model"}.png'))

    return model1, guide, store_copy, y_pred1_train, y_pred1_test, result_l, result_s



def train_inheritage_multiple(data,  depth=2, module_depth = 1, num_epochs=1000, growth_factor = 1.0, lr = 0.001, batch_size = 1024, plot = False, plot_savepath=None, j = 20, hid_dim=10, config=None, **kwargs):
    modellist = []
    guidelist = []
    paramslist = []
    paramstorelist = []
    train_mselist = []
    test_mselist = []
    # params = {}

    x_train, x_test, y_train, y_test = data
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device) # N * Dx
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device) # N * Dy
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    data = (x_train, x_test, y_train, y_test)
    # 投影到 hid_dim 维
    xdim = x_train.shape[1]
    # if xdim < hid_dim:
    #     ncopy = hid_dim // xdim + 1
    # else:
    #     ncopy = 1
    # x_train_temp = x_train.repeat(1, ncopy)
    # x_test_temp = x_test.repeat(1, ncopy)
    # xdim_temp = x_train_temp.shape[1]
    # projectorx = random_projection_matrix(xdim_temp, hid_dim)  
    # x_train_projected = x_train_temp @ projectorx # N * hid_dim
    # x_test_projected = x_test_temp @ projectorx
    # data = (x_train_projected, x_test_projected, y_train, y_test)
    out_dim = y_train.shape[1]
    projector = random_projection_matrix(hid_dim, out_dim).to(device)

    # projector = random_projection_matrix(hid_dim, y_train.shape[1]).to(device) # we want num_classes > hid_dim

    time0 = time.time()
    timelist = []
    for n in range(depth):
        module_depth = n+1
        model_token = f"layer{n+1}"
        print(f"Training layer {n+1} with token {model_token}")
        x_train, x_test, y_train, y_test = data

        # dim_list=[x_train.shape[1]]+[num_classes]*(n+1)  # 注意这里的维度设置
        dim_list = [xdim] + [hid_dim]*(n+1) # 注意这里的维度设置
        J_list = [j]*(n+1)
        grow_num_epochs = int(num_epochs * (growth_factor ** n))
        model1, mean_field_guide, store_copy, y_pred1_train, y_pred1_test, result_l, result_s = train_inheritage_single(
            data=data, num_epochs=grow_num_epochs, lr=lr, batch_size=batch_size, dim_list=dim_list, J_list=J_list, plot=plot, plot_savepath=plot_savepath, model_token=model_token, projector=projector, **kwargs
        )

        loc = result_l
        scale = result_s
        in_dim_list = dim_list[:-1]
        out_dim_list = dim_list[1:]
        # print(f"Input dimensions: {in_dim_list}, Output dimensions: {out_dim_list}")
        # print("=== loc shapes ===")
        # for i, t in enumerate(loc):
        #     print(f"loc[{i}]: shape = {t.shape}")

        # print("=== scale shapes ===")
        # for i, t in enumerate(scale):
        #     print(f"scale[{i}]: shape = {t.shape}")


        len_list = []
        for i in range(module_depth):
            len_list.append(J_list[i]*in_dim_list[i])
            len_list.append(out_dim_list[i]*2*J_list[i])
            len_list.append(out_dim_list[i])
        len_list.append(1)
        len_list

        params = {}
        for i in range(module_depth):   
            params[f'init_1st_w{i}'] = InitParams(
                loc[3*i+0].reshape(J_list[i], in_dim_list[i]).detach(),
                scale[3*i+0].reshape(J_list[i], in_dim_list[i]).detach()
            )
            params[f'init_2nd_w{i}'] = InitParams(
                loc[3*i+1].reshape(out_dim_list[i], 2*J_list[i]).detach(),
                scale[3*i+1].reshape(out_dim_list[i], 2*J_list[i]).detach()
            )
            params[f'init_2nd_b{i}'] = InitParams(
                loc[3*i+2].detach(),
                scale[3*i+2].detach()
            )
        # if config['init_method'] == 'ww':
        if module_depth > 1:
            i = module_depth - 1
            params[f'init_1st_w{i+1}'] = InitParams(
                    loc[3*i+0].reshape(J_list[i], in_dim_list[i]).detach(),
                    scale[3*i+0].reshape(J_list[i], in_dim_list[i]).detach()
                )
            params[f'init_2nd_w{i+1}'] = InitParams(
                    loc[3*i+1].reshape(out_dim_list[i], 2*J_list[i]).detach(),
                    scale[3*i+1].reshape(out_dim_list[i], 2*J_list[i]).detach()
                )
            params[f'init_2nd_b{i+1}'] = InitParams(
                    loc[3*i+2].detach(),
                    scale[3*i+2].detach()
                )
        # if config['init_method'] == 'zz':
        #     pass
            # i = module_depth
            # params[f'init_1st_w{i+1}'] = InitParams(
            #         loc[3*i+0].reshape(J_list[i], in_dim_list[i]).detach(),
            #         scale[3*i+0].reshape(J_list[i], in_dim_list[i]).detach()
            #     )
            # params[f'init_2nd_w{i+1}'] = InitParams(
            #         loc[3*i+1].reshape(out_dim_list[i], 2*J_list[i]).detach(),
            #         scale[3*i+1].reshape(out_dim_list[i], 2*J_list[i]).detach()
            #     )
            # params[f'init_2nd_b{i+1}'] = InitParams(
            #         torch.zeros_like(loc[3*i+2]).detach(),
            #         torch.ones_like(scale[3*i+2]).detach()
            #     )
        # 可选 scale 初始化
        # params['init_scale'] = InitParams(loc[-1][0].view(1,1), scale[-1][0].view(1,1))
        #加入继承
        #if n > 1:
        kwargs.update(params)   

        
        # data = (y_pred1_train, y_pred1_test, y_train, y_test)
        modellist.append(model1)
        guidelist.append(mean_field_guide)
        paramslist.append(params)
        paramstorelist.append(store_copy)


        # === 计算 MSE ===
        train_mse = ((y_pred1_train - y_train.cpu().numpy()) ** 2).mean()
        test_mse = ((y_pred1_test - y_test.cpu().numpy()) ** 2).mean()

        y_train_cpu = y_train.cpu().detach().numpy()
        print(f"[METRIC] Combined Train MSE after layer {n+1}: {train_mse:.6f}")
        y_test_cpu = y_test.cpu().detach().numpy()
        print(f"[METRIC] Combined Test MSE after layer {n+1}: {test_mse:.6f}")
        
        if plot:
            plot_output_predictions(x_train.cpu().numpy(), y_train.cpu().numpy(), y_pred1_train, name='Train'+model_token, savepath=plot_savepath)
            plot_output_predictions(x_test.cpu().numpy(), y_test.cpu().numpy(), y_pred1_test, name='Test'+model_token, savepath=plot_savepath)
            # os.makedirs(plot_savepath, exist_ok=True)

            # # === 转为 numpy 并解包 shape ===
            # x_obs = x_test[:, 0].cpu().numpy().squeeze()
            # y_obs = y_test.cpu().numpy()
            # y_pred = y_pred1_test
            # y_dim = y_obs.shape[1]

            # # === Plot Observation vs Prediction ===
            # fig, axes = plt.subplots(1, y_dim, figsize=(4 * y_dim, 3))
            # if y_dim == 1:
            #     axes = [axes]

            # for d in range(y_dim):
            #     axes[d].scatter(x_obs, y_obs[:, d], label="Observation", color='black', s=10, alpha=0.6)
            #     axes[d].scatter(x_obs, y_pred[:, d], label="Prediction", color='blue', s=10, alpha=0.6)
            #     axes[d].set_title(f"Output dim {d+1}")
            #     axes[d].set_xlabel("Input dim 0")
            #     axes[d].set_ylabel("Value")
            #     axes[d].legend()

            # fig.tight_layout()
            # fig.savefig(os.path.join(plot_savepath, f'obs_pred_combined_layer{n+1}.png'))
            # plt.close()

            # # === Plot Residuals ===
            # fig, axes = plt.subplots(1, y_dim, figsize=(4 * y_dim, 3))
            # if y_dim == 1:
            #     axes = [axes]

            # for d in range(y_dim):
            #     y_res = y_pred[:, d] - y_obs[:, d]
            #     axes[d].scatter(y_obs[:, d], y_res, s=8, alpha=0.6, label="Residual")
            #     axes[d].axhline(0, color='red', linestyle='--', linewidth=1)
            #     axes[d].set_xlabel('True Value')
            #     axes[d].set_ylabel('Residual')
            #     axes[d].set_title(f"Residuals Output {d+1}")
            #     axes[d].legend()

            # fig.tight_layout()
            # fig.savefig(os.path.join(plot_savepath, f'residuals_combined_layer{n+1}.png'))
            # plt.close()




        train_mselist.append(train_mse)
        test_mselist.append(test_mse)
        time1 = time.time()
        timelist.append(time1-time0)
        print(f"Time for layer {n+1}: {time1-time0:.2f} seconds")
        time0 = time1

    return modellist, train_mselist, test_mselist, timelist







def train_inheritage_single_multiclass(
    data, num_epochs=1000, lr=0.001,
    plot=False, plot_savepath=None, model_token=None,
    batch_size=1024, dim_list=None, J_list=None, num_classes=None, projector=None, **kwargs
):
    # print("dim_list:", dim_list, "J_list:", J_list )
    ###module_depth: 每个拼起来GP有几层 当然一层就够了所以暂时算是无用的参数
    #有需要请改lrd
    x_train, x_test, y_train, y_test = data
    if not isinstance(x_train, torch.Tensor):
        x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)  # 注意这里是 long 类型
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)  # 注意这里是 long 类型

    y_dim = y_train.shape[1] if y_train.ndim > 1 else 1
    if y_train.shape[1] == 1:
        y_train = y_train.squeeze(1)
        y_test  = y_test.squeeze(1)

    model1 = Model_inheritage_c_r(dim_list=dim_list, J_list=J_list, likelihood="classification", num_classes=num_classes, projector=projector, **kwargs)
    model1.to(device)
    print(model1)
    # recursive_to_device(model1, device)
    guide = AutoDiagonalNormal(model1)
    optimizer = ClippedAdam({"lr": lr, "lrd": 1})
    svi = SVI(model1, guide, optimizer, loss=Trace_ELBO())

    pyro.clear_param_store()

    num_batches = (x_train.shape[0] + batch_size - 1) // batch_size
    l = 100
    losslist1 = []
    interval = max(num_epochs//l, 1)
    progress_bar = trange(num_epochs)
    for epoch in progress_bar:
        epoch_loss = 0.0
        for i in range(num_batches):
            xb, yb = get_batch(x_train, y_train, batch_size, i)
            loss = svi.step(xb.to(device), yb.to(device))
            epoch_loss += loss

        avg_loss = epoch_loss / x_train.shape[0]
        if epoch % interval == 0:
            losslist1.append(avg_loss)
        progress_bar.set_postfix(loss=f"{avg_loss:.3f}")   

    # # === batched Evaluation ===
    predictive1 = Predictive(model1, guide=guide, num_samples=50, return_sites=["logits"])
    # print(predictive1(x_train)['logits'])
    y_pred1_train = predictive1(x_train)['logits'].mean(0)[0]
    y_prob1_train = torch.softmax(y_pred1_train, dim=-1)
    print("y_prob1_train shape",y_prob1_train.shape)
    y_pred1_test = predictive1(x_test)['logits'].mean(0)[0]
    y_prob1_test = torch.softmax(y_pred1_test, dim=-1)
    y_class1_train = y_prob1_train.argmax(dim=-1)  # 取概率最大的类别
    y_class1_test = y_prob1_test.argmax(dim=-1)    # 取概率最大的类别   
    train_acc = (y_class1_train == y_train).float().mean().item()
    test_acc = (y_class1_test == y_test).float().mean().item()
    print(f"[METRIC] Train Accuracy: {train_acc:.6f}")
    print(f"[METRIC] Test Accuracy: {test_acc:.6f}")

    # === Extract Parameters ===
    param_store = pyro.get_param_store()
    module_depth = len(J_list)
    in_dim_list = dim_list[:-1]
    out_dim_list = dim_list[1:]     
    len_list = []
    for i in range(module_depth):
        len_list.append(J_list[i]*in_dim_list[i])
        len_list.append(out_dim_list[i]*2*J_list[i])
        len_list.append(out_dim_list[i])
    len_list.append(1)  
    loc = param_store["AutoDiagonalNormal.loc"]
    scale = param_store["AutoDiagonalNormal.scale"] 
    result_l = split_array_by_lengths(loc, len_list)
    result_s = split_array_by_lengths(scale, len_list)
    store_copy = pyro.get_param_store().get_state()  # 保存参数副本 
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
        
        fig.tight_layout()
        os.makedirs(plot_savepath, exist_ok=True)
        fig.savefig(os.path.join(plot_savepath, f'loss_{model_token if model_token else "model"}.png'))
    return model1, guide, store_copy, y_pred1_train, y_pred1_test, result_l, result_s, train_acc, test_acc


def train_inheritage_multiple_multiclass(data,  depth=2, module_depth = 1, num_epochs=1000, growth_factor = 1.0, lr = 0.001, batch_size = 1024, plot = False, plot_savepath=None, j = 20, num_classes=None, hid_dim=10, config=None, **kwargs):    
    modellist = []
    guidelist = []
    paramslist = []
    paramstorelist = []
    train_acclist = []
    test_acclist = []
    # params = {}

    x_train, x_test, y_train, y_test = data
    num_classes = len(np.unique(y_train))
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)  # 注意这里是 long 类型
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)  # 注意这里是 long 类型
    
    # 投影到 hid_dim 维
    xdim = x_train.shape[1]
    if xdim < hid_dim:
        ncopy = hid_dim // xdim + 1
    else:
        ncopy = 1
    x_train_temp = x_train.repeat(1, ncopy)
    x_test_temp = x_test.repeat(1, ncopy)
    xdim_temp = x_train_temp.shape[1]
    projectorx = random_projection_matrix(xdim_temp, hid_dim)
    x_train_projected = x_train_temp @ projectorx
    x_test_projected = x_test_temp @ projectorx
    data = (x_train_projected, x_test_projected, y_train, y_test)

    time0 = time.time()
    timelist = []
    for n in range(depth):
        module_depth = n+1
        model_token = f"layer{n+1}"
        print(f"Training layer {n+1} with token {model_token}")
        x_train, x_test, y_train, y_test = data
        projector = random_projection_matrix(hid_dim, num_classes) # we want num_classes > hid_dim
        # dim_list=[x_train.shape[1]]+[num_classes]*(n+1)  # 注意这里的维度设置
        dim_list = [hid_dim]*(n+2)  # 注意这里的维度设置
        J_list = [j]*(n+1)
        grow_num_epochs = int(num_epochs * (growth_factor ** n))

        model1, mean_field_guide, store_copy, y_pred1_train, y_pred1_test, result_l, result_s, train_acc, test_acc = train_inheritage_single_multiclass(
            data=data, num_epochs=grow_num_epochs, lr=lr, batch_size=batch_size, dim_list=dim_list, J_list=J_list, num_classes=num_classes, plot=plot, plot_savepath=plot_savepath, model_token=model_token, projector=projector,  **kwargs
        )

        loc = result_l
        scale = result_s
        in_dim_list = dim_list[:-1]
        out_dim_list = dim_list[1:]
        len_list = []
        for i in range(module_depth):
            len_list.append(J_list[i]*in_dim_list[i])
            len_list.append(out_dim_list[i]*2*J_list[i])
            len_list.append(out_dim_list[i])
        len_list.append(1)
        len_list   

        
        params = {}
        for i in range(module_depth):   
            params[f'init_1st_w{i}'] = InitParams(
                loc[3*i+0].reshape(J_list[i], in_dim_list[i]).detach(),
                scale[3*i+0].reshape(J_list[i], in_dim_list[i]).detach()
            )
            params[f'init_2nd_w{i}'] = InitParams(
                loc[3*i+1].reshape(out_dim_list[i], 2*J_list[i]).detach(),
                scale[3*i+1].reshape(out_dim_list[i], 2*J_list[i]).detach()
            )
            params[f'init_2nd_b{i}'] = InitParams(
                loc[3*i+2].detach(),
                scale[3*i+2].detach()
            )
        # if config['init_method'] == 'ww':
        i = module_depth - 1
        params[f'init_1st_w{i+1}'] = InitParams(
                loc[3*i+0].reshape(J_list[i], in_dim_list[i]).detach(),
                scale[3*i+0].reshape(J_list[i], in_dim_list[i]).detach()
            )
        params[f'init_2nd_w{i+1}'] = InitParams(
                loc[3*i+1].reshape(out_dim_list[i], 2*J_list[i]).detach(),
                scale[3*i+1].reshape(out_dim_list[i], 2*J_list[i]).detach()
            )
        params[f'init_2nd_b{i+1}'] = InitParams(
                loc[3*i+2].detach(),
                scale[3*i+2].detach()
            )
        # if config['init_method'] == 'zz':
        #     pass
            # i = module_depth - 1
            # params[f'init_1st_w{i+1}'] = InitParams(
            #         loc[3*i+0].reshape(J_list[i], in_dim_list[i]).detach(),
            #         scale[3*i+0].reshape(J_list[i], in_dim_list[i]).detach()
            #     )
            # params[f'init_2nd_w{i+1}'] = InitParams(
            #         loc[3*i+1].reshape(out_dim_list[i], 2*J_list[i]).detach(),
            #         scale[3*i+1].reshape(out_dim_list[i], 2*J_list[i]).detach()
            #     )
            # params[f'init_2nd_b{i+1}'] = InitParams(
            #         torch.zeros_like(loc[3*i+2]).detach(),
            #         torch.ones_like(scale[3*i+2]).detach()
            #     )
        # 可选 scale 初始化
        # params['init_scale'] = InitParams(loc[-1][0].view(1,1), scale[-1][0].view(1,1))
        #加入继承
        #if n > 1:
        kwargs.update(params)   

        modellist.append(model1)
        guidelist.append(mean_field_guide)
        paramslist.append(params)
        paramstorelist.append(store_copy)

        train_acclist.append(train_acc)
        test_acclist.append(test_acc)

        time1 = time.time()
        timelist.append(time1-time0)
        print(f"Time for layer {n+1}: {time1-time0:.2f} seconds")
        time0 = time1
        
    return modellist, train_acclist, test_acclist, timelist

# def stacked_predict_classification(x_input, modellist, guidelist, paramstore_list):
#     x0 = x_input
#     x = x0
#     for m, g, store in zip(modellist, guidelist, paramstore_list):
#         pyro.clear_param_store()
#         pyro.get_param_store().set_state(store)  # 加载该层参数
#         predictive = Predictive(m, guide=g, num_samples=50)
#         pred_samples = predictive(x)['logits']
#         x = pred_samples.mean(0)[0]
#         if m != modellist[-1]:
#             x = torch.cat([x0, x], dim=1)  # 拼接输入和预测结果，最后一层不拼接
#     return x.detach().cpu().numpy()


def random_projection_matrix(d_in, d_out):#d_in > d_out
    A = torch.randn(d_in, d_in).to(device)
    Q, R = torch.linalg.qr(A)
    d = torch.diag(R)
    Q *= d.sign()
    return Q[:, :d_out]  # (d_in, d_out)