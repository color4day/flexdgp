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

from src.dgp_with_init.deep_layer import DeepGP, DeepGPNoBias, DeepEnsembleGP,ResDeepGP, dsmDeepGP
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

class Model_dsmdgp_c_r(PyroModule):
    def __init__(self, dim_list=[1,1,1], J_list=[5,10],
                out_dim = 1, device="cuda", projection = None, 
                likelihood: str = "classification",  # "regression" 或 "classification"
                num_classes: int | None = None, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.model = dsmDeepGP(dim_list, J_list, **kwargs).to(device)
        self.likelihood = likelihood
        self.num_classes = num_classes  # 分类时必须指定 K
        self.projection = projection
        # self.model.to('cuda')

    def forward(self, x, y=None):
        # f(x)：[N, D]；回归 D = out_dim；多分类时要求 D = K
        mu = self.model(x)
        logits_or_mean = mu @ self.projection  # [N, D] or [N, K]
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
class Model_dsmdgp(PyroModule):
    def __init__(self, dim_list=[1, 1, 1], J_list=[5, 10],out_dim = 1, device="cuda", projection=None, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.device = device
        self.model = dsmDeepGP(dim_list, J_list, **kwargs).to(self.device)
        self.kwargs = kwargs
        self.projection = projection

    def forward(self, x, y=None, sample_obs=True):
        mu = self.model(x)  # [B, D]
        mu = mu @ self.projection

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

def random_projection_matrix(d_in, d_out):#d_in > d_out
    A = torch.randn(d_in, d_in).to(device)
    Q, R = torch.linalg.qr(A)
    d = torch.diag(R)
    Q *= d.sign()
    return Q[:, :d_out]  # (d_in, d_out)


def train_dsmdgp_single(
    data, num_epochs=1000, lr=0.001, projection=None, 
    plot=False, plot_savepath=None, model_token=None,
    batch_size=1024, dim_list=None, J_list=None,
    **kwargs
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
    model1 = Model_dsmdgp(dim_list=dim_list, J_list=J_list, **kwargs,
                          out_dim=d_out, device=device, projection=projection, )
    model1.to(device)
    # print(model1)
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
    for name, value in param_store.items():  # 打印参数名称和形状 ???
        print(f"{name}: {value.shape}")
    module_depth = len(J_list)
    in_dim_list = dim_list[:-1]
    out_dim_list = dim_list[1:]
    print("in:", in_dim_list, "out:", out_dim_list)

    len_list = []
    for i in range(module_depth):
        len_list.append(J_list[i]*in_dim_list[i])
        len_list.append(out_dim_list[i]*2*J_list[i])
        len_list.append(out_dim_list[i])
    for i in range(module_depth):
        len_list.append(J_list[i]*in_dim_list[0])
        len_list.append(out_dim_list[i]*2*J_list[i])
        len_list.append(out_dim_list[i])
    len_list.append(1)
    print("parameter lenth list: ", len_list)

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



def train_dsmdgp_multiple(data,  depth=2, module_depth = 1, num_epochs=1000, growth_factor = 1,lr = 0.001, batch_size = 1024, plot = False, plot_savepath=None, j = 20, hid_dim=10,initflag = False, **kwargs):
    modellist = []
    guidelist = []
    paramslist = []
    paramstorelist = []
    train_mselist = []
    test_mselist = []
    # params = {}

    x_train, x_test, y_train, y_test = data
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    data = (x_train, x_test, y_train, y_test)
    
    # 投影到 hid_dim 维
    xdim = x_train.shape[1]
    # if xdim < hid_dim:
    #     ncopy = hid_dim // xdim + 1
    # else:
    #     ncopy = 1
    # projectionx = torch.cat([torch.eye(xdim)] * ncopy, dim=1).to(device) @ random_projection_matrix(xdim*ncopy, hid_dim).to(device)
    out_dim = y_train.shape[1]
    projection = random_projection_matrix(hid_dim, out_dim).to(device)
    # x_train, x_test, y_train, y_test = data

    time0 = time.time()
    timelist = []
    for n in range(depth):
        module_depth = n+1
        model_token = f"layer{n+1}"
        print(f"Training layer {n+1} with token {model_token}")
        dim_list = [xdim] + [hid_dim]*(n+1)
        jx = j ### temperary set for first layer
        J_list = [jx] + [j]*(n)
        grow_num_epochs = int(num_epochs * (growth_factor ** n))
        model1, mean_field_guide, store_copy, y_pred1_train, y_pred1_test, result_l, result_s = train_dsmdgp_single(
            data=data, num_epochs=grow_num_epochs, lr=lr, batch_size=batch_size, dim_list=dim_list, J_list=J_list, plot=plot, plot_savepath=plot_savepath, model_token=model_token, **kwargs,
            projection=projection, 
        )


        # === Extract Parameters ===
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
        # projectionx = loc[0].reshape(xdim, hid_dim).detach()
        # initialize W value
        current_index = 0
        for i in range(module_depth):
            params[f'init_1st_w{i}'] = InitParams(
                loc[3*i+0+current_index].reshape(J_list[i], in_dim_list[i]).detach(),
                scale[3*i+0+current_index].reshape(J_list[i], in_dim_list[i]).detach()
            )
            params[f'init_2nd_w{i}'] = InitParams(
                loc[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach(),
                scale[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach()
            )
            params[f'init_2nd_b{i}'] = InitParams(
                loc[3*i+2+current_index].detach(),
                scale[3*i+2+current_index].detach()
            )
        # if config['init_method'] == 'ww':
        
        if module_depth > 1 and initflag ==True:
            print("init is on")
            i = module_depth - 1
            params[f'init_1st_w{i+1}'] = InitParams(
                    loc[3*i+0+current_index].reshape(J_list[i], in_dim_list[i]).detach(),
                    scale[3*i+0+current_index].reshape(J_list[i], in_dim_list[i]).detach()
                )
            params[f'init_2nd_w{i+1}'] = InitParams(
                    loc[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach(),
                    scale[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach()
                )
            params[f'init_2nd_b{i+1}'] = InitParams(
                    loc[3*i+2+current_index].detach(),
                    scale[3*i+2+current_index].detach()
                )
        
            current_index = 3*(i+1)
        else:
            print("init is OFF")
            i = module_depth
            current_index = 3*(i)
        for i in range(module_depth):
            params[f'init_1st_ws{i}'] = InitParams(
                loc[3*i+0+current_index].reshape(J_list[i], in_dim_list[0]).detach(),
                scale[3*i+0+current_index].reshape(J_list[i], in_dim_list[0]).detach()
            )
            params[f'init_2nd_ws{i}'] = InitParams(
                loc[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach(),
                scale[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach()
            )
            params[f'init_2nd_bs{i}'] = InitParams(
                loc[3*i+2+current_index].detach(),
                scale[3*i+2+current_index].detach()
            )
        if initflag ==True:
            i = module_depth - 1
            params[f'init_1st_ws{i+1}'] = InitParams(
                    loc[3*i+0+current_index].reshape(J_list[i], in_dim_list[0]).detach(),
                    scale[3*i+0+current_index].reshape(J_list[i], in_dim_list[0]).detach()
                )
            params[f'init_2nd_ws{i+1}'] = InitParams(
                    loc[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach(),
                    scale[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach()
                )
            params[f'init_2nd_bs{i+1}'] = InitParams(
                    loc[3*i+2+current_index].detach(),
                    scale[3*i+2+current_index].detach()
                )

        # 可选 scale 初始化
        # params['init_scale'] = InitParams(loc[-1][0].view(1,1), scale[-1][0].view(1,1))

        #加入继承
        #if n > 0:
        kwargs.update(params)

        modellist.append(model1)
        guidelist.append(mean_field_guide)
        paramslist.append(params)
        paramstorelist.append(store_copy)


        # === 计算 MSE ===
        train_mse = ((y_pred1_train - y_train.cpu().numpy()) ** 2).mean()
        test_mse = ((y_pred1_test - y_test.cpu().numpy()) ** 2).mean()
        print(f"[METRIC] Combined Train MSE after layer {n+1}: {train_mse:.6f}")
        print(f"[METRIC] Combined Test MSE after layer {n+1}: {test_mse:.6f}")
        
        if plot:
            plot_output_predictions(x_train.cpu().numpy(), y_train.cpu().numpy(), y_pred1_train, name='Train'+model_token, savepath=plot_savepath)
            plot_output_predictions(x_test.cpu().numpy(), y_test.cpu().numpy(), y_pred1_test, name='Test'+model_token, savepath=plot_savepath)

        train_mselist.append(train_mse)
        test_mselist.append(test_mse)
        time1 = time.time()
        timelist.append(time1-time0)
        print(f"Time for layer {n+1}: {time1-time0:.2f} seconds")
        time0 = time1

    return modellist, train_mselist, test_mselist, timelist





def train_dsmdgp_single_multiclass(
    data, num_epochs=1000, lr=0.001, projection=None, 
    plot=False, plot_savepath=None, model_token=None,
    batch_size=1024, dim_list=None, J_list=None, 
    **kwargs
):
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

    d_out = y_train.max().item() + 1  # 类别数
    model1 = Model_dsmdgp_c_r(dim_list=dim_list, J_list=J_list, likelihood="classification", num_classes=d_out, **kwargs, out_dim=d_out, device=device, projection=projection, )
    model1.to(device)
    # print(model1)
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
    for name, value in param_store.items():  # 打印参数名称和形状 ???
        print(f"{name}: {value.shape}")
    module_depth = len(J_list)
    in_dim_list = dim_list[:-1]
    out_dim_list = dim_list[1:]
    print("in:", in_dim_list, "out:", out_dim_list) 
    len_list = []
    for i in range(module_depth):
        len_list.append(J_list[i]*in_dim_list[i])
        len_list.append(out_dim_list[i]*2*J_list[i])
        len_list.append(out_dim_list[i])
    for i in range(module_depth):
        len_list.append(J_list[i]*in_dim_list[0])
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


def train_dsmdgp_multiple_multiclass(data,  depth=2, module_depth = 1, num_epochs=1000, growth_factor = 1, lr = 0.001, batch_size = 1024, plot = False, plot_savepath=None, j = 20, hid_dim=10, initflag = False,**kwargs):
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
    data = (x_train, x_test, y_train, y_test)

    # 投影到 hid_dim 维
    xdim = x_train.shape[1]
    # if xdim < hid_dim:
    #     ncopy = hid_dim // xdim + 1
    # else:
    #     ncopy = 1
    # projectionx = torch.cat([torch.eye(xdim)] * ncopy, dim=1).to(device) @ random_projection_matrix(xdim*ncopy, hid_dim).to(device)
    out_dim = y_train.max().item() + 1  # 类别数
    projection = random_projection_matrix(hid_dim, out_dim).to(device)
    x_train, x_test, y_train, y_test = data

    time0 = time.time()
    timelist = []
    for n in range(depth):
        module_depth = n + 1
        model_token = f"layer{n+1}"
        print(f"Training layer {n+1} with token {model_token}")
        dim_list= [xdim] + [hid_dim]*(n+1)
        jx = j ### temperary set for first layer
        J_list = [jx] + [j]*(n)
        grow_num_epochs = int(num_epochs * (growth_factor ** n))


        model1, mean_field_guide, store_copy, y_pred1_train, y_pred1_test, result_l, result_s, train_acc, test_acc = train_dsmdgp_single_multiclass(
            data=data, num_epochs=grow_num_epochs, lr=lr, batch_size=batch_size, dim_list=dim_list, J_list=J_list, plot=plot, plot_savepath=plot_savepath, model_token=model_token, **kwargs,
            projection=projection, 
        )
        # print(model1)

        loc = result_l
        # print("loc is: ",loc)
        scale = result_s
        in_dim_list = dim_list[:-1]
        out_dim_list = dim_list[1:]

        # === Extract Parameters ===  
        params = {}
        # projectionx = loc[0].reshape(xdim, hid_dim).detach()
        current_index = 0
        for i in range(module_depth):
            params[f'init_1st_w{i}'] = InitParams(
                loc[3*i+0+current_index].reshape(J_list[i], in_dim_list[i]).detach(),
                scale[3*i+0+current_index].reshape(J_list[i], in_dim_list[i]).detach()
            )
            params[f'init_2nd_w{i}'] = InitParams(
                loc[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach(),
                scale[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach()
            )
            params[f'init_2nd_b{i}'] = InitParams(
                loc[3*i+2+current_index].detach(),
                scale[3*i+2+current_index].detach()
            )
        # if config['init_method'] == 'ww':
        if module_depth > 1 and initflag ==True:
            i = module_depth - 1
            params[f'init_1st_w{i+1}'] = InitParams(
                    loc[3*i+0+current_index].reshape(J_list[i], in_dim_list[i]).detach(),
                    scale[3*i+0+current_index].reshape(J_list[i], in_dim_list[i]).detach()
                )
            params[f'init_2nd_w{i+1}'] = InitParams(
                    loc[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach(),
                    scale[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach()
                )
            params[f'init_2nd_b{i+1}'] = InitParams(
                    loc[3*i+2+current_index].detach(),
                    scale[3*i+2+current_index].detach()
                )
        
            current_index = 3*(module_depth)
        else:
            current_index = 3*(module_depth - 1)

        # i = module_depth - 1
        # params[f'init_1st_w{i+1}'] = InitParams(
        #         loc[3*i+0+current_index].reshape(J_list[i], in_dim_list[i]).detach(),
        #         scale[3*i+0+current_index].reshape(J_list[i], in_dim_list[i]).detach()
        #     )
        # params[f'init_2nd_w{i+1}'] = InitParams(
        #         loc[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach(),
        #         scale[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach()
        #     )
        # params[f'init_2nd_b{i+1}'] = InitParams(
        #         loc[3*i+2+current_index].detach(),
        #         scale[3*i+2+current_index].detach()
        #     )
        # current_index = 3*(module_depth)
        for i in range(module_depth):
            params[f'init_1st_ws{i}'] = InitParams(
                loc[3*i+0+current_index].reshape(J_list[i], in_dim_list[0]).detach(),
                scale[3*i+0+current_index].reshape(J_list[i], in_dim_list[0]).detach()
            )
            params[f'init_2nd_ws{i}'] = InitParams(
                loc[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach(),
                scale[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach()
            )
            params[f'init_2nd_bs{i}'] = InitParams(
                loc[3*i+2+current_index].detach(),
                scale[3*i+2+current_index].detach()
            )
        if initflag ==True:
            i = module_depth - 1
            params[f'init_1st_ws{i+1}'] = InitParams(
                    loc[3*i+0+current_index].reshape(J_list[i], in_dim_list[0]).detach(),
                    scale[3*i+0+current_index].reshape(J_list[i], in_dim_list[0]).detach()
                )
            params[f'init_2nd_ws{i+1}'] = InitParams(
                    loc[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach(),
                    scale[3*i+1+current_index].reshape(out_dim_list[i], 2*J_list[i]).detach()
                )
            params[f'init_2nd_bs{i+1}'] = InitParams(
                    loc[3*i+2+current_index].detach(),
                    scale[3*i+2+current_index].detach()
                )
        
        # print("params:",params)
        
        # 可选 scale 初始化
        # params['init_scale'] = InitParams(loc[-1][0].view(1,1), scale[-1][0].view(1,1))

        #加入继承
        if n > 0:
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