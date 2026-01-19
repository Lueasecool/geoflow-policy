import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import wandb

# ================== 配置参数 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 28
channels = 1
batch_size = 64
lr = 3e-4
epochs = 100
num_classes = 10

# CurveFlow
curveflow_lambda = 0.1  # 曲率正则化权重
curveflow_m_grid_points = 100  # 曲率计算网格点数


def preprocess(x):
    return 2 * x - 1  # [-1, 1] 归一化


# ================== 数据加载 ==================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(preprocess)  # [-1, 1] 归一化
])


class CurveFlowCoefficientNet(nn.Module):
    """神经网络直接参数化CurveFlow系数（输出默认tanh到[-1,1]）"""

    def __init__(self, hidden_dim=64, output_activation="tanh"):
        super().__init__()
        self.output_activation = output_activation

        # 使用更深的网络来直接学习系数函数
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t):
        y = self.net(t)
        if self.output_activation == "tanh":
            return torch.tanh(y)
        elif self.output_activation == "sigmoid":
            return torch.sigmoid(y)
        elif self.output_activation == "softplus":
            return F.softplus(y)
        else:
            return y


class CurveFlow(nn.Module):
    """
    实现CurveFlow轨迹 z_t = a_phi(t) * x_0 + b_psi(t) * epsilon。
    系数 a_phi(t) 和 b_psi(t) 采用线性基线 + 有限幅度的非线性扰动。
    """

    def __init__(self, hidden_dim=64, delta_scale=0.5):
        super().__init__()
        # 直接参数化系数函数（tanh输出，零均值对称）
        self.a_phi_net = CurveFlowCoefficientNet(hidden_dim, output_activation="tanh")
        self.b_psi_net = CurveFlowCoefficientNet(hidden_dim, output_activation="tanh")
        self.delta_scale = delta_scale

    def a_phi(self, t):
        """线性基线(1-t) + 有限幅度扰动 t(1-t)*delta*raw_a"""
        raw_a = self.a_phi_net(t)  # [-1,1]
        return (1 - t) + t * (1 - t) * (self.delta_scale * raw_a)

    def b_psi(self, t):
        """线性基线 t + 有限幅度扰动 t(1-t)*delta*raw_b"""
        raw_b = self.b_psi_net(t)  # [-1,1]
        return t + t * (1 - t) * (self.delta_scale * raw_b)

    def forward(self, t):
        """计算给定时间t的两个系数"""
        return self.a_phi(t), self.b_psi(t)


# ================== 模型架构 ==================
class ConditionedDoubleConv(nn.Module):
    """带条件注入"""

    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels + cond_dim, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x, cond):
        x = F.silu(self.norm1(self.conv1(x)))
        cond = cond.expand(-1, -1, x.size(2), x.size(3))  # 动态广播条件
        x = torch.cat([x, cond], dim=1)
        return F.silu(self.norm2(self.conv2(x)))


class Down(nn.Module):
    """下采样"""

    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ConditionedDoubleConv(in_channels, out_channels, cond_dim)

    def forward(self, x, cond):
        x = self.maxpool(x)
        return self.conv(x, cond)


class Up(nn.Module):
    """上采样模块"""

    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConditionedDoubleConv(in_channels, out_channels, cond_dim)

    def forward(self, x1, x2, cond):
        x1 = self.up(x1)
        # 尺寸对齐
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, cond)


class ConditionalUNet(nn.Module):
    """维度安全的条件生成UNet"""

    def __init__(self):
        super().__init__()
        # 统一条件编码维度
        self.t_dim = 16
        self.label_dim = 16
        self.cond_dim = self.t_dim + self.label_dim  # 32

        #
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, self.t_dim)
        )
        # 标签嵌入
        self.label_embed = nn.Embedding(num_classes, self.label_dim)

        # 编码路径
        self.inc = ConditionedDoubleConv(1, 64, self.cond_dim)
        self.down1 = Down(64, 128, self.cond_dim)
        self.down2 = Down(128, 256, self.cond_dim)

        # 解码路径
        self.up1 = Up(256 + 128, 128, self.cond_dim)  # 输入通道修正
        self.up2 = Up(128 + 64, 64, self.cond_dim)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, t, labels):
        # 条件编码 (统一维)
        t_emb = self.time_embed(t.view(-1, 1))  # [B, 16]
        lbl_emb = self.label_embed(labels)  # [B, 16]
        cond = torch.cat([t_emb, lbl_emb], dim=1)  # [B, 32]
        cond = cond.unsqueeze(-1).unsqueeze(-1)  # [B, 32, 1, 1]

        # 编码器
        x1 = self.inc(x, cond)
        x2 = self.down1(x1, cond)
        x3 = self.down2(x2, cond)

        # 解码器
        x = self.up1(x3, x2, cond)
        x = self.up2(x, x1, cond)
        return self.outc(x)


# ================== CurveFlow训练器 ==================
class CurveFlowTrainer:
    def __init__(self, model, curveflow_model, device):
        self.model = model
        self.curveflow_model = curveflow_model
        self.device = device

        # 预计算曲率网格
        self.M = curveflow_m_grid_points
        self.dt = 1.0 / self.M
        self.t_grid = torch.linspace(0, 1, self.M + 1, device=self.device).view(-1, 1)
        self.t_grid.requires_grad_(True)  # 需要梯度以计算高阶导数

    def compute_curvature_loss(self):
        """使用自动微分计算曲率正则化损失"""
        if curveflow_lambda <= 0:
            return torch.tensor(0.0, device=self.device)

        # 计算系数
        a_grid, b_grid = self.curveflow_model(self.t_grid)

        # 计算一阶导数
        grad_outputs = torch.ones_like(a_grid)
        dot_a = torch.autograd.grad(
            outputs=a_grid,
            inputs=self.t_grid,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        dot_b = torch.autograd.grad(
            outputs=b_grid,
            inputs=self.t_grid,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        # 计算二阶导数
        ddot_a = torch.autograd.grad(
            outputs=dot_a,
            inputs=self.t_grid,
            grad_outputs=torch.ones_like(dot_a),
            create_graph=True,
            retain_graph=True
        )[0]

        ddot_b = torch.autograd.grad(
            outputs=dot_b,
            inputs=self.t_grid,
            grad_outputs=torch.ones_like(dot_b),
            create_graph=True,
            retain_graph=True
        )[0]

        # 计算曲率被积函数 (dot_a * ddot_b - dot_b * ddot_a)^2
        integrand = (dot_a * ddot_b - dot_b * ddot_a) ** 2

        # 积分近似 (黎曼和)
        integral = torch.sum(integrand) * self.dt

        return curveflow_lambda * integral


# ================== 训练与生成 ==================
model = ConditionalUNet().to(device)
curveflow_model = CurveFlow().to(device)
trainer = CurveFlowTrainer(model, curveflow_model, device)

# 优化器 - 同时优化UNet和CurveFlow模型
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(curveflow_model.parameters()),
    lr=lr
)


@torch.no_grad()
def generate_with_label(label, num_samples=16, device="cuda"):
    """使用CurveFlow生成指定标签的样本"""
    model.eval()
    curveflow_model.eval()

    # 初始噪声和标签 (对应 t=1)
    xT = torch.randn(num_samples, 1, 28, 28, device=device)
    labels = torch.full((num_samples,), label, device=device, dtype=torch.long)

    # 定义ODE函- 使用CurveFlow轨迹
    def ode_func(t: torch.Tensor, x: torch.Tensor):
        t_expanded = t.expand(x.size(0))  # [1] -> [num_samples]

        # 速度场由模型预测（已在训练中学习到 a,b 的隐式影响）
        vt = model(x, t_expanded, labels)
        return vt

    # 时间点：从 1 -> 0（从噪声到数据）
    t_eval = torch.linspace(1.0, 0.0, 25, device=device)

    # 解ODE（自适应步长）
    generated = odeint(
        ode_func,
        xT,
        t_eval,
        rtol=1e-5,
        atol=1e-5,
        method='dopri5'
    )

    # 后处理
    images = (generated[-1].clamp(-1, 1) + 1) / 2  # [0,1]
    return images.cpu().squeeze(1)  # 移除通道维度


def visualize_samples(samples, title="Generated Samples"):
    """可视化生成结果"""
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(samples[i].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
    plt.suptitle(title)
    # 不显示窗口
    # plt.show()
    os.makedirs("./data/output", exist_ok=True)
    plt.savefig(os.path.join("./data/output", "samples.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_100_digits(image_size=28, device="cuda", epoch=None, save_dir="./data/output"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))

    # 为0-9生成10张图
    for label in range(10):
        # 生成当前数字的10个样本
        generated = generate_with_label(
            label=label,
            num_samples=10
        ).numpy()  # 形状 (10, 28, 28)

        # 在当前行绘制
        for i in range(10):
            ax = plt.subplot(10, 10, i * 10 + 1 + label)
            plt.imshow(generated[i], cmap='gray')
            ax.axis('off')
            # 在每列第一行添加标签
            if i == 0:
                ax.text(14, -10, str(label), fontsize=20, ha='center')

    plt.tight_layout()
    # 保存到文件，不显示窗口
    filename = f"epoch_{epoch:04d}.png" if epoch is not None else "grid.png"
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def train():
    """CurveFlow训练循环"""
    wandb.init(project="curveflow-mnist", config={
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "curveflow_lambda": curveflow_lambda,
        "curveflow_m_grid_points": curveflow_m_grid_points,
    })
    for epoch in range(epochs):
        model.train()
        curveflow_model.train()
        total_loss = 0
        total_curvature_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            bsz = images.size(0)

            # 生成噪声
            noise = torch.randn_like(images)

            # CurveFlow: 使用 Beta(0.5, 0.5) 分布采样时间步（两端更密集）
            from torch.distributions import Beta
            beta_dist = Beta(torch.tensor([0.5], device=device), torch.tensor([0.5], device=device))
            t_samples = beta_dist.sample((bsz,)).squeeze(-1)

            # 计算CurveFlow系数和导数
            t_nn_input = t_samples.clone().view(-1, 1).requires_grad_()
            a_t_scalar, b_t_scalar = curveflow_model(t_nn_input)

            # 计算导数
            grad_outputs = torch.ones_like(a_t_scalar)
            dot_a_val = torch.autograd.grad(
                outputs=a_t_scalar,
                inputs=t_nn_input,
                grad_outputs=grad_outputs,
                create_graph=True,
            )[0]
            dot_b_val = torch.autograd.grad(
                outputs=b_t_scalar,
                inputs=t_nn_input,
                grad_outputs=grad_outputs,
                create_graph=True,
            )[0]

            # 重塑为广播形状
            a_t = a_t_scalar.view(-1, 1, 1, 1)
            b_t = b_t_scalar.view(-1, 1, 1, 1)
            dot_a = dot_a_val.view(-1, 1, 1, 1)
            dot_b = dot_b_val.view(-1, 1, 1, 1)

            # 计算CurveFlow轨迹
            noisy_latents = a_t * images + b_t * noise
            target = dot_a * images + dot_b * noise

            # 前向计算
            vt_pred = model(noisy_latents, t_samples, labels)

            # Flow Matching损失
            fm_loss = F.mse_loss(vt_pred, target)

            # 曲率正则化损失 (使用自动微分)
            curvature_loss = trainer.compute_curvature_loss()

            # 总损失
            loss = fm_loss + curvature_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(curveflow_model.parameters()), 2.0
            )
            optimizer.step()

            total_loss += fm_loss.item()
            total_curvature_loss += curvature_loss.item()

            pbar.set_postfix(loss=f"{fm_loss.item():.6f}", curv=f"{curvature_loss.item():.6f}")

        # 每10个epoch生成示例并保存
        fm_avg = total_loss / len(train_loader)
        curv_avg = total_curvature_loss / len(train_loader)
        total_avg = fm_avg + curv_avg
        wandb.log({
            "epoch": epoch,
            "fm_loss_avg": fm_avg,
            "curv_loss_avg": curv_avg,
            "total_loss_avg": total_avg,
        }, step=epoch)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - FM Loss: {fm_avg:.6f}, Curvature Loss: {curv_avg:.6f}, Total: {total_avg:.6f}")
            grid_path = plot_100_digits(epoch=epoch, save_dir="./data/output")
            wandb.log({"samples/grid": wandb.Image(grid_path), "epoch": epoch}, step=epoch)

    wandb.finish()


if __name__ == "__main__":
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    train()