from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops

from model.common.normalizer import LinearNormalizer
from model.diffusion.canonical_conditional_unet1d import CanonicalConditionalUnet1D
from model.diffusion.mask_generator import LowdimMaskGenerator
from common.pytorch_util import dict_apply
from model.vision.canonical_extractor import CanonicalEncoder
from policy.base_image_policy import BaseImagePolicy

# Flow Matching
try:
    from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
except Exception as e:
    raise ImportError("Please install torchcfm to use CFMCanonicalUNetPolicy. Error: {}".format(e))


class CurveFlowCoefficientNet(nn.Module):
    """神经网络直接参数化CurveFlow系数"""

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


class CurveFlowCanonicalUNetPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=50,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            encoder_output_dim=256,
            use_pc_color=False,
            pointnet_type="cp_so2",
            canonical_encoder_cfg=None,
            cond_predict_scale=True,
            # CFM-specific
            fm_sigma=0.02,
            t_discrete_steps=1000,
            use_integer_time_embed=True,
            time_steps_for_embed=1000,
            x0_std=1.0,
            t_embed_scale=1000.0,
            x_t_noise_sigma0=0.05,
            min_t_epsilon=0.01,
            # Switch
            use_curveflow=True,
            # CurveFlow parameters
            curveflow_lambda=0.1,
            curveflow_hidden_dim=64,
            curveflow_delta_scale=0.5,
            curveflow_m_grid_points=1000,
            # parameters passed to step
            **kwargs):
        super().__init__()
        
        # 初始化训练步数计数器
        self.global_step = 0

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        obs_encoder = CanonicalEncoder(observation_space=obs_dict,
                                        out_channel=encoder_output_dim,
                                        canonical_encoder_cfg=canonical_encoder_cfg,
                                        use_pc_color=use_pc_color,
                                        pointnet_type=pointnet_type,)

        # create model (same架构，但语义从噪声改为速度场)
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
        
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        cprint(f"[CurveFlowCanonicalUNetPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        model = CanonicalConditionalUnet1D(
                    input_dim=input_dim,
                    n_obs_steps=n_obs_steps,
                    local_cond_dim=None,
                    global_cond_dim=global_cond_dim,
                    diffusion_step_embed_dim=diffusion_step_embed_dim,
                    down_dims=down_dims,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
        )

        # CurveFlow模型（按开关启用）
        curveflow_model = None
        self.use_curveflow = use_curveflow
        if self.use_curveflow:
            curveflow_model = CurveFlow(
                hidden_dim=curveflow_hidden_dim,
                delta_scale=curveflow_delta_scale
            )

        self.obs_encoder = obs_encoder
        self.model = model
        self.curveflow_model = curveflow_model
        
        # Flow Matching
        self.fm = ConditionalFlowMatcher(sigma=fm_sigma)

        # 常用属性
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        # CFM/推理相关超参
        self.num_inference_steps = num_inference_steps
        self.t_discrete_steps = t_discrete_steps
        self.use_integer_time_embed = use_integer_time_embed
        self.time_steps_for_embed = time_steps_for_embed
        self.x0_std = x0_std
        self.t_embed_scale = t_embed_scale
        self.x_t_noise_sigma0 = x_t_noise_sigma0
        self.min_t_epsilon = min_t_epsilon

        # CurveFlow参数
        self.curveflow_lambda = curveflow_lambda
        self.curveflow_hidden_dim = curveflow_hidden_dim
        self.curveflow_delta_scale = curveflow_delta_scale
        self.curveflow_m_grid_points = curveflow_m_grid_points

        # 测试曲率损失计算将在模型移动到设备后进行

        # 掩码生成器（仅在 use_curveflow=False 的 FM 分支使用）
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )


    def _compute_curvature_loss(self):
        """计算曲率正则化损失 - 使用有限差分提高数值稳定性"""
        if (not hasattr(self, 'use_curveflow')) or (not self.use_curveflow):
            return torch.tensor(0.0, requires_grad=True, device=self.device if hasattr(self, 'device') else None)
        if (self.curveflow_lambda <= 0) or (self.curveflow_model is None):
            return torch.tensor(0.0, requires_grad=True)

        # 获取设备
        device = next(self.curveflow_model.parameters()).device
        
        # 创建时间网格 - 增加密度提高精度
        M = max(100, self.curveflow_m_grid_points // 2)  # 减少计算量但保持精度
        dt = 1.0 / M
        t_grid = torch.linspace(0, 1, M + 1, device=device).view(-1, 1)
        
        # 计算系数
        a_grid, b_grid = self.curveflow_model(t_grid)
        
        # 使用有限差分计算导数，避免高阶自动微分
        # 一阶导数
        dot_a = (a_grid[1:] - a_grid[:-1]) / dt
        dot_b = (b_grid[1:] - b_grid[:-1]) / dt
        
        # 二阶导数
        ddot_a = (dot_a[1:] - dot_a[:-1]) / dt
        ddot_b = (dot_b[1:] - dot_b[:-1]) / dt
        
        # 计算曲率被积函数 (dot_a * ddot_b - dot_b * ddot_a)^2
        # 注意：需要对齐长度，ddot_a 比 dot_a 短 1
        min_len = min(len(dot_a) - 1, len(ddot_a))
        if min_len > 0:
            integrand = (dot_a[:min_len] * ddot_b[:min_len] - dot_b[:min_len] * ddot_a[:min_len]) ** 2
            # 积分近似 (黎曼和)
            integral = torch.sum(integrand) * dt
        else:
            integral = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 添加数值稳定性检查
        if torch.isnan(integral) or torch.isinf(integral):
            integral = torch.tensor(0.0, device=device, requires_grad=True)

        return self.curveflow_lambda * integral

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            points_center,
            est_quat,
            condition_data_pc=None,
            condition_mask_pc=None,
            local_cond=None,
            global_cond=None,
            generator=None,
            **kwargs
            ):
        # 反向积分：t: 1 -> 0，dx/dt = -v_theta(x, t, cond)
        device = condition_data.device
        dtype = condition_data.dtype

        # 初始状态 x(t=1) = b(1) * N(0,1) （若无curveflow则退化为0.1*N）
        x = torch.randn(size=condition_data.shape, dtype=dtype, device=device)
        if hasattr(self, 'curveflow_model') and self.curveflow_model is not None and getattr(self, 'use_curveflow', False):
            t1 = torch.ones(1, 1, device=device, dtype=dtype)
            with torch.no_grad():
                _, b1 = self.curveflow_model(t1)
            b1 = b1.view(1, 1, 1)
            x = x * b1*0.1
        else:
            x = x * 0.1

        # 时间从 1→0 均匀离散（反向积分）
        steps = max(2, int(self.num_inference_steps))
        ts = torch.linspace(1.0, 0.0, steps=steps, device=device)
        dt = 1.0 / (steps - 1)

        for i, s in enumerate(ts):
            # 强制使用整数时间嵌入（1000步）
            Tsteps = 1000
            s_idx = torch.clamp((s * (Tsteps - 1)).long(), 0, Tsteps - 1)
            timestep_in = s_idx
            

            v = self.model(sample=x,
                           timestep=timestep_in,
                           local_cond=local_cond,
                           global_cond=global_cond,
                           points_center=points_center,
                           est_quat=est_quat)

            # 反向积分：x = x - v * dt
            x = x - v * dt

        # 与CFM一致：在条件位置对齐（通常为全零）
        x[condition_mask] = condition_data[condition_mask]
        return x

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        if 'robot0_eye_in_hand_image' in obs_dict:
            del obs_dict['robot0_eye_in_hand_image']
        if 'agentview_image' in obs_dict:
            del obs_dict['agentview_image']
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        nobs['robot0_eef_quat'] = obs_dict['robot0_eef_quat']

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        ret = self.obs_encoder(this_nobs)

        # reshape back to B, Do
        global_cond = ret['final_feat'].reshape(B, -1)

        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling (ODE正向积分)
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            points_center=ret['points_center'],
            est_quat=ret['est_quat'],
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        
        if 'robot0_eye_in_hand_image' in batch['obs']:
            del batch['obs']['robot0_eye_in_hand_image']
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nobs['robot0_eef_quat'] = batch['obs']['robot0_eef_quat']
        nactions = self.normalizer['action'].normalize(batch['action'])

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        ret = self.obs_encoder(this_nobs)

        # reshape back to B, Do
        global_cond = ret['final_feat'].reshape(batch_size, -1)  # [B, To*Do]

        # CurveFlow / 标准Flow Matching 分支
        # 采样时间
        t = torch.rand(batch_size, device=self.device)
        if self.min_t_epsilon is not None and self.min_t_epsilon > 0:
            eps = float(self.min_t_epsilon)
            t = t.clamp_(eps, 1.0 - eps)

        if self.use_curveflow:
            # CurveFlow路径与目标
            t_input = t.view(-1, 1).requires_grad_()
            a_t_scalar, b_t_scalar = self.curveflow_model(t_input)

            grad_outputs = torch.ones_like(a_t_scalar)
            dot_a_val = torch.autograd.grad(
                outputs=a_t_scalar,
                inputs=t_input,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            dot_b_val = torch.autograd.grad(
                outputs=b_t_scalar,
                inputs=t_input,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]

            a_t = a_t_scalar.view(-1, 1, 1)
            b_t = b_t_scalar.view(-1, 1, 1)
            dot_a = dot_a_val.view(-1, 1, 1)
            dot_b = dot_b_val.view(-1, 1, 1)

            noise = torch.randn_like(trajectory)*0.1
            x_t = a_t * trajectory + b_t * noise
            u_t = dot_a * trajectory + dot_b * noise
            condition_mask = self.mask_generator(trajectory.shape)
            loss_mask = ~condition_mask
            x_t[condition_mask] = trajectory[condition_mask]
        else:
            # 与 CFM 完全一致：从 x0 ~ N(0, x0_std) 到 x1=trajectory，由 fm 采样 (x_t, u_t)
            x0 = torch.randn_like(trajectory) * self.x0_std
            x1 = trajectory
            _, x_t, u_t = self.fm.sample_location_and_conditional_flow(x0, x1, t)
            # 掩码与条件注入
            condition_mask = self.mask_generator(trajectory.shape)
            loss_mask = ~condition_mask
            x_t[condition_mask] = trajectory[condition_mask]

        # 5) 时间嵌入
       
        if self.use_integer_time_embed:
            Tsteps = int(self.time_steps_for_embed)
            t_idx = torch.clamp((t * (Tsteps - 1)).long(), 0, Tsteps - 1)
            timestep_in = t_idx
        else:
            timestep_in = t * self.t_embed_scale
       

        # 6) 速度场预测 
        v_pred = self.model(sample=x_t, 
                            timestep=timestep_in, 
                            local_cond=local_cond, 
                            global_cond=global_cond,
                            points_center=ret['points_center'],
                            est_quat=ret['est_quat'])

       
            # 与原先一致：MSE
        per_elem = F.l1_loss(v_pred, u_t, reduction='none')
        v_scale = torch.std(u_t, dim=-1, keepdim=True).clamp(min=1e-6)
        per_elem = per_elem / v_scale
        per_elem = per_elem * loss_mask.type(per_elem.dtype)
        fm_loss = reduce(per_elem, 'b ... -> b (...)', 'mean').mean()
       

        # 8) 曲率正则化损失（仅在启用CurveFlow时生效）
        curvature_loss = self._compute_curvature_loss()

        # 9) 总损失
        total_loss = fm_loss + curvature_loss
        
        # 更新训练步数
        self.global_step += 1
        
        # 注意：梯度计算需要在backward()之后进行，这里只是准备调试信息

        # 添加调试信息
        if self.global_step % 100 == 0:
            print(f"Step {self.global_step}:")
            print(f"  t range: [{t.min().item():.3f}, {t.max().item():.3f}]")
            if self.use_curveflow:
                print(f"  a_t range: [{a_t.min().item():.3f}, {a_t.max().item():.3f}]")
                print(f"  b_t range: [{b_t.min().item():.3f}, {b_t.max().item():.3f}]")
                print(f"  dot_a range: [{dot_a.min().item():.3f}, {dot_a.max().item():.3f}]")
                print(f"  dot_b range: [{dot_b.min().item():.3f}, {dot_b.max().item():.3f}]")
            print(f"  x_t std: {x_t.std().item():.3f}")
            print(f"  u_t std: {u_t.std().item():.3f}")
            print(f"  v_pred std: {v_pred.std().item():.3f}")
            print(f"  fm_loss: {fm_loss.item():.3f}")
            print(f"  curvature_loss: {curvature_loss.item():.3f}")
            print(f"  total_loss: {total_loss.item():.3f}")
            
            # 检查CurveFlow模型参数 
            if self.use_curveflow:
                total_params = sum(p.numel() for p in self.curveflow_model.parameters())
                zero_params = sum(1 for p in self.curveflow_model.parameters() if torch.all(p == 0))
                param_norm = sum(p.norm().item() for p in self.curveflow_model.parameters())
                print(f"  CurveFlow params: {total_params}, zero params: {zero_params}, param norm: {param_norm:.6f}")

        loss_dict = {
            'bc_loss': total_loss.item(),
            'fm_loss': fm_loss.item(),
            'curvature_loss': curvature_loss.item(),
            't_mean': t.mean().item(),
            't_std': t.std().item(),
            'v_pred_std': v_pred.std().item(),
            'target_std': u_t.std().item(),
        }
        return total_loss, loss_dict