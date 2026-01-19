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

from canonical_policy.model.common.normalizer import LinearNormalizer
from canonical_policy.model.diffusion.canonical_conditional_unet1d import CanonicalConditionalUnet1D
from canonical_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from canonical_policy.common.pytorch_util import dict_apply
from canonical_policy.model.vision.canonical_extractor import CanonicalEncoder
from canonical_policy.policy.base_image_policy import BaseImagePolicy

# Flow Matching
try:
    from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
except Exception as e:
    raise ImportError("Please install torchcfm to use CFMCanonicalUNetPolicy. Error: {}".format(e))


class CFMCanonicalUNetPolicy(BaseImagePolicy):
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
            use_integer_time_embed=False,
            time_steps_for_embed=1000,
            x0_std=1.0,
            t_embed_scale=1000.0,
            x_t_noise_sigma0=0.05,
            min_t_epsilon=0.01,
            # parameters passed to step
            **kwargs):
        super().__init__()

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
        
        cprint(f"[CFMUnetCanonicalPolicy] pointnet_type: {self.pointnet_type}", "yellow")

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

        self.obs_encoder = obs_encoder
        self.model = model

        # mask generator（与 diffusion 版保持一致，用于条件/可见步掩码）
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
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
        # ODE 正向积分：dx/dt = v_theta(x, t, cond)
        device = condition_data.device
        dtype = condition_data.dtype

        # 尝试更小的初始噪声
        x = torch.randn(size=condition_data.shape, dtype=dtype, device=device) * 0.1  # 减小 x0_std

        # 时间从 0→1 均匀离散
        steps = max(2, int(self.num_inference_steps))
        ts = torch.linspace(0.0, 1.0, steps=steps, device=device)
        dt = 1.0 / (steps - 1)

        for i, s in enumerate(ts):
            if self.use_integer_time_embed:
                s_idx = torch.clamp((s * (self.time_steps_for_embed - 1)).long(), 0, self.time_steps_for_embed - 1)
                timestep_in = s_idx
            else:
                timestep_in = s * self.t_embed_scale

            v = self.model(sample=x,
                           timestep=timestep_in,
                           local_cond=local_cond,
                           global_cond=global_cond,
                           points_center=points_center,
                           est_quat=est_quat)

            # 注意：在推理中，condition_mask 通常是全零的
            # 所以这里不需要特殊处理

            x = x + v * dt

        # 最终确保条件满足
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
        cond_data = trajectory  # 训练时的条件与轨迹一致（与原版一致）

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        ret = self.obs_encoder(this_nobs)

        # reshape back to B, Do
        global_cond = ret['final_feat'].reshape(batch_size, -1)  # [B, To*Do]

        # generate impainting mask（与原逻辑一致）
        condition_mask = self.mask_generator(trajectory.shape)
        loss_mask = ~condition_mask

        # Flow Matching 训练
        # 1) 连续时间 t ∈ (ε, 1-ε) - 使用更均匀的分布
        t = torch.rand(batch_size, device=trajectory.device)
        if self.min_t_epsilon is not None and self.min_t_epsilon > 0:
            eps = float(self.min_t_epsilon)
            t = t.clamp_(eps, 1.0 - eps)

        # 2) 高斯起点 x0，与真实 x1=trajectory
        x0 = torch.randn_like(trajectory) * self.x0_std
        x1 = trajectory

        # 3) 采样 (x_t, u_t)
        _, x_t, u_t = self.fm.sample_location_and_conditional_flow(x0, x1, t)


        x_t[condition_mask] = cond_data[condition_mask]
        
        
        # 4) 时间嵌入：确保与训练时完全一致
        if self.use_integer_time_embed:
            Tsteps = int(self.time_steps_for_embed)
            t_idx = torch.clamp((t * (Tsteps - 1)).long(), 0, Tsteps - 1)
            timestep_in = t_idx
        else:
            timestep_in = t * self.t_embed_scale

        # 5) 速度场预测 
        v_pred = self.model(sample=x_t, 
                            timestep=timestep_in, 
                            local_cond=local_cond, 
                            global_cond=global_cond,
                            points_center=ret['points_center'],
                            est_quat=ret['est_quat'])

        # 6) 直接计算损失，不使用条件掩码
        # Flow Matching 中，条件掩码主要用于数据增强，而不是强制约束
        loss = F.l1_loss(v_pred, u_t, reduction='none')
        
        # 可选：添加速度场尺度归一化
        v_scale = torch.std(u_t, dim=-1, keepdim=True).clamp_(min=1e-6)
        loss = loss / v_scale
        
        # 应用损失掩码（只在非条件位置计算损失）
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        # 添加调试信息
        if hasattr(self, 'global_step') and self.global_step % 100 == 0:
            print(f"Step {self.global_step}:")
            print(f"  t range: [{t.min().item():.3f}, {t.max().item():.3f}]")
            print(f"  x_t std: {x_t.std().item():.3f}")
            print(f"  u_t std: {u_t.std().item():.3f}")
            print(f"  v_pred std: {v_pred.std().item():.3f}")
            print(f"  loss: {loss.item():.3f}")

        loss_dict = {
            'bc_loss': loss.item(),
            't_mean': t.mean().item(),
            't_std': t.std().item(),
            'v_pred_std': v_pred.std().item(),
            'u_t_std': u_t.std().item(),
        }
        return loss, loss_dict