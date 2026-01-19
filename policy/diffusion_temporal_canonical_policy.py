from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy

from canonical_policy.model.common.normalizer import LinearNormalizer
from canonical_policy.model.diffusion.canonical_conditional_unet1d import CanonicalConditionalUnet1D
from canonical_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from canonical_policy.common.pytorch_util import dict_apply
from canonical_policy.model.vision.canonical_extractor import CanonicalEncoder
from canonical_policy.policy.base_image_policy import BaseImagePolicy
from canonical_policy.model.vision.temporally_aware_canonical_transformer import TemporallyAwareCanonicalTransformer
from pytorch3d.transforms import matrix_to_quaternion


class DiffusionTemporalCanonicalPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
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
            transformer_cfg=None,
            **kwargs):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        self.obs_encoder = CanonicalEncoder(observation_space=obs_dict,
                                            out_channel=encoder_output_dim,
                                            canonical_encoder_cfg=canonical_encoder_cfg,
                                            use_pc_color=use_pc_color,
                                            pointnet_type=pointnet_type,)

        obs_feature_dim = self.obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionTemporalCanonicalPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        self.model = CanonicalConditionalUnet1D(
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

        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # temporal transformer for rotation smoothing
        if transformer_cfg is None:
            transformer_cfg = {}
        self.tact = TemporallyAwareCanonicalTransformer(**transformer_cfg)

    # ===== util =====
    def _estimate_quat_from_sequence(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if 'point_cloud' not in nobs:
            raise RuntimeError('point_cloud not found for temporal canonical transformer')
        pc = nobs['point_cloud'][:, :self.n_obs_steps, ...]  # [B, To, N, C]
        pc_xyz = pc[..., :3]
        R = self.tact(pc_xyz)  # [B, 3, 3]
        quat = matrix_to_quaternion(R)  # [B, 4] wijk
        # repeat for B*To to match downstream interface
        B = pc.shape[0]
        To = self.n_obs_steps
        quat_rep = quat.unsqueeze(1).repeat(1, To, 1).reshape(B * To, 4)
        return quat_rep

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            points_center,
            est_quat,
            local_cond=None,
            global_cond=None,
            generator=None,
            **kwargs):
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond,
                                global_cond=global_cond,
                                points_center=points_center,
                                est_quat=est_quat)
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'robot0_eye_in_hand_image' in obs_dict:
            del obs_dict['robot0_eye_in_hand_image']
        if 'agentview_image' in obs_dict:
            del obs_dict['agentview_image']
        nobs = self.normalizer.normalize(obs_dict)
        nobs['robot0_eef_quat'] = obs_dict['robot0_eef_quat']

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None

        # build features
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        ret = self.obs_encoder(this_nobs)
        global_cond = ret['final_feat'].reshape(B, -1)

        # override est_quat with temporally smoothed rotation
        est_quat_override = self._estimate_quat_from_sequence(nobs)  # [B*To, 4]
        ret['est_quat'] = est_quat_override

        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            points_center=ret['points_center'],
            est_quat=ret['est_quat'],
            **self.kwargs)

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

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
        nobs = self.normalizer.normalize(batch['obs'])
        nobs['robot0_eef_quat'] = batch['obs']['robot0_eef_quat']
        nactions = self.normalizer['action'].normalize(batch['action'])

        batch_size = nactions.shape[0]
        trajectory = nactions
        cond_data = trajectory

        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        ret = self.obs_encoder(this_nobs)
        global_cond = ret['final_feat'].reshape(batch_size, -1)

        # override est_quat by temporal smoother
        est_quat_override = self._estimate_quat_from_sequence(nobs)  # [B*To, 4]
        ret['est_quat'] = est_quat_override

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(sample=noisy_trajectory, 
                          timestep=timesteps, 
                          local_cond=None, 
                          global_cond=global_cond,
                          points_center=ret['points_center'],
                          est_quat=ret['est_quat'])

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.l1_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        loss_dict = {
                'bc_loss': loss.item(),
            }
        return loss, loss_dict 