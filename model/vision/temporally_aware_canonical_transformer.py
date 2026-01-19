import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.vision.canonical_utils.vec_pointnet import VecPointNet, VN_Regressor
from model.vision.canonical_utils.utils import construct_rotation_matrix


class EquivariantRotationEstimator(nn.Module):
    def __init__(self, pc_feature_dim: int = 128, regressor_hidden_dim: int = 256, num_layers: int = 2, ksize: int = 16):
        super().__init__()
        self.point_encoder = VecPointNet(h_dim=pc_feature_dim, c_dim=pc_feature_dim, num_layers=num_layers, ksize=ksize)
        self.regressor = VN_Regressor(pc_feat_dim=pc_feature_dim)

    def forward(self, points_xyz: torch.Tensor) -> torch.Tensor:
        feats = self.point_encoder(points_xyz)
        v1, v2 = self.regressor(feats)
        rot_mat = construct_rotation_matrix(v1, v2)
        return rot_mat


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_length: int):
        super().__init__()
        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.pe[indices]


class TemporallyAwareCanonicalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_time_steps: int = 128,
        max_num_points: int = 4096,
        vn_feature_dim: int = 128,
        vn_layers_init: int = 2,
        vn_layers_final: int = 2,
        vn_ksize: int = 16,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.psi_init = EquivariantRotationEstimator(
            pc_feature_dim=vn_feature_dim, regressor_hidden_dim=256, num_layers=vn_layers_init, ksize=vn_ksize
        )
        self.psi_final = EquivariantRotationEstimator(
            pc_feature_dim=vn_feature_dim, regressor_hidden_dim=256, num_layers=vn_layers_final, ksize=vn_ksize
        )

        self.input_projection = nn.Linear(3, d_model)
        self.output_projection = nn.Linear(d_model, 3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.time_pos_encoding = SinusoidalPositionEncoding(d_model, max_time_steps)
        self.point_pos_encoding = SinusoidalPositionEncoding(d_model, max_num_points)

    def _apply_initial_canonicalization(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, n, _ = x_seq.shape
        x_bt_n3 = x_seq.reshape(b * t, n, 3)
        rot_bt = self.psi_init(x_bt_n3)
        x_cn_bt_n3 = torch.matmul(x_bt_n3, rot_bt.transpose(1, 2))
        x_cn = x_cn_bt_n3.reshape(b, t, n, 3)
        return x_cn, rot_bt.reshape(b, t, 3, 3)

    def _temporal_fusion(self, x_cn_seq: torch.Tensor) -> torch.Tensor:
        b, t, n, _ = x_cn_seq.shape
        x_bt_n3 = x_cn_seq.reshape(b * t, n, 3)
        token = self.input_projection(x_bt_n3)
        device = token.device
        point_indices = torch.arange(n, device=device).unsqueeze(0).expand(b * t, -1)
        time_indices = torch.arange(t, device=device).unsqueeze(0).expand(b, -1).reshape(b * t)
        pos_spatial = self.point_pos_encoding(point_indices)
        pos_temporal = self.time_pos_encoding(time_indices).unsqueeze(1).expand(b * t, n, -1)
        token = token + pos_spatial + pos_temporal
        if self.use_gradient_checkpointing and self.training:
            # Apply gradient checkpointing layer-by-layer to save memory
            for layer in self.transformer_encoder.layers:
               token = checkpoint(layer, token)
            if self.transformer_encoder.norm is not None:
                token = self.transformer_encoder.norm(token)
        else:
            token = self.transformer_encoder(token)
        x_bt_n3_hat = self.output_projection(token)
        x_hat_last = x_bt_n3_hat.reshape(b, t, n, 3)[:, -1]
        return x_hat_last

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x_cn_seq, _ = self._apply_initial_canonicalization(x_seq)
        x_cn_hat_last = self._temporal_fusion(x_cn_seq)
        rot_final = self.psi_final(x_cn_hat_last)
        return rot_final
