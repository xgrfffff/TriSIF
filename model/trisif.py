"""TriSIF stage 2: cloud--time-series fusion for irradiance forecasting.

Copy this file and cloud_predictor.py together; only PyTorch is required.
Pass a pretrained CloudPredictor to TriSIF(cloud_predictor). The complete cloud
predictor stays frozen and in eval mode, including when TriSIF.train() is called.

Inputs: sky/satellite history [B,T,C,H,W], irradiance history [B,L,4] in the
order [DNI,DHI,CSGHI,GHI], future CSGHI [B,F,1], solar masks [B,F,1,H,W].
All inputs are already aligned and normalized; masks are in [0,1].
Output: (GHI [B,F,1], optional scalar MSE, metrics). Predictions and GHI targets
use the same normalization. No future sky/satellite observations are required.

Example (after stage-1 training and checkpoint loading):
    model = TriSIF(cloud_predictor)
    prediction, loss, metrics = model(sky, satellite, history, clear_sky, masks, ghi)

The temporal--variable encoder follows TimeXer's global-token mechanism.
Module names follow trisif.tex; state-dict names differ from the research repo.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .cloud_predictor import SpatialConv
else:
    from cloud_predictor import SpatialConv


class Attention(nn.Module):
    """Preserve floor(D/heads) projections, including D=512 and heads=12."""

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.num_heads, self.depth = num_heads, d_model // num_heads
        inner = self.num_heads * self.depth
        self.query = nn.Linear(d_model, inner)
        self.key = nn.Linear(d_model, inner)
        self.value = nn.Linear(d_model, inner)
        self.output = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch = query.shape[0]
        q = self.query(query).reshape(batch, -1, self.num_heads, self.depth)
        k = self.key(key).reshape(batch, -1, self.num_heads, self.depth)
        v = self.value(value).reshape(batch, -1, self.num_heads, self.depth)
        scores = torch.einsum("blhe,bshe->bhls", q, k) / math.sqrt(self.depth)
        if mask is not None:
            scores = scores.masked_fill(mask, -torch.inf)
        weights = self.dropout(scores.softmax(-1))
        result = torch.einsum("bhls,bshd->blhd", weights, v)
        return self.output(result.reshape(batch, query.shape[1], -1))


class SpatialTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, value):
        normalized = self.norm1(value)
        value = value + self.attention(normalized, normalized, normalized,
                                       need_weights=False)[0]
        return value + self.ffn(self.norm2(value))


class CloudRepresentationRefinement(nn.Module):
    """ViT refinement and mean pooling of decoder features (Eq. cloud_refine)."""

    def __init__(self, image_size, in_channels=64, patch_size=32, d_model=512,
                 num_heads=8, d_ff=2048, depth=3, dropout=0.1):
        super().__init__()
        if patch_size < 1 or any(s % patch_size for s in image_size):
            raise ValueError("Cloud patch size must divide both feature-map dimensions")
        self.patch_size = patch_size
        count = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.refine = SpatialConv(in_channels, 32)
        self.patch_projection = nn.Linear(32 * patch_size**2, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position = nn.Parameter(torch.randn(1, count + 1, d_model))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            SpatialTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features):
        batch, steps = features.shape[:2]
        value = self.refine(features.flatten(0, 1))
        p = self.patch_size
        value = value.unfold(2, p, p).unfold(3, p, p)
        value = value.permute(0, 2, 3, 1, 4, 5).reshape(batch * steps, -1, 32 * p * p)
        value = self.patch_projection(value)
        cls = self.cls_token.expand(batch * steps, -1, -1)
        value = self.dropout(torch.cat((cls, value), dim=1) + self.position)
        for block in self.blocks:
            value = block(value)
        # The original ViT pools all tokens, including its learned CLS token.
        return self.norm(value).mean(1).reshape(batch, steps, -1)


class TemporalVariableLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = Attention(d_model, num_heads, dropout)
        self.cross_attention = Attention(d_model, num_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, variables):
        tokens = self.norm1(tokens + self.dropout(self.self_attention(tokens, tokens, tokens)))
        global_token = tokens[:, -1:]
        global_token = self.norm2(global_token + self.dropout(
            self.cross_attention(global_token, variables, variables)))
        tokens = torch.cat((tokens[:, :-1], global_token), dim=1)
        hidden = self.dropout(F.relu(self.conv1(tokens.transpose(1, 2))))
        hidden = self.dropout(self.conv2(hidden).transpose(1, 2))
        return self.norm3(tokens + hidden)


class TemporalVariableEncoder(nn.Module):
    """GHI temporal tokens and DNI/DHI/CSGHI variable tokens (Eqs. time_tokens--time_encoder)."""

    def __init__(self, seq_len=6, pred_len=6, d_model=512, num_heads=12,
                 d_ff=2048, depth=2, dropout=0.1):
        super().__init__()
        self.seq_len, self.pred_len = seq_len, pred_len
        self.value_embedding = nn.Linear(1, d_model, bias=False)
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        position = torch.arange(seq_len).float().unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2).float()* -(math.log(10000.) / d_model))
        angles = position * scale
        encoding = torch.zeros(seq_len, d_model)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, :d_model // 2].cos()
        self.register_buffer("position", encoding.unsqueeze(0))
        self.history_projection = nn.Linear(seq_len, d_model)
        self.csghi_projection = nn.Linear(seq_len + pred_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TemporalVariableLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.horizon_projection = nn.Linear(seq_len + 1, pred_len)

    def forward(self, history, csghi_future):
        batch = history.shape[0]
        clear_sky = torch.cat((history[:, :, 2:3], csghi_future), dim=1)
        variables = torch.cat((self.history_projection(history[:, :, :2].transpose(1, 2)),
                               self.csghi_projection(clear_sky.transpose(1, 2))), dim=1)
        variables = self.dropout(variables)
        tokens = self.value_embedding(history[:, :, 3:4]) + self.position
        tokens = self.dropout(torch.cat((tokens, self.global_token.expand(batch, -1, -1)), dim=1))
        for layer in self.layers:
            tokens = layer(tokens, variables)
        return self.horizon_projection(self.norm(tokens).transpose(1, 2)).transpose(1, 2)


class TemporallyLocalizedCrossAttention(nn.Module):
    """TLCA: each lead time retrieves its own and preceding w-1 cloud features."""

    def __init__(self, pred_len=6, d_model=512, num_heads=12, window_size=3,
                 dropout=0.1):
        super().__init__()
        self.attention = Attention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        steps = torch.arange(pred_len)
        distance = steps[:, None] - steps[None, :]
        self.register_buffer("mask", (distance < 0) | (distance >= window_size), persistent=False)

    def forward(self, timeseries, clouds):
        value = timeseries + self.dropout(self.attention(timeseries, clouds, clouds, self.mask))
        return self.output_norm(self.norm(value))


class IrradianceRegression(nn.Module):
    """Retain the original post-fusion self-attention, FFN and scalar readout."""

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attention = Attention(d_model, num_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, 1)

    def forward(self, value):
        value = self.norm1(value + self.dropout(self.attention(value, value, value)))
        hidden = self.dropout(F.relu(self.conv1(value.transpose(1, 2))))
        hidden = self.dropout(self.conv2(hidden).transpose(1, 2))
        return self.output(self.output_norm(self.norm2(value + hidden)))


class TriSIF(nn.Module):
    def __init__(self, cloud_predictor, seq_len_ts=6, d_model=512,
                 patch_size=32, img_num_heads=8, img_d_ff=2048, img_layers=3,
                 ts_num_heads=12, ts_d_ff=2048, ts_layers=2,
                 fusion_num_heads=12, fusion_d_ff=2048, window_size=3, dropout=0.1):
        super().__init__()

        self.pred_len = cloud_predictor.pred_len
        self.cloud_predictor = cloud_predictor.requires_grad_(False).eval()
        self.cloud_refinement = CloudRepresentationRefinement(
            cloud_predictor.ground_size, cloud_predictor.feature_channels,
            patch_size, d_model, img_num_heads, img_d_ff, img_layers, dropout)
        self.temporal_variable_encoder = TemporalVariableEncoder(
            seq_len_ts, self.pred_len, d_model, ts_num_heads, ts_d_ff, ts_layers, dropout)
        self.tlca = TemporallyLocalizedCrossAttention(
            self.pred_len, d_model, fusion_num_heads, window_size, dropout)
        self.regression = IrradianceRegression(d_model, fusion_num_heads, fusion_d_ff, dropout)

    def train(self, mode=True):
        super().train(mode)
        self.cloud_predictor.eval()
        return self

    def forward(self, ground_img_his, satellite_img_his, irradiance_ts_his,
                csghi_ts_future, sun_mask_future, ghi_ts_future=None):

        with torch.no_grad():
            features, _, _, _ = self.cloud_predictor(
                ground_img_his, satellite_img_his, sun_mask_future, return_features=True)
        clouds = self.cloud_refinement(features)
        timeseries = self.temporal_variable_encoder(irradiance_ts_his, csghi_ts_future)
        prediction = self.regression(self.tlca(timeseries, clouds))
        loss, meta = None, {}
        if ghi_ts_future is not None:
            if ghi_ts_future.shape != prediction.shape:
                raise ValueError("Future GHI must have shape [B,pred_len,1]")
            loss = F.mse_loss(prediction, ghi_ts_future)
            meta["mse_loss"] = loss.detach()
        return prediction, loss, meta
